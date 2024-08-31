import torch
import numpy as np
import renderer.util as dr

from util.context import context
from model.scene_module import SceneModule


class SceneTrimmer(SceneModule):
    """ Keeps rendering memory usage manageable by
        trimming sounds to less than the scene duration
    """

    def __init__(self, batched_events, r, source_type):
        super().__init__()
        self.source_type = source_type
        self.r = r
        self.steps = r.steps
        self.dt_y = r.steps["t"]
        self.win_pad = r.win_pad
        self.elem_pad = 2
        self.batched_events = batched_events
        self.trim_events = False

    @property
    def t_audio(self):
        return self.r.new_timepoints

    @property
    def dt_audio(self):
        return 1./self.audio_sr

    def window_timing(self, n_events):
        # If there are any non-stationary kernels in this source
        if self.batched_events:
            self.gp_t_rs = self.r.gp_t[:, :, 0].expand(context.batch_size*n_events, -1)
        else:
            self.gp_t_rs = self.r.gp_t[:, :, 0]

    def pad_gp(self, y):
        padded_y = torch.cat((
            y[:, 0, None].expand(y.shape[0], self.win_pad),
            y,
            y[:, -1, None].expand(y.shape[0], self.win_pad)
        ), dim=1)
        t_before = torch.linspace(
            -self.dt_y*self.win_pad,
            self.gp_t_rs[0,0] - self.dt_y,
            self.win_pad, device=y.device
        )[None, :]
        t_before = t_before.expand(self.gp_t_rs.shape[0],t_before.shape[1])
        t_after = torch.linspace(
            self.gp_t_rs[0,-1] + self.dt_y,
            self.gp_t_rs[0,-1] + self.win_pad*self.dt_y,
            self.win_pad, device=y.device
            )[None, :]
        t_after = t_after.expand(self.gp_t_rs.shape[0],t_after.shape[1])
        self.t_y = torch.cat(
            (t_before,self.gp_t_rs,t_after), dim=1
            )  # t_win 
        return padded_y, self.t_y

    def trim_to_scene(self, x):
        
        n_samples = int(np.round(self.audio_sr*self.scene_duration))
        start_point = max(0, int(np.floor((x.shape[1] - n_samples)/2))-1)
        x = x[:,start_point:start_point+n_samples]

        return x

class EventTrimmer(SceneTrimmer):

    def __init__(self, batched_events, r, source_type):
        super().__init__(batched_events, r, source_type)
        self.trim_events = True

    def trim_to_longest_event(self, y, events):

        t_y = self.t_y
        self.original_y_len = self.t_y.shape[1]
        onsets = torch.stack([e.onset.timepoint for e in events], -1)
        offsets = torch.stack([e.offset.timepoint for e in events], -1)
        # You could have onsets which go way beyond scene,
        # in which case it doesn't matter to render them anyway.
        section_start_time = (onsets.view(-1, 1) - context.renderer["ramp_duration"] - self.elem_pad*self.dt_y).clamp(min=t_y[0,0].item(), max=t_y[0, -1])
        n_shift_yidxs = torch.searchsorted(t_y, section_start_time)
        t_y_start = t_y.gather(1, n_shift_yidxs)

        # Total duration of sound we need to render, in seconds (length of longest event + buffer), and aligned to dt_y
        _unpadded_section_durations = (torch.minimum(offsets, t_y[0,-1]) + context.renderer["ramp_duration"]) - t_y_start.reshape(onsets.shape)
        padded_section_durations = (torch.ceil(_unpadded_section_durations/self.dt_y) + self.elem_pad) * self.dt_y
        max_duration = padded_section_durations.max().item()
        if max_duration > self.scene_duration:
            # With padding, max_duration can get longer than
            # scene_duration, in which case don't bother trimming!
            self.window_idxs = None
            self.audio_timepoints = self.t_audio
            self.audio_idxs = None
            self.L = None
            self.trim_this_iter = False
            return y, t_y, self.audio_timepoints

        # Shift events in y so that they all start at time 0
        max_duration_yidx = int(np.round(max_duration/self.dt_y))
        y_idxs = torch.arange(max_duration_yidx, device=onsets.device)[None, :] + n_shift_yidxs.long()
        # Create y_extend, in case there's a long duration but a short event
        # at the end -- you need to be able to index off the end
        y_extend = torch.cat(
            [y, y[:, -1, None].repeat(1, max_duration_yidx)], axis=1
            )
        y_trim = y_extend.gather(1, y_idxs)
        ty_extend = torch.cat([t_y, t_y[:, -1, None].repeat(1, max_duration_yidx)], axis=1)
        ty_trim = ty_extend.gather(1, y_idxs)

        # Shift audio timepoints
        max_duration_aidx = int(np.round((max_duration-self.dt_y)/self.dt_audio))
        section_start_aidx = torch.searchsorted(self.t_audio, section_start_time)
        ta_idxs = torch.arange(
            max_duration_aidx,
            device=onsets.device
        )[None, :] + section_start_aidx
        extension_for_t_audio = torch.arange(1, 1+max_duration_aidx, device=self.t_audio.device) * 1./self.audio_sr + self.t_audio[-1]
        t_audio_extend = torch.cat([self.t_audio, extension_for_t_audio], axis=0)
        ta_trim = t_audio_extend[None, :].expand(ta_idxs.shape[0], t_audio_extend.shape[0]).gather(1, ta_idxs)
        self.L = {
            "max_y_len": max_duration_yidx,
            "max_audio_len": max_duration_aidx,
            "excitation_len": self.t_audio.shape[0],
            "extend_len": t_audio_extend.shape[0],
            "y_extend_len":y_extend.shape[1],
            "y_orig_shape":y.shape
            }

        self.ty_trim = ty_trim
        self.window_idxs = y_idxs
        self.audio_timepoints = ta_trim
        self.audio_idxs = ta_idxs
        self.trim_this_iter = True

        return y_trim, ty_trim, ta_trim

    def trim_with_existing_idxs(self, y):
        # Create y_extend, in case there's a long duration but a short event at the end
        # you need to be able to index off the end
        y_extend = torch.cat([y, y[:, -1, None].repeat(1, self.L["max_y_len"])], axis=1) 
        y_trim = y_extend.gather(1, self.window_idxs)
        return y_trim

    def trim_rendering_tools(self, subbands=None, excitation=None):

        win_arr = torch.cat((
            self.r.win_arr, self.r.win_arr[:, -1, None, :].repeat(1, self.L["max_audio_len"], 1)
        ), dim=1)
        win_arr = win_arr.expand(
            self.audio_idxs.shape[0], win_arr.shape[1], win_arr.shape[2]
            )
        win_arr = win_arr.gather(
            1,
            self.audio_idxs[:, :, None].expand(
                self.audio_idxs.shape[0], self.audio_idxs.shape[1], win_arr.shape[2]
                )
            )
        # We're going to use padded amplitude to do this so we don't get edge
        # artifacts - get rid of the win_arr[1:-1] indexing
        win_arr = torch.cat((
            win_arr, win_arr[:, :, -1, None].repeat(1, 1, self.L["max_y_len"])
            ), dim=2)
        win_arr = win_arr.gather(
            2,
            self.window_idxs[:,None,:].expand(
                self.window_idxs.shape[0], win_arr.shape[1], self.window_idxs.shape[1])
            )

        if self.source_type == "noise" or self.source_type == "harmonic":
            filterbank, bandwidthsHz, filt_freq_cutoffs, filt_freqs = dr.make_overlapping_freq_windows(win_arr.shape[2], self.steps, self.audio_sr)
            filterbank = torch.Tensor(filterbank[np.newaxis, :, :]).to(win_arr.device)
            bandwidthsHz = torch.Tensor(bandwidthsHz[np.newaxis, :, :]).to(win_arr.device)

        if self.source_type == "noise":
            excitation = torch.cat((
                excitation, excitation[-1, None].repeat(self.L["max_audio_len"])
                ), dim=0)
            excitation = excitation[None, :].expand(
                self.audio_idxs.shape[0], excitation.shape[0]
                )
            excitation = excitation.gather(1, self.audio_idxs)

        if self.source_type == "whistle":
            return {"win_arr": win_arr}
        elif self.source_type == "harmonic":
            return {
                "win_arr": win_arr,
                "filterbank": filterbank,
                "bandwidthsHz": bandwidthsHz,
                "filt_freqs":torch.Tensor(filt_freqs).to(win_arr.device)
                }
        elif self.source_type == "noise":
            return {
                "win_arr": win_arr,
                "filterbank": filterbank,
                "bandwidthsHz": bandwidthsHz,
                "excitation": excitation
                }

    def pad_to_excitation(self, trimmed_sound):
        """
        Inputs:
            trimmed_sound: [batch * n_events, n_timepoints]

        full_shapes = [t_audio.shape[0], t_audio_extend.shape[0]]
        """
        # Shift all 'rendered' events back to their correct position
        full_sound = torch.zeros([
            trimmed_sound.shape[0], self.L["extend_len"]
            ], device=trimmed_sound.device)
        full_sound.scatter_(1, self.audio_idxs, trimmed_sound)
        full_sound = full_sound[:, :self.L["excitation_len"]]
        return full_sound

    def combine_into_source(self, trimmed_events):
        """
        Inputs:
            trimmed_events: (batch_size, n_events, n_timepoints)

        Returns:
            full_sound: (batch_size, n_timepoints)
        """ 

        # self.audio_idxs: [batch*n_events, n_timepoints]
        z = torch.zeros([
            trimmed_events.shape[0], self.L["extend_len"]
            ], device=trimmed_events.device)
        for i in range(trimmed_events.shape[1]):
            idx = self.audio_idxs.reshape(*trimmed_events.shape)[:, i, :]  # batch, n_timepoints
            src = trimmed_events[:, i, :]  # batch, n_timepoints
            z.scatter_add_(1, idx, src)
        full_sound = z[:, :self.L["excitation_len"]]
        return full_sound

    def pad_to_latent(self, trimmed_events, events, V=0.0):
        """
        trimmed_events = (Batch_size, time_dimension, ...)
        full_shapes = [t_audio.shape[0], t_audio_extend.shape[0]]
        """
        # Shift all latent events back to their correct position
        if len(trimmed_events.shape) == 3:
            z_shape = (
                trimmed_events.shape[0],
                self.L["y_extend_len"],
                trimmed_events.shape[2]
                )
        else:
            z_shape = (trimmed_events.shape[0], self.L["y_extend_len"])

        z = torch.full(z_shape, V, device=trimmed_events.device)

        if len(trimmed_events.shape) == 3:
            I = self.window_idxs[:, :, None] + torch.zeros([1, 1, trimmed_events.shape[2]], device=trimmed_events.device)
        else:
            I = self.window_idxs

        scattered_latents = z.scatter(1, I.long(), trimmed_events)

        onsets = torch.stack([e.onset.timepoint for e in events], -1)
        offsets = torch.stack([e.offset.timepoint for e in events], -1)
        event_active = ((self.gp_t_rs >= onsets.T)*(self.gp_t_rs <= offsets.T))
        if len(trimmed_events.shape) == 3:
            full_latents = scattered_latents[:, :self.L["y_orig_shape"][1], :]
            masked_full_latents = full_latents[:, 1:-1, :] * event_active[:, :, None]
            masked_full_latents[~event_active[:, :, None].expand(-1, -1, masked_full_latents.shape[2])] = V
        else:
            full_latents = scattered_latents[:, :self.L["y_orig_shape"][1]]
            masked_full_latents = full_latents[:, 1:-1] * event_active
            masked_full_latents[~event_active] = V

        return masked_full_latents

