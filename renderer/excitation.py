import numpy as np
import torch
import torch.utils.checkpoint
from scipy.fftpack import ifft

from torchinterp1d import Interp1d

import renderer.util
from util.context import context
from util import softbound
from model.scene_module import SceneModule


class Periodic(SceneModule):

    def __init__(self, r, source_type, source_colour="pink", attenuation_constant=None, n_harmonics=200):
        super().__init__()

        # Rendering Variables
        self.hi_lim_freq = self.audio_sr/2. - 1
        self.lo_lim_freq = 30.
        self.hi_ramp_hz = 1000
        self.lo_ramp_hz = 25.
        self.lo_lim_erb = renderer.util.freq_to_ERB(self.lo_lim_freq)
        self.hi_lim_erb = renderer.util.freq_to_ERB(self.hi_lim_freq)
        self.erb_ramp = renderer.util.freq_to_ERB(self.lo_ramp_hz + self.lo_lim_freq) - renderer.util.freq_to_ERB(self.lo_lim_freq)

        self.tstep = r.steps["t"]
        
        self.source_type = source_type 
        self.source_colour = source_colour
        if not (self.source_colour == "pink" or self.source_colour == "attenuated_pink"):
            raise Exception("Please set source_colour to pink or attenuated_pink")
        self.attenuation_constant = attenuation_constant
        if self.source_type == "harmonic":
            self.register_buffer("harmonic_idxs", torch.arange(1,n_harmonics)[np.newaxis,:])
        elif self.source_type == "whistle":
            self.register_buffer("harmonic_idxs", torch.Tensor([1])[np.newaxis,:])

        if r.steps["scale"] == "ERB":
            self.freq_to_scale = renderer.util.torch_freq_to_ERB 
            self.scale_to_freq = renderer.util.torch_ERB_to_freq
        elif r.steps["scale"] == "octave":
            self.freq_to_scale = renderer.util.torch_freq_to_octave
            self.scale_to_freq = renderer.util.torch_octave_to_freq

        self.c = r.rendering_constant

    def prep_latents(self, gps, events, r, trimmer):

        trimmer.window_timing(n_events=len(events))
        # If the f0 in particular is non-stationary
        if gps["f0"].non_stationary:
            event_idxs = [el.event_idx for el in events]
            f0 = gps["f0"].feature.y_render[:, :, event_idxs]
            f0 = f0.transpose(1, 2).reshape(context.batch_size*len(events), -1)
        elif any([gps[k].non_stationary for k in gps.keys() if k != "f0"]) and (len(events) > 1):
            f0 = torch.repeat_interleave(gps["f0"].feature.y, len(events), dim=0)
        else:
            f0 = gps["f0"].feature.y  # batch_size, timepoints
        f0, t_gp = trimmer.pad_gp(f0)

        if trimmer.trim_events:
            rendered_f0, t_gp, trimmed_t_audio = trimmer.trim_to_longest_event(f0, events)
            return t_gp, rendered_f0, trimmed_t_audio
        else:
            rendered_f0 = f0
            return t_gp, rendered_f0, r.new_timepoints

    def normalize(self, wave, axis, keepdims=True):
        rms = torch.sqrt(torch.mean(wave**2., axis, keepdims=keepdims) + self.c)
        normalized_wave = wave / (rms + self.c)
        return normalized_wave

    def components(self, old_timepoints, erbf0, erbf, new_timepoints):

        # Put harmonics in batch dimension - shape = (batch*n_elem*nharmonics, t)
        hbatched_erbf = erbf.reshape(erbf.shape[0]*erbf.shape[1], erbf.shape[2])

        # No event cov or trim, only matching harmonics if they are there.
        if old_timepoints.shape[0] == 1:
            old_timepoints = old_timepoints.repeat(erbf.shape[0], 1)
        # Otherwise, old_timepoints should already share batchdim= (batch_size) or batchdim=(batch_size*n_elems)
        # No trimming. otherwise would have batchdim=batch_size*n_elem
        if len(new_timepoints.shape) == 1:
            new_timepoints = new_timepoints[np.newaxis, :]
            new_timepoints = new_timepoints.expand(erbf.shape[0], new_timepoints.shape[1])

        # repeat_interleave is appropriate for matching the erbf.shape[0]*erbf.shape[1] in the line above
        hbatched_old_timepoints = old_timepoints if erbf.shape[1] == 1 else torch.repeat_interleave(old_timepoints, erbf.shape[1], dim=0)
        hbatched_new_timepoints = new_timepoints if erbf.shape[1] == 1 else torch.repeat_interleave(new_timepoints, erbf.shape[1], dim=0)

        erbfin = Interp1d()(hbatched_old_timepoints, hbatched_erbf, hbatched_new_timepoints)
        erbfin = erbfin.reshape(erbf.shape[0], erbf.shape[1], new_timepoints.shape[1])

        # Input dimensions: mini-batch x channels x [optional depth] x [optional height] x width.
        # For us, channels wil be number of frequencies
        # If generating a tone, shape will be batch x 1 x time
        # If generated a harmonic, shape will be batch x n_harmonics x time
        unclamped_f = self.scale_to_freq(erbfin)
        f = unclamped_f.clamp(min=self.lo_lim_freq, max=self.hi_lim_freq)

        erbf0in = Interp1d()(old_timepoints, erbf0, new_timepoints)
        unclamped_f0 = self.scale_to_freq(erbf0in)
        f0 = unclamped_f0.clamp(min=self.lo_lim_freq, max=self.hi_lim_freq)

        if self.source_colour == "pink":
            amplitude_ratio = torch.sqrt(f0[:, None, :]/f)  # batch x n_harmonics x time
        elif self.source_colour == "attenuated_pink":
            amplitude_ratio = torch.sqrt(f0[:, None, :]/f * torch.exp(-self.attenuation_constant*(f-f0[:, None, :])/f0[:, None, :]))
        else:
            amplitude_ratio = 1.0

        harmonic_components = torch.sin(2*np.pi*torch.cumsum(f*(1/self.audio_sr),2)) 
        harmonic_components = self.normalize(harmonic_components, 2) * amplitude_ratio

        return harmonic_components, unclamped_f

    def forward(self, gps, events, r, trimmer):

        old_timepoints, erbf0, new_timepoints = self.prep_latents(gps, events, r, trimmer)

        # Sum up the FM harmonics
        # The GP can sometimes sample negative values for erbf0 if the variance is high
        # This will lead to a nan when converting between scale & freq
        # ERB_to_freq(0) = 0, so the sound will be "zero'd" out in the "where" lines below, because 0 < lo_lim_freq
        erbf0 = softbound(erbf0, self.lo_lim_erb, self.hi_lim_erb, self.erb_ramp)
        self.rendered_f0 = erbf0
        f0 = self.scale_to_freq(erbf0)[:, np.newaxis, :]  # batch, :, timepoints
        f = f0 * self.harmonic_idxs[:, :, np.newaxis]
        erbf = self.freq_to_scale(f)  # batch nharmonics timepoints+2

        # Get harmonic components and threshold appropriately
        harmonics, unclamped_f = torch.utils.checkpoint.checkpoint(
            self.components, old_timepoints, erbf0, erbf, new_timepoints
            )
        scale = (1 - (unclamped_f - self.hi_lim_freq + self.hi_ramp_hz)/self.hi_ramp_hz).clamp(min=0, max=1) * \
                ((unclamped_f - self.lo_lim_freq)/self.lo_ramp_hz).clamp(min=0, max=1)
        harmonics = harmonics * scale

        # Create overall frequency-modulated excitation
        excitation_wave = torch.sum(harmonics, 1) #batch, time
        self.excitation_wave = self.normalize(excitation_wave, 1)

        return self.excitation_wave


class Aperiodic(SceneModule):

    def __init__(self, r, renderer_options):
        super().__init__()
        self.source_colour = renderer_options["source_colour"]
        if self.source_colour == "pink":
            self.beta = 1
        elif self.source_colour == "white":
            self.beta = 0
        elif self.source_colour == "red":
            self.beta = 2
        else:
            self.beta = renderer_options["beta"]

        self.band = [0, int(self.audio_sr/2.)] if len(renderer_options["band"]) == 0 else renderer_options["band"]  # in Hz

        self.c = r.rendering_constant

    def prep_latents(self, events, trimmer):
        trimmer.window_timing(n_events=len(events))

    def colour_spectrum(self, spectrum, binFactor, n):
        # For pink noise, divide each component of the spectrum by f^(1/2).
        # Since power is the square of amplitude,
        # this means we divide the power at each component by f
        # if you want to include arbitrary beta parameter, use instead np.power(pinkWeights,beta/2.)
        # beta = 0: white noise,  beta = 2: red (Brownian) noise
        w = binFactor*np.arange(1, n+1)
        if self.source_colour == "pink":
            coloured_spectrum = np.divide(spectrum, np.sqrt(w))
        else:
            coloured_spectrum = np.divide(spectrum, np.power(w,self.beta/2.))
        return coloured_spectrum

    def __call__(self, gps, events, r, trimmer):
        """ Generates pink noise: 
        power spectral density (energy or power per frequency interval) 
        is inversely proportional to the frequency of the signal. 
        In pink noise, each octave (halving/doubling in frequency) 
        carries an equal amount of noise energy.
        """

        self.prep_latents(events, trimmer)

        duration = r.excitation_duration
        n = int(np.round(duration*self.audio_sr))
        binFactor = n/(1.*self.audio_sr)  # move between frequencies and bin indexes
        lowPassBin = int(np.round(self.band[0]*binFactor))
        highPassBin = int(np.round(self.band[1]*binFactor))

        x_real = np.zeros(n)
        x_imag = np.zeros(n)
        drng = np.random.default_rng(13081992)
        x_real[lowPassBin:highPassBin+1]=drng.standard_normal(1+highPassBin-lowPassBin)
        x_imag[lowPassBin:highPassBin+1]=drng.standard_normal(1+highPassBin-lowPassBin)
        spectrum = x_real + x_imag*1j

        coloured_spectrum = self.colour_spectrum(spectrum, binFactor, n)
        noise = np.real(ifft(coloured_spectrum))
        noise = noise[0:n]
        noise /= (np.sqrt(np.mean(np.square(noise))) + self.c)

        return noise


class Ramps(SceneModule):

    def __init__(self, batched_events):
        super().__init__()
        self.batched_events = batched_events
        self.amplitude_threshold = 1e-12  # necessary for gammatonegram gradient stability

    def create(self, events, timepoints):
        """
        Inputs:
            timepoints: n_timepoints

        Returns:
            R: batch, n_timepoints
        """

        last_offset = 0
        Rs = []
        for i in range(len(events)):
            if self.batched_events:
                R = events[i].ramps(timepoints)[:, None, :]
            else:
                R = events[i].ramps(timepoints)
            Rs.append(R)
        
        if self.batched_events:
            R = torch.cat(Rs, dim=1)  # multiple events
        else:
            R = torch.sum(torch.stack(Rs), 0)
        
        return R

    def create_trimmed_ramps(self, events, timepoints):
        """
        Inputs:
            timepoints: (batch*n_events), n_timepoints

        Returns:
            R: (batch*n_events), n_timepoints
        """

        timepoints = timepoints.reshape(context.batch_size, len(events), -1)
        Rs = []
        for i in range(len(events)):
            if self.batched_events:
                R = events[i].ramps(timepoints[:, i, :])[:, None, :]
            else:
                R = events[i].ramps(timepoints[:, i, :])
            Rs.append(R)
        
        if self.batched_events:
            R = torch.cat(Rs, dim=1)  # multiple events
        else:
            R = torch.sum(torch.stack(Rs), 0)
        
        return R

    def __call__(self, x, events, r, trimmer, dream=False):
        if trimmer.trim_events and trimmer.trim_this_iter:
            R = self.create_trimmed_ramps(events, trimmer.audio_timepoints)
            R[R < self.amplitude_threshold] = 0.0
            if self.batched_events:
                trimmed_event_waves = x.reshape(context.batch_size, len(events), -1) * R
            else:
                trimmed_event_waves = x * R
            if dream:  # for dataset generation
                padded_event_waves = trimmer.pad_to_excitation(
                    trimmed_event_waves.reshape(context.batch_size*len(events), -1)
                    )
                event_waves = trimmer.trim_to_scene(padded_event_waves).reshape(context.batch_size, len(events), -1)
                source_wave = torch.sum(event_waves, 1)
            else:
                excitation_wave = trimmer.combine_into_source(trimmed_event_waves)
                source_wave = trimmer.trim_to_scene(excitation_wave)
                event_waves = None
            
        else:
            # x: non-stationary=(batch*len(events), data_len); stationary=(batch, data_len)
            x = trimmer.trim_to_scene(x)
            R = self.create(events, r.scene_timepoints)
            R[R < self.amplitude_threshold] = 0.0
            if self.batched_events:
                x = x.reshape(context.batch_size, len(events), -1)
                event_waves = x * R
                source_wave = torch.sum(event_waves, 1)
            else:
                source_wave = x * R
                event_waves = None

        return source_wave, event_waves
