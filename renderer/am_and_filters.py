import torch
from torch.nn import functional as F
import numpy as np

from torchinterp1d import Interp1d
import torch_dct as tdct

import renderer.util
from util.context import context
from model.scene_module import SceneModule

#####################
# Differentiable rendering functions related to
# amplitude modulation and spectral filtering
#####################

interp1d = Interp1d()


def amplitude_prep(gps, events, r, trimmer):
    """ Properly reshape latent variables for rendering """
    if gps["amplitude"].non_stationary:
        event_idxs = [el.event_idx for el in events]
        amplitude = gps["amplitude"].feature.y_render[:, :, event_idxs]
        amplitude = amplitude.transpose(1, 2).reshape(
            context.batch_size*len(events), -1
            ) 
    elif any([gps[k].non_stationary for k in gps.keys() if k != "amplitude"]) and (len(events) > 1):
        amplitude = torch.repeat_interleave(gps["amplitude"].feature.y, len(events), dim=0)
    else:
        amplitude = gps["amplitude"].feature.y
    amplitude, t_gp = trimmer.pad_gp(amplitude)

    if trimmer.trim_events and ("f0" not in gps.keys()):
        # Aperiodic sources: first time doing trim
        trimmed_amplitude, t_gp, trimmed_t_audio = trimmer.trim_to_longest_event(amplitude, events)
        return trimmed_amplitude
    elif trimmer.trim_events and trimmer.trim_this_iter:
        # Periodic type sources: use the indexes that we used to trim f0
        trimmed_amplitude = trimmer.trim_with_existing_idxs(amplitude)
        return trimmed_amplitude
    else:
        return amplitude


def filter_prep(gps, events, r):
    """ Properly reshape latent variables for rendering """
    spectrum = gps["spectrum"].feature.y
    temporal_event_dimension = any([gps[k].non_stationary for k in gps.keys() if k != "spectrum"])
    if gps["spectrum"].non_stationary:
        # spectrum: batch * (n_events * f)
        # spectrum_desired: (batch * n_events) * f
        # collapse batch and event dimension as if its repeat_interleave.
        spectrum = spectrum.view(context.batch_size*len(events), -1)
    elif temporal_event_dimension and (len(events) > 1):
        # batch * f --> (batch * n_events) * f
        spectrum = torch.repeat_interleave(spectrum, len(events), dim=0)
    return spectrum


class AmplitudeModulation():
    """ Amplitude modulation for an FM whistle excitation """

    def __init__(self, r, source_type="whistle"):
        self.rms_ref = r.rms_ref
        self.source_type = source_type
        self.c = r.rendering_constant

    def prep_latents(self, gps, events, r, trimmer):
        """ Properly shape latent variables for rendering """
        amplitude = amplitude_prep(gps, events, r, trimmer)
        if trimmer.trim_events and trimmer.trim_this_iter:
            trim_tools = trimmer.trim_rendering_tools()
            return amplitude, trim_tools["win_arr"]
        else:
            return amplitude, r.win_arr

    def __call__(self, excitation, gps, events, r, trimmer):
        """ Renders an amplitude modulated FM tone, from an FM tone"""
        filt, win_arr = self.prep_latents(gps, events, r, trimmer)
        energy = self.rms_ref * 10.0**(filt/20.) - self.c
        if win_arr.shape[0] == 1:
            AM_FM_tone = torch.einsum(
                "bc,ac,ab->ab", win_arr[0], energy, excitation
                )
        else:
            AM_FM_tone = torch.einsum(
                "abc,ac,ab->ab", win_arr, energy, excitation
                )
        return AM_FM_tone


class AMFilterHarmonic():
    """ Applies a filter and amplitude modulation to a harmonic excitation,
        where the filter does not move with f0.
        Less numerically stable than Panpipes
    """

    def __init__(self, r, source_type="harmonic"):
        self.c = r.rendering_constant
        self.rms_ref = r.rms_ref
        
    def prep_latents(self, gps, events, r, trimmer):
        """ Properly shape latent variables for rendering """
        amplitude = amplitude_prep(gps, events, r, trimmer)
        spectrum = filter_prep(gps, events, r)

        # win_arr: batch, sig_len, n_windows
        # filterbank: batch, sig_len, n_channels
        if trimmer.trim_events and trimmer.trim_this_iter:
            trim_tools = trimmer.trim_rendering_tools()
            self.win_arr = trim_tools["win_arr"]
            self.filterbank = trim_tools["filterbank"]
            self.bandwidthsHz = trim_tools["bandwidthsHz"]
            self.filt_freqs = trim_tools["filt_freqs"]
        else:
            self.win_arr = r.win_arr
            self.filterbank = r.filterbank
            self.bandwidthsHz = r.bandwidthsHz
            self.filt_freqs = r.filt_freqs

        if self.win_arr.shape[1] != self.filterbank.shape[1]:
            raise ValueError("Incorrect array lengths.")

        return amplitude, spectrum

    def level_to_energy(self, dB):
        """Convert level into amplitude"""
        E = self.rms_ref * 10.0 ** (dB/20.0) - self.c
        E = E.permute(0, 2, 1)  # batch channels windows
        return E

    def create_filter(self, excitation, E):
        """ Creates a filter to be applied multiplicatively
            in the frequency domain to each time frame (ie., window)
        
        Parameters
        ----------
        excitation: Tensor
            harmonic, fm modulated excitation
            Shape: batch, sig_len
        E: Tensor
            Amplitudes to be applied
            Shape: batch n_channels n_windows

        Returns 
        -------
        filt: Tensor 
            Scaled filterbank
            Shape: batch siglen nwindows
        windowed_excitation: Tensor
            Windowed excitation
            Shape: batch signlen nwindows
        """
        windowed_excitation = excitation[:, :, np.newaxis] * self.win_arr
        filt = torch.bmm(self.filterbank.expand(
            E.shape[0],
            self.filterbank.shape[1],
            self.filterbank.shape[2]
        ), E)
        return filt, windowed_excitation

    def apply_filter(self, filt, normed_excitation):
        """Applies spectral envelope created in `create_filter`"""
        fft_sig = tdct.dct(normed_excitation.permute(0, 2, 1))
        AM_FM_tone = tdct.idct((filt.permute(0, 2, 1) * fft_sig).sum(1))
        return AM_FM_tone

    def __call__(self, excitation, gps, events, r, trimmer, f0=None):
        """ Renders an amplitude modulated, 
            spectrally filtered harmonic tone
            from latent variables 
        """
        amplitude, spectrum = self.prep_latents(gps, events, r, trimmer)
        self.tf_grid = spectrum[:, np.newaxis, :] + amplitude[:, :, np.newaxis]
        E = self.level_to_energy(self.tf_grid)
        filt, normed_excitation = self.create_filter(excitation, E)
        AM_FM_tone = self.apply_filter(filt, normed_excitation)
        return AM_FM_tone


class Panpipes(AMFilterHarmonic):
    """ Applies a filter and amplitude modulation to a harmonic excitation,
        where the filter shifts with f0. Preferred harmonic renderer.
    """

    def __init__(self, r):
        super().__init__(r)
        if r.steps["scale"] == "ERB":
            self.freq_to_scale = renderer.util.torch_freq_to_ERB
            self.scale_to_freq = renderer.util.torch_ERB_to_freq
        elif r.steps["scale"] == "octave"
            self.freq_to_scale = renderer.util.torch_freq_to_octave
            self.scale_to_freq = renderer.util.torch_octave_to_freq

    def shift_E(self, E, scaled_f0, V=0):
        """ Shifts energy_grid according to f0, so the filter
            moves differentiably with the fundamental
        
        Parameters
        ----------
        E: Tensor
            Amplitudes to be applied
            shape: (batch_size, n_channels, n_windows)
        f0: Tensor
            shape: (batch_size, n_windows)
        
        Returns 
        -------
        shifted_E: Tensor
            Spectrum shifted to move with f0
            shape: (batch_size, n_chanels, n_windows)    
        """
        n_batch, n_channels, n_windows = E.shape
        device = E.device

        # By how many channels do we want to shift the spectrum
        lo_lim_freq = context.config["renderer"]["lo_lim_freq"]
        n_channels = len(renderer.util.get_event_gp_freqs(context.audio_sr, context.config['renderer']['steps']))
        channel_width = (renderer.util.freq_to_ERB(context.audio_sr/2. - 1.) - renderer.util.freq_to_ERB(lo_lim_freq))/(n_channels+1)
        shift_by = (scaled_f0 - renderer.util.freq_to_ERB(lo_lim_freq)) / channel_width - 1
   
        # Unroll
        E_unroll = E.permute(0, 2, 1).reshape(n_batch * n_windows, n_channels)
        shift_by = shift_by.reshape(n_batch * n_windows)

        # Shift
        x_pad = torch.cat([
            torch.tensor([-n_channels, -1], device=device),
            torch.arange(n_channels, device=device),
            torch.tensor([n_channels, 2*n_channels], device=device)
        ])[None, :].expand(n_batch*n_windows, n_channels+4)
        E_pad = F.pad(E_unroll, [2, 2], value=V)
        xnew = torch.arange(n_channels, device=device)[None, :] - shift_by[:, None]
        shifted_E = interp1d(x_pad, E_pad, xnew )

        # Reshape
        shifted_E = shifted_E.reshape(n_batch, n_windows, n_channels).permute(0, 2, 1)

        return shifted_E

    def __call__(self, excitation, gps, events, r, trimmer, f0):
        """ Renders an amplitude modulated, spectrally filtered harmonic tone from latent variables """
        amplitude, spectrum = self.prep_latents(gps, events, r, trimmer)
        # Applied modulation is the outer product of the spectrum and amplitude vectors
        # Create extended_spectrum, f0_for_shift, tf_grid attributes for neural network data generation
        self.extended_spectrum = spectrum[:, :, np.newaxis].expand(-1, -1, amplitude.shape[1])
        self.f0_for_shift = f0
        self.tf_grid = spectrum[:, np.newaxis, :] + amplitude[:, :, np.newaxis]
        E = self.level_to_energy(self.tf_grid)
        shifted_E = self.shift_E(E, f0)
        filt, normed_excitation = self.create_filter(excitation, shifted_E)
        #AM_FM_tone: shape: (batch_size, sig_len)
        AM_FM_tone = self.apply_filter(filt, normed_excitation)
        return AM_FM_tone


class AMFilterNoise(SceneModule):
    """ Applies a filter and amplitude modulation to a noise excitation. """

    def __init__(self, r, source_type="noise"):
        super().__init__() 
        self.c = r.rendering_constant
        self.rms_ref = r.rms_ref
        self.tstep = r.steps["t"]

    def prep_latents(self, excitation, gps, events, r, trimmer):
        """ Properly shape latent variables for rendering """
        amplitude = amplitude_prep(gps, events, r, trimmer)
        spectrum = filter_prep(gps, events, r)

        if trimmer.trim_events and trimmer.trim_this_iter:
            trim_tools = trimmer.trim_rendering_tools(excitation=excitation)
            trimmed_excitation = trim_tools["excitation"]
            self.win_arr = trim_tools["win_arr"]
            self.filterbank = trim_tools["filterbank"]
            self.bandwidthsHz = trim_tools["bandwidthsHz"]
        else:
            self.win_arr = r.win_arr
            self.filterbank = r.filterbank
            self.bandwidthsHz = r.bandwidthsHz
            trimmed_excitation = excitation[None, :]

        if self.win_arr.shape[1] != self.filterbank.shape[1]:
            raise ValueError("Incorrect array lengths.")

        return amplitude, spectrum, trimmed_excitation

    def SL_to_energy_ratio(self, spectrum_level):
        """Convert spectrum level into amplitude"""
        E = 10.0**( (spectrum_level + 10*torch.log10(self.bandwidthsHz[:, 0, :]))/20.0 ) - self.c
        return E

    def level_to_energy_ratio(self, dB):
        """Convert level into amplitude"""
        E = 10.0 ** (dB/20.0) - self.c
        return E

    def __call__(self, excitation, gps, events, r, trimmer):
        """ Renders an amplitude modulated, spectrally filtered noise from latent variables """
        excitation = torch.Tensor(excitation).to(r.win_arr.device)
        amplitude, spectrum, trimmed_excitation = self.prep_latents(excitation, gps, events, r, trimmer)
        A_energy_ratio = self.level_to_energy_ratio(amplitude)
        S_energy_ratio = self.SL_to_energy_ratio(spectrum)
        self.tf_grid = 20.0*torch.log10(S_energy_ratio[:, np.newaxis, :]+self.c)  + amplitude[:, :, np.newaxis]

        mod_excitation = torch.einsum("bsw,bw,bs->bs", self.win_arr, A_energy_ratio, trimmed_excitation)
        filt = (self.filterbank * S_energy_ratio[:, None, :]).sum(2) 
        fft_sig = tdct.dct(mod_excitation) 
        am_fm_noise = self.rms_ref * tdct.idct(filt * fft_sig)

        return am_fm_noise
