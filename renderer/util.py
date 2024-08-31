import numpy as np
import scipy
from scipy.fftpack import idct
import scipy.interpolate

import torch

from util.context import context

# NUMPY -- differentiation not needed

rendering_log_const = 1e-12


def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)


def freq_to_ERB(freq):
    return 9.265*np.log(1 + freq/(24.7*9.265))


def ERB_to_freq(ERB):
    return 24.7*9.265*(np.exp(ERB/9.265) - 1)


def freq_to_octave(freq):
    return np.log2(freq)


def octave_to_freq(ve):
    return 2.0**ve


def torch_freq_to_ERB(freq):
    return 9.265*torch.log(1 + freq/(24.7*9.265))


def torch_ERB_to_freq(ERB):
    return 24.7*9.265*(torch.exp(ERB/9.265) - 1)


def torch_freq_to_octave(freq):
    return torch.log2(freq)


def torch_octave_to_freq(ve):
    return 2.0**ve


def get_event_gp_times(onset, offset, tstep):
    return [np.round(t, decimals=3) for t in np.arange(start=0, stop=offset, step=tstep) if t >= onset]


def get_event_gp_freqs(audio_sr, steps, lo_lim_freq=None):

    scale = steps["scale"]
    fstep = steps["f"]

    if scale == "ERB":
        freq_to_scale = freq_to_ERB
    elif scale == "octave":
        freq_to_scale = freq_to_octave

    lo = freq_to_scale(context.renderer["lo_lim_freq"] if lo_lim_freq is None else lo_lim_freq)
    hi = freq_to_scale(np.floor(audio_sr/2.) - 1.)

    return np.round(np.arange(start=lo, stop=hi, step=fstep), decimals=3)

# RAMPS
# Smooth onsets and offsets to reduce artifacts
def hann_ramp(sr, ramp_duration = 0.010):
    # Make ramps
    t = np.arange(start=0, stop=ramp_duration, step=1/sr)
    off_ramp = 0.5*(1. + np.cos( (np.pi/ramp_duration)*t ))
    return off_ramp


def ramp_sound(s, sr, ramp_duration=0.010):
    t = np.arange(start=0, stop=ramp_duration, step=1/sr)
    off_ramp = 0.5*(1. + np.cos( (np.pi/ramp_duration)*t ))
    s[-len(off_ramp):] *= off_ramp
    s[:len(off_ramp)] *= off_ramp[::-1]
    return s


# WINDOWS
def make_windows_rcos_flat_no_ends(filt_len, tstep, sr):
    """ Creates windows in order to segment and differentially alter the excitation signal
        From Josh McDermott's Sound Texture Synthesis code (MATLAB)
    """

    desired_win_half_length_smp = int(np.floor(tstep*sr))
    signal_length_smp = int((filt_len - 1)*desired_win_half_length_smp)
    N = filt_len

    ramp_prop = 0.5
    ramp_length_smp = int(np.floor(signal_length_smp/(N-1)))
    flat_length_smp = 0

    win_length_smp = flat_length_smp + 2*ramp_length_smp
    windows = np.zeros((signal_length_smp, N))
    windows[:flat_length_smp+1 ,0] = 2
    ramp_off = np.cos(np.arange(1,ramp_length_smp)/ramp_length_smp*np.pi)+1
    ramp_on = np.cos(np.arange(-ramp_length_smp, 1)/ramp_length_smp*np.pi)+1
    windows[1+flat_length_smp:flat_length_smp+ramp_length_smp, 0] = ramp_off
    start_pt = flat_length_smp
    for n_idx in range(1, N-1):
        windows[start_pt:start_pt+ramp_length_smp+1, n_idx] = ramp_on
        windows[start_pt+ramp_length_smp+1:start_pt+ramp_length_smp+flat_length_smp+1,n_idx] = 2
        windows[start_pt+ramp_length_smp+flat_length_smp+1:start_pt+2*ramp_length_smp+flat_length_smp,n_idx] = ramp_off
        start_pt = start_pt + flat_length_smp+ramp_length_smp

    windows[start_pt:start_pt+ramp_length_smp,N-1] = ramp_on[:-1]
    windows[start_pt+ramp_length_smp+1:signal_length_smp,N-1] = 2
    windows /= 2

    return windows, signal_length_smp


def get_bandwidthsHz(audio_sr, config):

    max_freq = audio_sr/2. - 1
    steps = config["renderer"]["steps"]

    if steps["scale"] == "ERB":
        freq_to_scale = freq_to_ERB
        scale_to_freq = ERB_to_freq
    elif steps["scale"] == "octave":
        freq_to_scale = freq_to_octave
        scale_to_freq = octave_to_freq

    loscale = freq_to_scale(config["renderer"]["lo_lim_freq"])
    hiscale = freq_to_scale(max_freq)

    n_channels = int(len(get_event_gp_freqs(audio_sr, steps,lo_lim_freq=config["renderer"]["lo_lim_freq"])) + 2)
    scale_cutoffs_1D = np.linspace(loscale, stop=hiscale, num=n_channels)
    scale_cutoffs = np.array([[scale_cutoffs_1D[i], scale_cutoffs_1D[i+2]] for i in range(n_channels - 2)])  # 50% overlap
    freq_cutoffs = scale_to_freq(scale_cutoffs)
    rfc = freq_cutoffs[::-1, :]
    bandwidthsHz = rfc[:, 1] - rfc[:, 0]

    return bandwidthsHz


def freq_win_shape(x):
    return np.cos(x / 2.0)


def make_overlapping_freq_windows(filt_len, steps, audio_sr):
    """Computes array of log spaced cos-shaped windows 

    Input
    -----
    filt_len: len of time dimension of filter array
    steps = {"t": time step, "f": freq step, "scale":ERB or octaves}

    Output
    ------
    win_arr (n_samples_in_scene, n_channels): 
        overlapping cos-shaped windows for amplitude modulation in frequency

    From Josh McDermott's Sound Texture Synthesis code (MATLAB)
    """

    n_freqs = int((filt_len - 1)*np.floor(steps["t"]*audio_sr))
    max_freq = audio_sr/2. - 1 

    if steps["scale"] == "ERB": 
        freq_to_scale = freq_to_ERB 
        scale_to_freq = ERB_to_freq
    elif steps["scale"] == "octave":
        freq_to_scale = freq_to_octave
        scale_to_freq = octave_to_freq
    
    loscale = freq_to_scale(context.renderer["lo_lim_freq"])
    hiscale = freq_to_scale(max_freq)
    n_channels = int(len(get_event_gp_freqs(audio_sr, steps)) + 2)
    freqs = np.linspace(0, stop=max_freq, num=n_freqs)    

    scale_cutoffs_1D = np.linspace(loscale, stop=hiscale, num=n_channels)
    scale_cutoffs = np.array([[scale_cutoffs_1D[i], scale_cutoffs_1D[i+2]] for i in range(n_channels - 2)]) #50% overlap
    freq_cutoffs = scale_to_freq(scale_cutoffs)

    win_arr = np.zeros((n_freqs, n_channels))
    for curr_channel in range(n_channels - 2):
        
        curr_lo_freq = freq_cutoffs[curr_channel][0]
        curr_hi_freq = freq_cutoffs[curr_channel][1]
        curr_lo_freq_idx = np.argmax(freqs > curr_lo_freq)
        curr_hi_freq_idx = np.argmin(freqs < curr_hi_freq)

        curr_lo_scale = scale_cutoffs[curr_channel][0]
        curr_hi_scale = scale_cutoffs[curr_channel][1]
        curr_mean_scale = (curr_hi_scale + curr_lo_scale)/2
        scale_bandwidth = curr_hi_scale - curr_lo_scale

        curr_scale = freq_to_scale(freqs[curr_lo_freq_idx:curr_hi_freq_idx])
        normalized_domain = 2*(curr_scale - curr_mean_scale)/scale_bandwidth
        curr_win = freq_win_shape(np.pi*normalized_domain)
        win_arr[curr_lo_freq_idx:curr_hi_freq_idx, curr_channel + 1] = curr_win

    rfc = freq_cutoffs[::-1, :]
    bandwidthsHz = rfc[:, 1] - rfc[:, 0]
    # Inversion necessary to line up with the corr_grid, to make
    # sure noise comes out appropriately (eg pink spectrum looks
    # flat on cochleagram)
    bandwidthsHz = np.tile(bandwidthsHz[np.newaxis,:], (filt_len,1))

    return win_arr[:, 1:-1], bandwidthsHz, freq_cutoffs, freqs


def white_excitation(sig_len, sr):
    """ Generate bandpass white noise
    Input
    -----
    duration (s)
    sr (sampling rate, Hz)
    
    Returns
    -------
    1D vector of constant amplitude white noise
    """
    nyq_freq = int(np.floor(sr/2))
    hi_lim_freq = nyq_freq - 1
    lo_idx = int(np.ceil(context.renderer.lo_lim_freq/(1. * nyq_freq)*sig_len))
    hi_idx = int(np.floor(hi_lim_freq/(1. * nyq_freq)*sig_len))

    noise_spec = np.zeros((sig_len,))
    noise_spec[lo_idx:hi_idx] = np.random.randn(hi_idx - lo_idx)
    source = idct(noise_spec)
    source /= ((np.mean(source**2))**0.5 + rendering_log_const)

    return source


def edges(thop, num_frames, twin):
    frame_left = thop*np.arange(num_frames)
    frame_right = frame_left + twin
    return [[frame_left[frame_idx], frame_right[frame_idx]] for frame_idx in range(len(frame_left))]


# Create differentiable ramps
slope_x = [0.0460, 0.0306, 0.0230, 0.0184, 0.0154]
slope_y = [200., 300., 400., 500., 600.]
slope_interpolator = scipy.interpolate.interp1d(slope_x, slope_y, kind="linear")

def differentiable_ramp(t, t_0, is_onset=True, sample_loc="inner"):
    """
    Inputs:
        t: [n_timepoints] OR [batch_size, n_timepoints]
        t_0: [batch_size, 1]
    Returns:
        R: [batch_size, n_timepoints]
    """

    slope = slope_interpolator(context.renderer["ramp_duration"]).item()
    ramp_half_duration = context.renderer["ramp_duration"]/2.0
    if is_onset:
        if sample_loc == "inner":
            T = t_0 - ramp_half_duration
        elif sample_loc == "outer":
            T = t_0 + ramp_half_duration
        elif sample_loc == "middle":
            T = t_0
        R = torch.sigmoid(slope*(t-T))
    else:
        if sample_loc == "inner":
            T = t_0 + ramp_half_duration
        elif sample_loc == "outer":
            T = t_0 - ramp_half_duration
        elif sample_loc == "middle":
            T = t_0
        R = torch.sigmoid(-slope*(t-T))

    return R


def step_ramp(t, t_0, is_onset=True):
    if is_onset:
        R = 1.0*(t >= t_0)
    else:
        R = 1.0*(t <= t_0)
    return R


def spectrum_from_torch_rfft(input, sr, rms_ref):
    """ Get spectrum based on rfft """
    n = input.shape[-1]
    y = torch.fft.rfft(input, n=n, dim=-1, norm=None)
    f = torch.fft.rfftfreq(n, d=1./sr)
    # Go from complex magnitude to dB level
    magnitude = torch.abs(y)/n
    power = 2 * magnitude.square()
    spectrum = 10*torch.log10(power / rms_ref**2)
    return f, spectrum
