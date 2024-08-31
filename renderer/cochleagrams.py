from __future__ import division
import os
import sys
import numpy as np

import torch
import torchaudio

from renderer.util import hann_ramp


if os.path.isdir(os.environ["ch_dir"]):
    sys.path.append(os.environ["ch_dir"])
    import chcochleagram


#######################
# Code for generating cochleagrams in two ways.
#
# First method: full cochleagram generation
#   https://github.com/jenellefeather/chcochleagram
#
# Second method: Gammatone-like spectrograms
#   PyTorch version of:
#   D. P. W. Ellis (2009). "Gammatone-like spectrograms", web resource.
#   http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/
#   On the corresponding webpage, Dan notes that he would be grateful
#   if you cited him if you use his work (as above).
#   This python code does not contain all features present in MATLAB code.
#######################

# chcochleagram generation
def cgm_settings(signal_size, audio_sr, cgm_params):
    """ Settings for chcochleagram generation """

    pad_factor = cgm_params["pad_factor"]
    use_rfft = cgm_params["use_rfft"]

    ## Define cochlear filters
    coch_filter_kwargs = {'use_rfft': use_rfft,
                          'pad_factor': pad_factor,
                          'filter_kwargs': cgm_params["half_cos_filter_kwargs"]
                          }
    filters = chcochleagram.cochlear_filters.ERBCosFilters(signal_size,
                                                           audio_sr,
                                                           **coch_filter_kwargs)

    ##Envelope extraction operations
    envelope_extraction_hilbert = chcochleagram.envelope_extraction.HilbertEnvelopeExtraction(signal_size,
                                                                                      audio_sr,
                                                                                      use_rfft, 
                                                                                      pad_factor)
    rectify_subbands = chcochleagram.envelope_extraction.RectifySubbands(signal_size,
                                                                            audio_sr,
                                                                            use_rfft,
                                                                            pad_factor)

    ## Downsampling operations
    # High env_sr (for rectified subbands, eg to mimic phase locking limit of 4 kHz)
    downsampling_hi = chcochleagram.downsampling.HannPooling1d(
        audio_sr, **cgm_params["hann_downsampling_kwargs"]
        )
    # Low env_sr (to make envelopes for cochleagram)
    downsampling_lo = chcochleagram.downsampling.SincWithKaiserWindow(
        audio_sr, **cgm_params["sk_downsampling_kwargs"]
        )

    ## Compression operations
    if cgm_params["compression_type"] == "log":
        compression = chcochleagram.compression.LogCompression(ref=cgm_params["ref"])
        def threshold(x, scene_max=None):
            scene_max = x.max() if scene_max is None else scene_max
            cutoff = max(scene_max - cgm_params["rel_threshold"], cgm_params["abs_threshold"])
            x[x < cutoff] = cgm_params["abs_threshold"]
            return x
    else:
        raise Excpetion(f"{cgm_params['compression_type']} compression type not implemented.")

    ## Combining for cochleagram generation
    cochleagram = chcochleagram.cochleagram.Cochleagram(filters,
                                                        envelope_extraction_hilbert,
                                                        downsampling_lo,
                                                        compression=compression)
    
    return {
        "cochleagram": cochleagram,
        "envelope_extraction_hilbert": envelope_extraction_hilbert,
        "rectify_subbands": rectify_subbands,
        "downsampling_hi": downsampling_hi,
        "downsampling_lo": downsampling_lo,
        "compression": compression
        }, threshold


def subbands(x, cochleagram, rectify_subbands, compression, threshold, downsampling=None, subbands_only=False):
    """ Create subbands with chcochleagram """
    subbands = cochleagram.compute_subbands(x)
    if subbands_only:
        return subbands
    rectified_subbands = rectify_subbands(subbands)
    if downsampling is not None:
        rectified_subbands = downsampling(rectified_subbands)
    rectified_subbands = compression(rectified_subbands)
    if threshold is not None:
        rectified_subbands = threshold(rectified_subbands)
    return subbands, rectified_subbands


def envelopes(subbands, envelope_extraction_hilbert, compression,  threshold, downsampling=None, extracted_envs_only=False):
    """ Create envelopes with chcochlegram """
    extracted_envs = envelope_extraction_hilbert(subbands) 
    if extracted_envs_only:
        return extracted_envs
    if downsampling is not None:
        envelopes = downsampling(extracted_envs)
        envelopes = compression(envelopes)
    else:
        envelopes = compression(extracted_envs)
    if threshold is not None:
        envelopes = threshold(envelopes)

    return envelopes, extracted_envs


def envelopes_to_cgm(envelopes, downsampling, compression, threshold):
    """ Combine envelopes into cochleagram """
    # To prevent edge artefacts: 1. ramp the observation
    off_ramp = torch.Tensor(hann_ramp(downsampling.sr, ramp_duration=0.005)).to(envelopes.device)
    on_ramp = torch.flip(off_ramp,dims=(0,))
    scene_ramp = torch.cat((on_ramp,torch.ones(envelopes.shape[2]-2*off_ramp.shape[0]).to(envelopes.device), off_ramp))
    envelopes = envelopes * scene_ramp[None, None, :]

    # To prevent edge artefacts: 2. downsample with padding
    # and then clip the edges
    # afterwards we will clip 5 from each side - prevent edge artifacts
    L_in = envelopes.shape[2]
    stride = downsampling.downsample_factor
    n_to_clip = 5
    kernel_size = downsampling.window_size
    L_out = L_in/stride + 2*n_to_clip
    padding = int(np.ceil((L_out * stride + kernel_size - L_in)/2 - stride/2))

    D0 = envelopes.shape[0]
    D1 = envelopes.shape[1]
    p = torch.zeros(D0, D1, padding).to(envelopes.device)
    envelopes = torch.cat((p, envelopes, p), dim=2)

    envelopes = downsampling(envelopes)
    envelopes = envelopes[:, :, n_to_clip:-n_to_clip]
    envelopes = compression(envelopes)
    if threshold is not None:
        envelopes = threshold(envelopes)

    return envelopes


# gammatonegram-like spectrogram
def fft2gammatonemx(nfft, sr=20000, nfilts=64, width=1.0, minfreq=100,  maxfreq=10000, maxlen=1024): 
    """ Convert spectrogram to gammatonegram, based on Dan Ellis' code

        Generate a matrix of weights to combine FFT bins into Gammatone bins.
        nfft: source FFT size at sampling rate sr
        nfilts: number of output bands
        width: constant width of each band in Bark
        minfreq, maxfreq: range covered in Hz
    """

    wts = np.zeros([nfilts, nfft])

    # after Slaney's MakeERBFilters
    EarQ = 9.26449; minBW = 24.7; order = 1

    nFr = np.array(range(nfilts)) + 1
    em = EarQ*minBW
    cfreqs = (maxfreq+em)*np.exp(nFr*(-np.log(maxfreq + em)+np.log(minfreq + em))/nfilts)-em
    cfreqs = cfreqs[::-1]

    GTord = 4
    ucircArray = np.array(range(int(nfft/2 + 1)))
    ucirc = np.exp(1j*2*np.pi*ucircArray/nfft);

    ERB = width*np.power(np.power(cfreqs/EarQ,order) + np.power(minBW,order),1/order)
    B = 1.019 * 2 * np.pi * ERB
    r = np.exp(-B/sr)
    theta = 2*np.pi*cfreqs/sr
    pole = r*np.exp(1j*theta)
    T = 1/sr
    ebt = np.exp(B*T)
    cpt = 2*cfreqs*np.pi*T
    ccpt = 2*T*np.cos(cpt)
    scpt = 2*T*np.sin(cpt)
    A11 = -np.divide(np.divide(ccpt, ebt) + np.divide(np.sqrt(3+2**1.5)*scpt, ebt), 2)
    A12 = -np.divide(np.divide(ccpt, ebt) - np.divide(np.sqrt(3+2**1.5)*scpt, ebt), 2)
    A13 = -np.divide(np.divide(ccpt, ebt) + np.divide(np.sqrt(3-2**1.5)*scpt, ebt), 2)
    A14 = -np.divide(np.divide(ccpt, ebt) - np.divide(np.sqrt(3-2**1.5)*scpt, ebt), 2)
    zros = -np.array([A11, A12, A13, A14])/T
    wIdx = range(int(nfft/2 + 1))
    gain = np.abs((-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) - np.sqrt(3 - 2**(3/2))*  np.sin(2*cfreqs*np.pi*T))) *(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) + np.sqrt(3 - 2**(3/2)) *  np.sin(2*cfreqs*np.pi*T)))*(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) -  np.sqrt(3 + 2**(3/2))*np.sin(2*cfreqs*np.pi*T))) *(-2*np.exp(4*1j*cfreqs*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cfreqs*np.pi*T)*T* (np.cos(2*cfreqs*np.pi*T) + np.sqrt(3 + 2**(3/2))*np.sin(2*cfreqs*np.pi*T))) /(-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cfreqs*np.pi*T) +  2*(1 + np.exp(4*1j*cfreqs*np.pi*T))/np.exp(B*T))**4)
    wts[:, wIdx] = ((T**4)/np.reshape(gain,(nfilts,1))) * np.abs(ucirc-np.reshape(zros[0],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[1],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[2],(nfilts,1)))*np.abs(ucirc-np.reshape(zros[3],(nfilts,1)))*(np.abs(np.power(np.multiply(np.reshape(pole,(nfilts,1))-ucirc,np.conj(np.reshape(pole,(nfilts,1)))-ucirc),-GTord)))    
    wts = wts[:, range(maxlen)]

    return wts, cfreqs


def gtg_settings(sr=20000,twin=0.025,thop=0.010,N=64, fmin=1,fmax=10000,width=1.0,return_freqs=False, return_all=False):
    """ Return settings for gammatonegram, based on Dan Ellis' code.

    Calculate spectrogram-like time-frequency magnitude matrix
    based on Gammatone subband filters

    Waveform x at sampling rate sr
    N is the number of channels
    fmin, fmax: lowest and highest frequencies
    twin: window duration in seconds
    thop: hop size in seconds
    width: scale bandwidth of filters relative to default
    gtg_center_freqs returns the center frequencies in Hz of each row
    """
    nfft = int(2**(np.ceil(np.log(2*twin*sr)/np.log(2))))
    nhop = int(np.round(thop*sr))
    nwin = int(np.round(twin*sr))
    [gtm, gtg_center_freqs] = fft2gammatonemx(nfft, sr, N, width, fmin, fmax, int(nfft/2+1))

    if return_freqs:
        return gtg_center_freqs
    elif return_all:
        return nfft, nhop, nwin, gtm, gtg_center_freqs
    else:
        return nfft, nhop, nwin, gtm


def gammatonegram(x, nfft, nhop, nwin, gtm, rms_ref, window, log_constant=1e-12, dB_threshold=20):
    """ Return gammatonegram, based on Dan Ellis' code.

    Calculate spectrogram-like time-frequency magnitude matrix
    based on Gammatone subband filters
    Torch specrtrogram default center = True, this means "input will be padded on both sides so that the t-th frame is centered at time t x hop_length
    """

    Sxx = torchaudio.functional.spectrogram(
        x, pad=0, window=window, n_fft=nfft, hop_length=nhop, win_length=nwin, power=1.0, normalized=False
        )  # size: (batch, freq, time)
    y = (1/nfft)*torch.matmul(gtm, Sxx)
    y = y / rms_ref
    y = torch.clamp(y, min=log_constant)
    y = 20.*torch.log10(y)  # convert to dB
    y = torch.clamp(y, min=dB_threshold)

    return y
