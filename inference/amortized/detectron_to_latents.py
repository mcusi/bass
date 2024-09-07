import numpy as np
import scipy.interpolate
import torch

import renderer.util

################
# Process output of detectron2 into event proposals that can be used by the generative model
# Reverses process in get_targets_from_event in dataloaders.py
################


def get_params_for_prediction(dataset_config, detectron_config):
    """ Get relevant parameters from dataset and detectron configs """
    class_names = dataset_config['hyperpriors']['source_type']['args']
    if detectron_config.BASA.TF_REPRESENTATION == "cgm":
        seconds_per_frame = 1./dataset_config["renderer"]["tf"]["cgm_params"]["sk_downsampling_kwargs"]["env_sr"]
    elif detectron_config.BASA.TF_REPRESENTATION == "gtg":
        seconds_per_frame = dataset_config["renderer"]["tf"]["gtg_params"]["thop"]
    audio_sr = dataset_config["dream"]["audio_sr"]
    fstep = dataset_config["renderer"]["steps"]["f"]
    min_frames = int(np.round(dataset_config["renderer"]["steps"]["t"]/seconds_per_frame))
    return class_names, seconds_per_frame, audio_sr, fstep, min_frames


def make_event_proposal(detectron_output_idx, predictions, f_coch, dataset_config, detectron_config):
    """ Make event proposal from detectron predictions, for one index

        Inputs
        ------
        detectron_output_idx: int
            Index of detectron2 prediction to make an event proposal for
        predictions: dict[str, Tensor]
            Predictions from detectron2 
        f_coch: array
        dataset_config: dict
        detectron_config: dict

        Returns
        -------
        proposal: dict[str,float or str,dict[str, array]]
            Dict of latent variables created from detectron prediction
    """

    pred_classes = predictions["classes"]
    pred_boxes = predictions["boxes"]
    pred_f0 = predictions["f0"]
    pred_amplitude = predictions["amplitude"]
    pred_spa = predictions["spa"]
    scores = predictions["scores"]

    class_names, seconds_per_frame, audio_sr, fstep, min_frames = get_params_for_prediction(dataset_config, detectron_config)
    t_coch = torch.arange(predictions["ibm"].shape[2], dtype=torch.float32)*seconds_per_frame

    # Define initial event proposal
    proposal = {
        "rank": scores[detectron_output_idx],
        "source_type": class_names[pred_classes[detectron_output_idx]]
    }

    # Get onset and offset in seconds, from predicted boxes
    onset_idx = min(
        int(np.round(pred_boxes[detectron_output_idx, 0])),
        len(t_coch) - 1
        )
    offset_idx = int(np.round(pred_boxes[detectron_output_idx, 2]))
    offset_idx = max(
        offset_idx,
        min(onset_idx + min_frames, len(t_coch) - 1)
        )
    if offset_idx == len(t_coch) - 1:
        onset_idx = min(onset_idx, offset_idx - min_frames)
    proposal["onset"] = (t_coch[-1]*pred_boxes[detectron_output_idx, 0]/len(t_coch)).item()
    proposal["offset"] = (t_coch[-1]*pred_boxes[detectron_output_idx, 2]/len(t_coch)).item()

    # Get Gaussian processes: f0, and amplitude, spectrum
    proposal["gps"] = {}
    if (pred_f0 is not None) and ("noise" not in proposal["source_type"]):
        proposal["gps"]["f0"] = process_f0(
            pred_f0[detectron_output_idx, :, :],
            onset_idx,
            offset_idx,
            t_coch,
            f_coch
            )
    if (pred_amplitude is not None) and (proposal["source_type"] == "whistle"):
        proposal["gps"]["amplitude"] = process_amplitude(
            pred_amplitude[detectron_output_idx, :, :],
            onset_idx,
            offset_idx,
            t_coch
            )
    if (pred_spa is not None) and (proposal["source_type"] != "whistle"):
        foffidx = max(2, int(np.floor(pred_boxes[detectron_output_idx, 3])))
        fonidx = min(
            foffidx-2,
            int(np.floor(pred_boxes[detectron_output_idx, 1]))
            )
        f0 = None if "noise" in proposal["source_type"] else proposal["gps"]["f0"]["y"]
        proposal["gps"]["spectrum"], proposal["gps"]["amplitude"] = process_spa(
            pred_spa[detectron_output_idx, :, :],
            f0,
            onset_idx,
            offset_idx,
            fonidx,
            foffidx,
            t_coch,
            f_coch,
            audio_sr,
            fstep,
            dataset_config
            )

    return proposal


def process_f0(pred_f0, onset_idx, offset_idx, t_coch, f_coch):
    """ Process predicted f0 from detectron2 into formatted latent variables for generative model

        Input
        -----
        pred_f0: Tensor
            Detectron2 network output for fundamental frequency, same size as cochleagram input
            Shape: (freq., time)
        onset_idx: int
            Index of onset in cochleagram
        offset_idx: int
            Index of offset in cochleagram
        t_coch: array
            Values for the time dimension of the cochleagram.
            Length = pred_f0.shape[1]
        f_coch: array
            Values for the frequency dimension of the cochleagram.
            Length = pred_f0.shape[0]

        Returns
        -------
        f0_proposal: dict[str, array]
            Indicates the time (x) values and f0 (y) values for this event
    """
    f = f_coch[pred_f0.argmax(0).cpu().numpy()]
    usable_value = (pred_f0.max(0)[0] > 0).cpu().numpy()
    f0_proposal = {
        "x": t_coch[onset_idx:offset_idx+1][usable_value[onset_idx:offset_idx+1]].detach().cpu().numpy(),
        "y": f[onset_idx:offset_idx+1][usable_value[onset_idx:offset_idx+1]].detach().cpu().numpy(),
        }
    return f0_proposal


def process_amplitude(pred_amplitude, onset_idx, offset_idx, t_coch):
    """ Process predicted amplitude from detectron2 into formatted latent variables for generative model"""
    # Shape: (f, t). See scaling in dataloader, at end of get_latents_from_event
    a = 100*pred_amplitude.max(0)[0]
    usable_value = (pred_amplitude.max(0)[0] > 0).cpu().numpy()
    amplitude_proposal = {
        "x": t_coch[onset_idx:offset_idx+1][usable_value[onset_idx:offset_idx+1]].detach().cpu().numpy(),
        "y": a[onset_idx:offset_idx+1][usable_value[onset_idx:offset_idx+1]].detach().cpu().numpy()
        }
    return amplitude_proposal


def process_spa(pred_spa, f0, onset_idx, offset_idx, fon_idx, foff_idx, t_coch, f_coch, audio_sr, fstep, dataset_config):
    """ Process predicted spectrum plus amplitude from detectron2
        into formatted latent variables for generative model
        (separate amplitude and spectrum variables)

        Input
        -----
        pred_spa: Tensor
            Detectron2 network output for spectrum plus amplitude,
            same size as cochleagram input
            Shape: (freq., time)
        f0: array or None
            Processed f0 if it exists
            Shape: (time,)
        onset_idx: int
            Index of onset in cochleagram
        offset_idx: int
            Index of offset in cochleagram
        fon_idx: int
            Index of bottom frequency in cochleagram
        foff_idx: int
            Index of top frequency in cochleagram
        t_coch: array
            Values for the time dimension of the cochleagram.
            Length = pred_spa.shape[1]
        f_coch: array
            Values for the frequency dimension of the cochleagram.
            Length = pred_spa.shape[0]
        audio_sr: int
        fstep: float
        dataset_config: dict

        Returns
        -------
        amplitude_proposal: dict[str, array]
            Indicates the time (x) values and amplitude (y)
            values for this event
        spectrum_proposal: dict[str, array]
            Indicates the ERB (x) values and spectrum amplitude (y)
            values for this event
    """

    # Get amplitude trajectory over time, by taking a mean over frequency
    # See scaling in dataloader, at end of get_latents_from_event
    p = 100*(pred_spa[fon_idx:foff_idx+1, onset_idx:offset_idx+1].T) #results in (t, f)
    usable_timepoints = pred_spa[fon_idx:foff_idx+1, onset_idx:offset_idx+1].max(0)[0] > 0
    p = p[usable_timepoints, :]
    amplitude_0 = p.mean(1).cpu().numpy()
    amplitude_proposal = {
        "x": t_coch[onset_idx:offset_idx+1][usable_timepoints],
        "y": amplitude_0

    }

    # Get spectrum in frequency, by factoring out the mean amplitude value
    spectrum_0 = p.cpu().numpy() - amplitude_0[:, None]
    # Network has predicted spectrum levels at frequencies = f_coch
    # We want to put it back into channels that are linear in ERB space
    new_erb = get_event_gp_freqs_simplified(audio_sr, fstep)
    finterp = f_coch[fon_idx:foff_idx+1].cpu().numpy()
    interpol8r = scipy.interpolate.interp1d(
        finterp, spectrum_0, kind="nearest", fill_value="extrapolate"
        )
    linearly_spaced_spectrum = interpol8r(new_erb)  # shape n_windows, n_channels

    # Last, make adjustments for sound type
    if f0 is not None:
        unshifted_spectrum = unshift_spectrum_by_f0(
            linearly_spaced_spectrum, f0, dataset_config
            )
    else:
        unshifted_spectrum = linearly_spaced_spectrum
        # Need to shift from dB to dB/Hz
        bandwidthsHz = renderer.util.get_bandwidthsHz(audio_sr, dataset_config)
        unshifted_spectrum = unshifted_spectrum - 10*np.log10(bandwidthsHz)[None, :]

    # Define proposal
    spectrum_proposal = {
        "x": new_erb,
        "y": np.mean(unshifted_spectrum, axis=0)
    }

    return spectrum_proposal, amplitude_proposal


def unshift_spectrum_by_f0(time_varying_spectrum, scaled_f0, dataset_config):
    """
    Spectrum shifts with f0 in generative model. Use
    predicted fundamental frequency to determine the
    unshifted spectrum from the spectrum prediction.

    Inputs
    ------
    time_varying_spectrum: array
        Spectrum at each timepoint
        Shape: (freq. channels, time windows)
    f0: array
        Predicted f0 at each timepoint
        Shape: (time windows,)

    Returns
    -------
    unshifted_spectrum: array
        Shape: (freq. channels, time windows)
    """

    n_windows, n_channels = time_varying_spectrum.shape

    # By how many channels do we want to shift the spectrum
    lo_lim_freq = dataset_config["renderer"]["lo_lim_freq"]
    audio_sr = dataset_config['renderer']['steps']['audio_sr']
    n_channels = len(renderer.util.get_event_gp_freqs(
        audio_sr, dataset_config['renderer']['steps']
        ))
    channel_width = (renderer.util.freq_to_ERB(audio_sr/2. - 1.) - renderer.util.freq_to_ERB(lo_lim_freq))/(n_channels+1)
    shift_by = (scaled_f0 - renderer.util.freq_to_ERB(lo_lim_freq)) / channel_width - 1
    shift_by = -shift_by  # Add negative to shift other way

    # Shift
    xnew = np.arange(n_channels)[None, :] - shift_by[:, None]
    unshifted_spectrum = np.zeros(time_varying_spectrum.shape)
    for win in range(n_windows):
        interpol8r = scipy.interpolate.interp1d(
            np.arange(n_channels),
            time_varying_spectrum[win, :],
            kind="linear", fill_value="extrapolate"
            )
        unshifted_spectrum[win, :] = interpol8r(xnew[win, :])  # shape n_windows, n_channels

    return unshifted_spectrum


def get_event_gp_freqs_simplified(audio_sr, fstep):
    """ Return array of frequencies given a step size in ERB """
    lo_lim_freq = 20.
    lo = renderer.util.freq_to_ERB(lo_lim_freq)
    hi = renderer.util.freq_to_ERB(np.floor(audio_sr/2.) - 1.)
    return np.round(np.arange(start=lo, stop=hi, step=fstep), decimals=3)


def silent_box(inputs, outputs, i):
    """ Returns a boolean indicating whether a network output is silent """
    return (inputs["grams"] * (outputs["ibm"][i, :, :] > 0).T).max().item() <= 20.
