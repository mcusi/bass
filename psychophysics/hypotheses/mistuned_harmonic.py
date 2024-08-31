import os
import yaml
import numpy as np
import scipy.interpolate

from renderer.util import freq_to_ERB, get_event_gp_freqs
import psychophysics.generation.mistuned_harmonic as gen
import psychophysics.hypotheses.hutil as hutil
from util import context


def full_design(audio_folder, hypothesis_config_name, overwrite={}):
    """Return all stimuli and initial hypotheses for mistuned_harmonic expts"""

    with open(os.path.join(os.environ["config_dir"], hypothesis_config_name + '.yaml'), 'r') as f:
        hypothesis_config = yaml.load(f, Loader=yaml.FullLoader)

    with context(
        audio_sr=hypothesis_config["renderer"]["steps"]["audio_sr"],
        rms_ref=hypothesis_config["renderer"]["tf"]["rms_ref"]
    ):
        # 1. Create sounds for inference if not already made
        settings_fn = os.path.join(
            os.environ["sound_dir"], audio_folder, "mh_expt_settings.npy"
        )
        if not os.path.isfile(settings_fn):
            os.makedirs(os.path.join(
                os.environ["sound_dir"], audio_folder, ""
            ))
            _, settings = gen.expt(audio_folder, overwrite=overwrite)
        else:
            settings = np.load(settings_fn, allow_pickle=True).item()
        if settings["audio_sr"] != context.audio_sr or settings["rms_ref"] != context.rms_ref:
            raise Exception("Conflicting config & settings for hypothesis defn.")

    # 2. Compile the stimuli, along with initial-hypotheses for each 
    experiments = {}
    lev = settings["mistuned_level"]
    for mistuned_idx in settings["mistuned_idxs"]:
        for harmonic_f0 in settings["fundamentals"]:
            for duration_ms in settings["durations"]:
                for p_val in settings["mistuned_percents"]:
                    obs_name = f"f{harmonic_f0:04d}_dur{duration_ms}_harm{mistuned_idx}_p{p_val}_dB{lev}"
                    # Create 3 initializations
                    # HplusW and HplusQW are both part of the "hw" hypothesis,
                    # but different initializations.
                    Honly = harmonic_full_hypothesis(
                        duration_ms,
                        harmonic_f0,
                        settings,
                        hypothesis_config
                    )
                    HplusW = harmonic_empty_hypothesis(
                        duration_ms,
                        harmonic_f0,
                        p_val,
                        mistuned_idx,
                        settings,
                        hypothesis_config
                    )
                    HplusWq = harmonic_partial_hypothesis(
                        duration_ms,
                        harmonic_f0,
                        p_val,
                        mistuned_idx,
                        settings,
                        hypothesis_config
                    )
                    # Add to experiments
                    experiments.update(hutil.format(Honly, obs_name, "h"))
                    experiments.update(hutil.format(HplusW, obs_name, "hw"))
                    experiments.update(hutil.format(HplusWq, obs_name, "hwq"))
                    
    return experiments


# Helpers to make hypotheses
def create_tone_hypothesis(onset, offset, f0, amplitude, source_type, config):
    onset = onset + config["renderer"]["ramp_duration"]
    offset = offset - config["renderer"]["ramp_duration"]
    ts = np.arange(
        onset + 0.001,
        offset + config["hypothesis"]["delta_gp"]["t"],
        config["hypothesis"]["delta_gp"]["t"]
    )
    source = {
        "source_type": source_type,
        "events": [{"onset": onset, "offset": offset}],
        "features": {
            "f0": {
                "x": ts,
                "y": np.full(ts.shape, freq_to_ERB(f0))
            },
            "amplitude": {
                "x": ts,
                "y": np.full(ts.shape, amplitude)
                }
            }
        }
    return source


def create_whistle_hypothesis(onset, offset, f0, amplitude, config):
    return create_tone_hypothesis(onset, offset, f0, amplitude, "whistle", config)


def de_attenuate(desired_amplitude, harmonic_idx, config):
    """ Reduce level according to harmonic idx """
    # harmonic_idx = 1 = fundamental
    alpha = config["renderer"]["source"]["harmonic"]["attenuation_constant"]
    return 10*np.log10(10**(desired_amplitude/10.) * harmonic_idx * np.exp(alpha*(harmonic_idx-1)))


def create_harmonic_hypothesis(onset, offset, f0, amplitude, mistuned_idx, n_harmonics, config, db_to_subtract_at_mistuned=30):
    """ Add spectrum into tone hypothesis to get full harmonic hypothesis """
    audio_sr = config['renderer']['steps']['audio_sr']
    n_channels = len(get_event_gp_freqs(audio_sr, config['renderer']['steps']))
    channel_width = (freq_to_ERB(audio_sr/2. - 1.) - freq_to_ERB(config["renderer"]["lo_lim_freq"]))/(n_channels+1)
    db_low = 30
    if mistuned_idx is False:
        # Set spectrum by harmonic number
        xy_small = np.array([
            [freq_to_ERB(f0) - channel_width,    amplitude],
            [freq_to_ERB(f0),                    amplitude],
            *[[freq_to_ERB(f0 * i),              de_attenuate(amplitude, i, config)] for i in range(2,n_harmonics+1)],
            [freq_to_ERB(f0 * (n_harmonics+1)),  amplitude-db_low],
            [freq_to_ERB(f0) + (n_channels-2) * channel_width, amplitude-db_low]
        ])
    elif mistuned_idx < 1 or mistuned_idx > n_harmonics:
        raise Exception("Needs to be 1 <= mistuned_idx < n_harmonics.")
    else:
        # Set spectrum by harmonic number
        xy_small = np.array([
            [freq_to_ERB(f0) - channel_width,    amplitude],
            *[[freq_to_ERB(f0 * i),              amplitude-db_to_subtract_at_mistuned if i == mistuned_idx else de_attenuate(amplitude, i, config)] for i in range(1,n_harmonics+1)],
            [freq_to_ERB(f0 * (n_harmonics+1)),  amplitude-db_low],
            [freq_to_ERB(f0) + (n_channels-2) * channel_width, amplitude-db_low]
        ])
    # Interpolate to get hypothesis spacing
    x_small = xy_small[:, 0] - xy_small[0, 0] + freq_to_ERB(config["renderer"]["lo_lim_freq"])
    y_small = xy_small[:, 1]
    spectrum = {}
    spectrum["x"] = np.arange(
        x_small[0],
        x_small[-1]+config["hypothesis"]["delta_gp"]["f"],
        config["hypothesis"]["delta_gp"]["f"]
    )
    interpol8r = scipy.interpolate.interp1d(
        x_small, y_small, 
        kind="nearest", fill_value="extrapolate"
    )
    spectrum["y"] = interpol8r(spectrum["x"])
    source = create_tone_hypothesis(
        onset, offset, f0, np.mean(spectrum["y"]), "harmonic", config
    )
    spectrum["y"] = spectrum["y"] - np.mean(spectrum["y"])
    source["features"]["spectrum"] = spectrum
    return source


def harmonic_full_hypothesis(duration_ms, harmonic_f0, stimulus_settings, hypothesis_config):
    """ Full harmonic with equal amplitude components """
    component_level = stimulus_settings["harmonic_level"]
    onset = stimulus_settings["pad"]
    offset = stimulus_settings["pad"] + duration_ms/1000. + 2*stimulus_settings["ramp_duration"] 
    fundamental_idx = [i for i in range(len(stimulus_settings["fundamentals"])) if stimulus_settings["fundamentals"][i] == harmonic_f0][0]
    n_harmonics = stimulus_settings["n_harmonics"][fundamental_idx]
    return [create_harmonic_hypothesis(onset, offset, harmonic_f0, component_level, False, n_harmonics, hypothesis_config)]


def harmonic_empty_hypothesis(duration_ms, harmonic_f0, percent_mistuning, mistuned_idx, stimulus_settings, hypothesis_config, whistle_amp=None, db_to_subtract_at_mistuned=30):
    """ Harmonic with completely missing component at the mistuned_idx """
    component_level = stimulus_settings["harmonic_level"]
    onset = stimulus_settings["pad"]
    offset = stimulus_settings["pad"] + duration_ms/1000. + 2*stimulus_settings["ramp_duration"]
    fundamental_idx = [i for i in range(len(stimulus_settings["fundamentals"])) if stimulus_settings["fundamentals"][i] == harmonic_f0][0]
    n_harmonics = stimulus_settings["n_harmonics"][fundamental_idx]
    # Create and return harmonic source
    harmonic_source = create_harmonic_hypothesis(
        onset, offset, harmonic_f0, component_level,
        mistuned_idx, n_harmonics, hypothesis_config,
        db_to_subtract_at_mistuned=db_to_subtract_at_mistuned
        )
    # Whistle at full amplitude of the mistuned stimulus
    whistle_amp = stimulus_settings["mistuned_level"] if whistle_amp is None else whistle_amp
    whistle_f0 = harmonic_f0 * (mistuned_idx + percent_mistuning/100.)
    whistle_source = create_whistle_hypothesis(
        onset, offset, whistle_f0, whistle_amp, hypothesis_config
    )
    return [harmonic_source, whistle_source]


def harmonic_partial_hypothesis(duration_ms, harmonic_f0, percent_mistuning, mistuned_idx, stimulus_settings, hypothesis_config):
    """Shift the amplitudes in the frequency of the mistuned_idx"""
    return harmonic_empty_hypothesis(
        duration_ms,
        harmonic_f0,
        percent_mistuning,
        mistuned_idx,
        stimulus_settings,
        hypothesis_config,
        whistle_amp=stimulus_settings["mistuned_level"]-10,
        db_to_subtract_at_mistuned=6
    )
