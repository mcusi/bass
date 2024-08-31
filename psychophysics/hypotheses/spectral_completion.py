import os
import yaml
import numpy as np 

import psychophysics.hypotheses.hutil as hutil
import psychophysics.generation.spectral_completion as gen
from util import context


def full_design(sound_group, inference_config_name, overwrite={}):
    """ Initializes each spectral completion experiment w/ their tabs/maskers,
        but with several different levels for the the masked middle portion.
    """

    with open(os.path.join(os.environ["config_dir"], inference_config_name + '.yaml'), 'r') as f:
        inference_config = yaml.load(f, Loader=yaml.FullLoader)  

    seed = overwrite.get("seed", 0)
    with context(
        audio_sr=inference_config["renderer"]["steps"]["audio_sr"],
        rms_ref=inference_config["renderer"]["tf"]["rms_ref"]
    ):
        # 1. Create sounds for inference if not already made
        soundpath = os.path.join(os.environ["sound_dir"], sound_group, "")
        if not os.path.isdir(soundpath):
            os.makedirs(soundpath, exist_ok=True)
        if not os.path.isfile(soundpath + f"sc_fig1_settings_seed{seed}.npy"):
            fig1_settings = gen.fig1(soundpath, overwrite)
        else:
            fig1_settings = np.load(soundpath + f"sc_fig1_settings_seed{seed}.npy", allow_pickle=True).item()
        if not os.path.isfile(soundpath + f"sc_fig2_settings_seed{seed}.npy"):
            fig2_settings = gen.fig2(soundpath, overwrite)
        else:
            fig2_settings = np.load(soundpath + f"sc_fig2_settings_seed{seed}.npy", allow_pickle=True).item()
        for settings in [fig1_settings, fig2_settings]:
            if settings["audio_sr"] != context.audio_sr or settings["rms_ref"] != context.rms_ref:
                raise Exception("Conflicting config and settings for hypothesis definition.")

    # 2. Make cache of maskers and tabs for initialization inference
    # Choose mids which span the full y-axis
    fig1_mids = np.array([-20., 0., 10., 20., 30.])
    fig2_mids = np.array([-5., -2.5, 0., 5., 10., 12.5, 20.])
    cache = make_cache(
        inference_config, fig1_settings, fig2_settings, fig1_mids, fig2_mids
        )

    # 3. Use cache to make initializations
    experiments = {}
    # create fig1 initializations
    for sound_name in ["1i", "1ii", "1iii", "1iv", "1v"]: 
        for middle_level in fig1_mids:
            initialization = combine_cache(sound_name, middle_level, cache)
            experiments.update(hutil.format(initialization, f"sc_{sound_name}_seed{seed}", f"mid{int(middle_level):03d}"))

    # create fig2 initializations
    for sound_name in ["2i", "2ii", "2iii", "2iv", "2v", "2vi"]:
        for middle_level in fig2_mids:
            initialization = combine_cache(sound_name, middle_level, cache)
            experiments.update(hutil.format(initialization, f"sc_{sound_name}_seed{seed}", f"mid{int(middle_level):03d}"))

    return experiments

# 2) Cache of modal initializations

def fig1_spectrum_level_func(middle_level, fig1_settings, silence_db_per_hz):
    def spectrum_level_function(f):
        if f < fig1_settings["lowBand"][0]:
            return silence_db_per_hz
        elif fig1_settings["highBand"][1] <= f:
            return silence_db_per_hz - 5
        elif fig1_settings["lowBand"][0] <= f < fig1_settings["lowBand"][1]:
            return fig1_settings["SL"]
        elif fig1_settings["highBand"][0] <= f < fig1_settings["highBand"][1]:
            return fig1_settings["SL"]
        elif fig1_settings["midBand"][0] <= f < fig1_settings["midBand"][1]:
            return middle_level
    return spectrum_level_function


def fig2_spectrum_level_func(middle_level, stimulus_name, fig2_settings, silence_db_per_hz):
    def spectrum_level_function(f):
        if f < fig2_settings["lowBand"][0]:
            return silence_db_per_hz
        elif fig2_settings["highBand"][1] <= f:
            return silence_db_per_hz - 5
        elif fig2_settings["lowBand"][0] <= f < fig2_settings["lowBand"][1]:
            return fig2_settings["SLs"][stimulus_name][0]
        elif fig2_settings["highBand"][0] <= f < fig2_settings["highBand"][1]:
            return fig2_settings["SLs"][stimulus_name][2]
        elif fig2_settings["midBand"][0] <= f < fig2_settings["midBand"][1]:
            return middle_level
    return spectrum_level_function


def make_cache(inference_config, fig1_settings, fig2_settings, fig1_mids, fig2_mids):

    # Load current cache if any
    cache = {
        "fig1": {"tabs": {}, "maskers": {}},
        "fig2": {"tabs": {}, "maskers": {}}
        }

    # Load config 
    audio_sr = fig1_settings["audio_sr"]
    silence_db_per_hz = 0.

    # Figure 1
    # Tabs
    for middle_level in fig1_mids:
        onset = fig1_settings["tabPad"]
        offset = onset + fig1_settings["tabDur"]
        spectrum_level_function = fig1_spectrum_level_func(middle_level, fig1_settings, silence_db_per_hz)
        erbs, spectrum_init, amplitude_init = hutil.create_noise_latents(spectrum_level_function, inference_config, audio_sr)
        H = create_source(onset, offset, {"x": erbs, "y": spectrum_init}, amplitude_init, inference_config)
        cache["fig1"]["tabs"][middle_level] = H

    # Masker
    for stimulus_name in ["iii", "iv", "v"]: 
        onset = fig1_settings["maskerPad"]
        offset = onset + fig1_settings["maskerDur"]
        if stimulus_name == "iii" or stimulus_name == "v":
            def spectrum_level_function(f):
                if f < fig1_settings["lowBand"][1]:
                    return silence_db_per_hz
                elif fig1_settings["highBand"][0] <= f:
                    return silence_db_per_hz - 5
                elif fig1_settings["midBand"][0] <= f < fig1_settings["midBand"][1]:
                    return fig1_settings["SL"]
        else:
            def spectrum_level_function(f):
                if f < fig1_settings["lowBand"][1]:
                    return silence_db_per_hz
                elif fig1_settings["highBand"][0] <= f:
                    return silence_db_per_hz - 5
                elif fig1_settings["iv"]["bottomMidBand"][1] <= f < fig1_settings["iv"]["topMidBand"][0]:
                    return silence_db_per_hz
                elif fig1_settings["iv"]["bottomMidBand"][0] <= f < fig1_settings["iv"]["bottomMidBand"][1]:
                    return fig1_settings["SL"] + fig1_settings["iv"]["SLshift"]
                elif fig1_settings["iv"]["topMidBand"][0] <= f < fig1_settings["iv"]["topMidBand"][1]:
                    return fig1_settings["SL"] + fig1_settings["iv"]["SLshift"]

    erbs, spectrum_init, amplitude_init = hutil.create_noise_latents(
        spectrum_level_function, inference_config, audio_sr
        )
    H = create_source(
        onset, offset, {"x": erbs, "y": spectrum_init}, amplitude_init, inference_config
        )
    if stimulus_name == "v":
        second_event_onset = fig1_settings["fullDur"] - fig1_settings["maskerPad"] - fig1_settings["v"]["duration"]
        second_event_offset = fig1_settings["fullDur"] - fig1_settings["maskerPad"]
        H["events"][0]["offset"] = second_event_onset - fig1_settings["tabDur"]
        H["events"].append({
            "onset": second_event_onset,
            "offset": second_event_offset
            })
    cache["fig1"]["maskers"][stimulus_name] = H

    # Fig 2
    # Tabs
    for stimulus_name in ["i", "ii", "iii", "iv", "v", "vi"]:
        cache["fig2"]["tabs"][stimulus_name] = {}
        for middle_level in fig2_mids:
            onset = fig2_settings["tabPad"]
            offset = onset + fig2_settings["tabDur"]
            spectrum_level_function = fig2_spectrum_level_func(
                middle_level, stimulus_name, fig2_settings, silence_db_per_hz
                )
            erbs, spectrum_init, amplitude_init = hutil.create_noise_latents(
                spectrum_level_function, inference_config, audio_sr
                )
            H = create_source(
                onset, offset, {"x": erbs, "y": spectrum_init}, amplitude_init, inference_config
                )
            cache["fig2"]["tabs"][stimulus_name][middle_level] = H

    # Masker
    for stimulus_name in ["i", "ii", "iii", "iv", "v", "vi"]:
        onset = fig2_settings["maskerPad"]
        offset = onset + fig2_settings["maskerDur"]

        def spectrum_level_function(f):
            if f < fig1_settings["lowBand"][1]:
                return silence_db_per_hz
            elif fig1_settings["highBand"][0] <= f:
                return silence_db_per_hz - 5
            elif fig1_settings["midBand"][0] <= f < fig1_settings["midBand"][1]:
                return fig2_settings["SLs"][stimulus_name][1]

        erbs, spectrum_init, amplitude_init = hutil.create_noise_latents(
            spectrum_level_function, inference_config, audio_sr
            )
        H = create_source(
            onset, offset, {"x": erbs, "y": spectrum_init}, amplitude_init, inference_config
            )
        cache["fig2"]["maskers"][stimulus_name] = H

    return cache


def create_source(onset, offset, spectrum_dict, amplitude, config):
    onset = onset + config["renderer"]["ramp_duration"]
    offset = offset - config["renderer"]["ramp_duration"]
    ts = np.arange(
        onset + 0.001,
        offset + config["hypothesis"]["delta_gp"]["t"],
        config["hypothesis"]["delta_gp"]["t"]
        )
    feature_dict = {
        "amplitude": {
            "x": ts, "y": np.full(ts.shape, amplitude)
            }
        }
    feature_dict["spectrum"] = spectrum_dict
    return {
        "source_type": "noise",
        "events": [{"onset": onset, "offset": offset}], 
        "features": feature_dict
        }


# 3. Combine cache into initializations

def combine_cache(sound_name, middle_level, cache):
    if sound_name == "1i" or sound_name == "1ii":
        sources = [cache["fig1"]["tabs"][middle_level]]
    elif sound_name == "1iii" or sound_name == "1iv" or sound_name == "1v":
        sources = [
            cache["fig1"]["tabs"][middle_level],
            cache["fig1"]["maskers"][sound_name[1:]]
            ]
    elif sound_name[0] == "2":
        sources = [
            cache["fig2"]["tabs"][sound_name[1:]][middle_level],
            cache["fig2"]["maskers"][sound_name[1:]]
            ]
    return sources
