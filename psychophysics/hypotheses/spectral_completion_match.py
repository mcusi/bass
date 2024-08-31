import os
import yaml
import numpy as np
import soundfile as sf

import psychophysics.hypotheses.hutil as hutil
from psychophysics.hypotheses.spectral_completion import fig1_spectrum_level_func, fig2_spectrum_level_func, create_source
import psychophysics.generation.spectral_completion as gen
from util import context, manual_seed


def full_design(sound_group, inference_config_name, overwrite={}):
    """ Creates initializes for the stimuli used to match target in spectral completion"""

    # 1. Load settings from spectral completion comparison stimuli
    seed = overwrite.get("seed", 0)
    if sound_group[-6:] != "_match":
        raise Exception("Sound group should include _match")
    comparison_soundpath = os.path.join(os.environ["sound_dir"], sound_group[:-6], "")
    fig1_settings = np.load(
        os.path.join(comparison_soundpath, f"sc_fig1_settings_seed{seed}.npy"),
        allow_pickle=True
        ).item()
    fig2_settings = np.load(
        os.path.join(comparison_soundpath, f"sc_fig2_settings_seed{seed}.npy"),
        allow_pickle=True
        ).item()
    with open(os.path.join(os.environ["config_dir"], inference_config_name + '.yaml'), 'r') as f:
        inference_config = yaml.load(f, Loader=yaml.FullLoader)
    for settings in [fig1_settings, fig2_settings]:
        if settings["audio_sr"] != inference_config["renderer"]["steps"]["audio_sr"] or settings["rms_ref"] != inference_config["renderer"]["tf"]["rms_ref"]:
            raise Exception("Conflicting config and settings for hypothesis definition.")

    # 2. Create sounds
    manual_seed(seed)
    soundpath = os.path.join(os.environ["sound_dir"], sound_group, "")
    os.makedirs(soundpath, exist_ok=True)
    with context(audio_sr=settings["audio_sr"], rms_ref=settings["rms_ref"]):
        fig1_mids = np.arange(-15, 40, 2.5)
        fig2_mids = np.arange(-5, 13, 1.25)
        silence_db_per_hz = 0.
        fig1_tabs_sounds, fig1_settings, audio_sr = fig1_tabs(fig1_mids, fig1_settings)
        fig2_tabs_sounds, fig2_settings, audio_sr = fig2_tabs(fig2_mids, fig2_settings)

    # 3. Create explanations for match stimuli
    experiments = {}
    for middle_level, sound in fig1_tabs_sounds.items():
        sound_name = f"sc_1_{middle_level}_seed{seed}"
        sf.write(os.path.join(soundpath, sound_name + ".wav"), sound, audio_sr)
        onset = fig1_settings["tabPad"]
        offset = onset + fig1_settings["tabDur"]
        spectrum_level_function = fig1_spectrum_level_func(
            middle_level, fig1_settings, silence_db_per_hz
            )
        erbs, spectrum_init, amplitude_init = hutil.create_noise_latents(
            spectrum_level_function, inference_config, audio_sr
            )
        H = create_source(
            onset, offset, {"x": erbs, "y": spectrum_init}, amplitude_init, inference_config
            )
        experiments.update(hutil.format(
            [H], sound_name, f"mid{int(middle_level):03d}"
            ))

    for stimulus_name, tabs_dict in fig2_tabs_sounds.items():
        for middle_level, sound in tabs_dict.items():
            sound_name = f"sc_2{stimulus_name}_{middle_level}_seed{seed}"
            sf.write(
                os.path.join(soundpath, sound_name + ".wav"), sound, audio_sr
                )
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
            experiments.update(hutil.format([H], sound_name, f"mid{int(middle_level):03d}"))

    return experiments

# Generation code for comparison stimuli (for subsequent analysis)


def fig1_tabs(middle_SLs, fig1_settings):
    """ Creates comparison stimuli for figure 1
            Comparison stimuli, use middle_SLs = np.arange(-20,40,2.5)
    """
    s = fig1_settings
    durations = [s["tabDur"], s["tabDur"], s["tabDur"]]
    pads = [s["tabPad"], s["tabPad"], s["tabPad"]]
    sounds = {}
    for middle_SL in middle_SLs:
        SLs = [s["SL"], middle_SL, s["SL"]]
        sounds[middle_SL], audio_sr = gen.combine(s["stdFreqs"], SLs, durations, pads, "")
    return sounds, s, audio_sr


def fig1_maskers(fig1_settings):
    """Create maskers only for each stimulus"""
    s = fig1_settings
    sounds = {}
    # iii
    SLs = [s["SL"]]
    durations = [s["maskerDur"]]
    pads = [s["maskerPad"]]
    sounds["iii"], audio_sr = gen.combine([s["midBand"]], SLs, durations, pads)
    # iv
    freqs = [s["iv"]["bottomMidBand"], s["iv"]["topMidBand"]]
    SLs = [s["SL"] + s["iv"]["SLshift"],s["SL"] + s["iv"]["SLshift"]]
    durations = [s["maskerDur"], s["maskerDur"]]
    pads = [s["maskerPad"], s["maskerPad"]]
    sounds["iv"], audio_sr = gen.combine(freqs, SLs, durations, pads)
    # v
    freqs = [s["midBand"], s["midBand"]]
    SLs = [s["SL"], s["SL"]]
    durations = [s["v"]["duration"], s["v"]["duration"]]
    pads = [s["maskerPad"], s["fullDur"] - s["maskerPad"] - s["v"]["duration"]], [s["fullDur"] - s["maskerPad"] - s["v"]["duration"], s["maskerPad"]]
    sounds["v"], audio_sr = gen.combine(freqs, SLs, durations, pads)
    return sounds, s, audio_sr


def fig2_tabs(middle_SLs, fig2_settings):
    """ Creates comparison stimuli for figure 2
            Comparison stimuli, use middle_SLs = np.arange(-5,25,2.5)
    """
    s = fig2_settings
    durations = [s["tabDur"], s["tabDur"], s["tabDur"]]
    pads = [s["tabPad"], s["tabPad"], s["tabPad"]]
    sounds = {}
    for fn, tabSL in s["SLs"].items():    
        sounds[fn] = {}
        for middle_SL in middle_SLs:
            SLs = [tabSL[0], middle_SL, tabSL[0]]
            sounds[fn][middle_SL], audio_sr = gen.combine(s["freqs"], SLs, durations, pads, "")
    return sounds, s, audio_sr


def fig2_maskers(fig2_settings):
    """Create maskers only for each stimulus"""
    s = fig2_settings
    durations = [s["maskerDur"]]
    pads = [s["maskerPad"]]
    sounds = {}
    for k, v in s["SLs"].items():
        sounds[k], audio_sr = gen.combine([s["freqs"][1]], [v[1]], durations, pads)
    return sounds, s, audio_sr
