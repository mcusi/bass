import os
import yaml
from copy import deepcopy
import numpy as np
import scipy.signal
import scipy.interpolate

from psychophysics.hypotheses.mistuned_harmonic import de_attenuate
import psychophysics.generation.onset_asynchrony as gen
import psychophysics.hypotheses.hutil as hutil
from renderer.util import freq_to_ERB, ERB_to_freq, get_event_gp_freqs
from util import context

"""
Creation of initializations which are used in the onset asynchrony psychophysics experiment
1) Create isolated vowel stimuli which are used to create the modal explanations - equiv to basic and 0ms conditions
2) Determine the initialization for the isolated vowels
3) Combine tones with cached vowels into multiple initializations for each stimulus
"""


def full_design(sound_group, config_name, overwrite={}):

    with open(os.path.join(os.environ["config_dir"], config_name + '.yaml'), 'r') as f:
        hypothesis_config = yaml.load(f, Loader=yaml.FullLoader)

    with context(
        audio_sr=hypothesis_config["renderer"]["steps"]["audio_sr"],
        rms_ref=hypothesis_config["renderer"]["tf"]["rms_ref"]
    ):
        # 1. Create sounds for inference if not already made
        soundpath = os.path.join(os.environ["sound_dir"], sound_group, "")
        seed = overwrite["seed"]
        if not os.path.isdir(soundpath):
            print("Making ", soundpath)
            os.makedirs(soundpath, exist_ok=True)
        if not os.path.isfile(os.path.join(soundpath, "oa_expt1_settings.npy")):
            print("Making experiment 1 sounds...")
            _, expt1_settings = gen.expt1(sound_group, overwrite=overwrite)
        else:
            expt1_settings = np.load(
                os.path.join(soundpath + "oa_expt1_settings.npy"),
                allow_pickle=True
            ).item()
        if expt1_settings["audio_sr"] != context.audio_sr or expt1_settings["rms_ref"] != context.rms_ref:
            raise Exception("Conflicting config/settings for hypothesis defn.")

        # 2. Make cache of maskers and tabs for initialization
        cache = make_cache(hypothesis_config, expt1_settings, seed, overwrite)
        
    # 3. Use cache to make initializations 
    experiments = {}
    expt1_initializations = expt1(cache, expt1_settings, hypothesis_config)
    experiments.update(expt1_initializations)

    return experiments


def get_amplitudes(x, sr, rms_ref, f0):
    f, Pxx = scipy.signal.periodogram(x, fs=sr, scaling="spectrum")
    spectrum = 10*np.log10(2*Pxx/rms_ref**2)
    amplitudes = []; n_harmonics=49
    for i in range(1,n_harmonics):
        closest_freq = np.argmin(np.abs(f - f0*i))
        amplitudes.append(spectrum[closest_freq-2:closest_freq+2].max())
    return amplitudes


def create_spectrum(f0, amplitudes, config):
    audio_sr = config['renderer']['steps']['audio_sr']
    n_channels = len(get_event_gp_freqs(audio_sr, config['renderer']['steps']))  # 115
    channel_width = (freq_to_ERB(audio_sr/2. - 1.) - freq_to_ERB(config["renderer"]["lo_lim_freq"]))/(n_channels+1)  # 0.29680244
    n_harmonics = len(amplitudes)
    xy_small = np.array([
        [freq_to_ERB(f0) - channel_width,    amplitudes[0]],
        *[[freq_to_ERB(f0 * i),              de_attenuate(amplitudes[i-1], i, config)] for i in range(1,n_harmonics+1)],
        [freq_to_ERB(f0 * (n_harmonics+1)),  amplitudes[-1]],
        [freq_to_ERB(f0) + (n_channels-2) * channel_width, amplitudes[-1]]
    ])
    x_small = xy_small[:, 0] - xy_small[0, 0] + freq_to_ERB(config["renderer"]["lo_lim_freq"])
    y_small = xy_small[:, 1]
    spectrum = {}
    spectrum["x"] = np.arange(
        x_small[0],
        x_small[-1]+config["hypothesis"]["delta_gp"]["f"],
        config["hypothesis"]["delta_gp"]["f"]
        )
    interpol8r = scipy.interpolate.interp1d(
        x_small, 
        y_small,
        kind="cubic",
        bounds_error=False,
        fill_value=amplitudes[-1]
        )
    spectrum["y"] = interpol8r(spectrum["x"])
    spectrum["y"][spectrum["y"] < amplitudes[-1]] = amplitudes[-1]
    return spectrum


# 2) Cache of modal initializations
def make_cache(inference_config, stimulus_settings, seed, overwrite):

    # Load config 
    cache = {}

    # Create long vowels so we can very accurately estimate the spectrum
    # it should otherwise have the same settings as the experimental stimuli
    long_overwrite = deepcopy(overwrite)
    long_overwrite.update({"vowelDuration":1000, "onsets":[0.], "offsets":[0.]})
    long_vowel_dict, long_settings = gen.expt1("", overwrite=long_overwrite, save_wav=False)

    # Cache spectrum values
    # Keys: '{F1}_basic', '{F1}_on0_off0'
    actual_onset = stimulus_settings["pad_duration"] + max(stimulus_settings["onsets"])
    actual_offset = actual_onset + stimulus_settings["vowel_duration"]
    for F1 in stimulus_settings["F1s"]:
        for stimulus_type in ["basic", "on0_off0"]:
            k = "_".join([str(F1), stimulus_type])
            print(f"Caching seed {seed} for {k}", flush=True)
            # Load long vowel signal in order to accurately estimate spectrum
            long_observation = long_vowel_dict[f"{F1}_{stimulus_type}"]
            audio_sr = long_settings["audio_sr"]; rms_ref=long_settings["rms_ref"]; f0 = long_settings["f0"] 
            # Get onsets for long vowel so we can just look at that and not the padding
            long_onset = np.max(long_settings["onsets"]) + long_settings["pad_duration"]
            long_offset = long_onset + long_settings["vowel_duration"]
            long_onset += long_settings["rampDurations"][0]
            long_offset -= long_settings["rampDurations"][1]
            long_onset_idx = int(audio_sr*long_onset); long_offset_idx=int(audio_sr*long_offset)
            # Compute amplitudes of each harmonic peak
            amplitudes = get_amplitudes(long_observation[long_onset_idx:long_offset_idx], audio_sr, rms_ref, f0)
            #Interpolate to get an initial spectrum for the Scene, and set a constant amplitude
            spectrum_dict = create_spectrum(f0, amplitudes, inference_config)
            constant_amplitude_val = np.mean(spectrum_dict["y"])
            spectrum_dict["y"] -= constant_amplitude_val
            # Create hypothesis dict
            H = create_tone_hypothesis(
                actual_onset,
                actual_offset,
                stimulus_settings["f0"],
                constant_amplitude_val,
                "harmonic",
                inference_config
                )
            H["features"]["spectrum"] = spectrum_dict
            cache[k] = H

    return cache


def create_tone_hypothesis(onset, offset, f0, amplitude, source_type, config):
    onset = onset + config["renderer"]["ramp_duration"]
    offset = max(
        offset - config["renderer"]["ramp_duration"],
        onset + config["renderer"]["steps"]["t"]
        )
    ts = np.arange(
        onset + 0.001,
        offset + config["hypothesis"]["delta_gp"]["t"],
        config["hypothesis"]["delta_gp"]["t"]
        )
    source = {
        "source_type": source_type, 
        "events":[{
            "onset": onset,
            "offset": offset
            }], 
        "features":{
            "f0":{
                "x":ts,
                "y":np.full(ts.shape, freq_to_ERB(f0))
            }, 
            "amplitude":{
                "x":ts,
                "y":np.full(ts.shape, amplitude)
            }
            }
        }
    return source


def create_whistle_hypothesis(onset, offset, amplitude_idx, stimulus_settings, config, double=False, modify_amplitude=0):
    f0 = stimulus_settings["toneFrequency"] if double is False else stimulus_settings["toneFrequency"]*2
    source_type = "whistle"
    amplitude = stimulus_settings["amps_vowel_x2"][amplitude_idx] + modify_amplitude
    return  create_tone_hypothesis(onset, offset, f0, amplitude, source_type, config)

# 3) Combining cache and tone initializations for Expt 1
def vowel_only(F1, vowel_init_type, cache):
    return [cache[str(F1) + "_" + vowel_init_type]]


def basic_plus_overlapping_tone(F1, cache, stimulus_settings, config, double=False):
    v = deepcopy(cache[str(F1) + "_basic"])
    tone_idx = np.argmin(np.abs(ERB_to_freq(v["features"]["spectrum"]["x"]) - 500))
    # tone_idxs = [tone_idx-1, tone_idx, tone_idx+1]
    v["features"]["spectrum"]["y"][tone_idx] = 0
    tone_onset = stimulus_settings["pad_duration"] + max(stimulus_settings["onsets"])
    tone_offset = tone_onset + stimulus_settings["vowel_duration"]
    tone_duration = stimulus_settings["vowel_duration"]
    amplitude_idx = [i for i in range(len(stimulus_settings["F1s"])) if stimulus_settings["F1s"][i] == F1][0]
    return [v, create_whistle_hypothesis(tone_onset, tone_offset, amplitude_idx, stimulus_settings, config, double=double, modify_amplitude=-6)]

def onzero_plus_overlapping_tone(F1, cache, stimulus_settings, config, double=False):
    v = cache[str(F1) + "_basic"]
    tone_onset = stimulus_settings["pad_duration"] + max(stimulus_settings["onsets"])
    tone_offset = tone_onset + stimulus_settings["vowel_duration"]
    tone_duration = stimulus_settings["vowel_duration"]
    amplitude_idx = [i for i in range(len(stimulus_settings["F1s"])) if stimulus_settings["F1s"][i] == F1][0]
    return [v, create_whistle_hypothesis(tone_onset, tone_offset, amplitude_idx, stimulus_settings, config, double=double)]

def basic_plus_tone_to_end(F1, tone_duration, cache, stimulus_settings, config, double=False):
    v = cache[str(F1) + "_basic"]
    tone_onset = stimulus_settings["pad_duration"] + max(stimulus_settings["onsets"]) - tone_duration
    tone_offset = tone_onset + tone_duration + stimulus_settings["vowel_duration"]
    amplitude_idx = [i for i in range(len(stimulus_settings["F1s"])) if stimulus_settings["F1s"][i] == F1][0]
    return [v, create_whistle_hypothesis(tone_onset, tone_offset, amplitude_idx, stimulus_settings, config, double=double)]

def onzero_plus_tone_to_start(F1, tone_duration, cache, stimulus_settings, config, double=False):
    v = cache[str(F1) + "_on0_off0"]
    tone_onset = stimulus_settings["pad_duration"] + max(stimulus_settings["onsets"]) - tone_duration
    tone_offset = tone_onset + tone_duration
    amplitude_idx = [i for i in range(len(stimulus_settings["F1s"])) if stimulus_settings["F1s"][i] == F1][0]
    return [v, create_whistle_hypothesis(tone_onset, tone_offset, amplitude_idx, stimulus_settings, config, double=double)]

def basic_plus_tone_past(F1, tone_duration_on, tone_duration_off, cache, stimulus_settings, config, double=False):
    v = cache[str(F1) + "_basic"]
    tone_onset = stimulus_settings["pad_duration"] + max(stimulus_settings["onsets"]) - tone_duration_on
    tone_offset = tone_onset + tone_duration_on + stimulus_settings["vowel_duration"] + tone_duration_off
    amplitude_idx = [i for i in range(len(stimulus_settings["F1s"])) if stimulus_settings["F1s"][i] == F1][0]
    return [v, create_whistle_hypothesis(tone_onset, tone_offset, amplitude_idx, stimulus_settings, config, double=double)]

def onzero_plus_tone_past(F1, tone_duration_on, tone_duration_off, cache, stimulus_settings, config, double=False):
    v = cache[str(F1) + "_on0_off0"]
    tone_onset = stimulus_settings["pad_duration"] + max(stimulus_settings["onsets"]) - tone_duration_on
    tone_offset = tone_onset + tone_duration_on + stimulus_settings["vowel_duration"] + tone_duration_off
    amplitude_idx = [i for i in range(len(stimulus_settings["F1s"])) if stimulus_settings["F1s"][i] == F1][0]
    return [v, create_whistle_hypothesis(tone_onset, tone_offset, amplitude_idx, stimulus_settings, config, double=double)]

def expt1(cache, stimulus_settings, config):

    experiments = {}
    for F1 in stimulus_settings["F1s"]:
        # Only test vowel only conditions for basic and on0_off0 conditions
        experiments.update(hutil.format(vowel_only(F1, "basic", cache), f"{F1}_basic", "vowel"))
        experiments.update(hutil.format(vowel_only(F1, "on0_off0", cache), f"{F1}_on0_off0", "vowel"))

        if stimulus_settings.get("onsets_and_offsets", None) is not None:
            onsets_and_offsets = stimulus_settings["onsets_and_offsets"]
        else:
            onsets_and_offsets = [(onset, offset) for onset in stimulus_settings["onsets"] for offset in stimulus_settings["offsets"]]

        for onset, offset in onsets_and_offsets:
            if onset == 0 and offset == 0:  # (Don't only use) vowel_only initialization above.
                H = basic_plus_overlapping_tone(F1, cache, stimulus_settings, config)
                experiments.update(hutil.format(H, f"{F1}_basic", "vowel+tone"))
                H = onzero_plus_overlapping_tone(F1, cache, stimulus_settings, config)
                experiments.update(hutil.format(H, f"{F1}_on0_off0", "vowel+tone"))
            elif offset == 0:
                H = basic_plus_tone_to_end(F1, onset, cache, stimulus_settings, config)
                experiments.update(hutil.format(H, f"{F1}_on{int(1000*onset)}_off0", "basic-init"))
                H = onzero_plus_tone_to_start(F1, onset, cache, stimulus_settings, config)
                experiments.update(hutil.format(H, f"{F1}_on{int(1000*onset)}_off0", "onzero-init"))
            elif offset > 0:
                H = basic_plus_tone_past(F1, onset, offset, cache, stimulus_settings, config)
                experiments.update(hutil.format(H, f"{F1}_on{int(1000*onset)}_off{int(1000*offset)}", int(1000*offset)), "basic-init"))
                H = onzero_plus_tone_past(F1, onset, offset, cache, stimulus_settings, config)
                experiments.update(hutil.format(H, f"{F1}_on{int(1000*onset)}_off{int(1000*offset)}", "onzero-init"))

    return experiments