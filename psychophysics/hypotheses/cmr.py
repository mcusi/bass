
import os
import yaml
from glob import glob
import numpy as np
import scipy.interpolate

import renderer.util
import psychophysics.generation.cmr as gen
import psychophysics.hypotheses.hutil as hutil
from util import context


def full_design(audio_folder, config_name, overwrite={}):
    """Return all stimuli and initial hypotheses for CMR experiments"""

    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    with context(
        audio_sr=config["renderer"]["steps"]["audio_sr"],
        rms_ref=config["renderer"]["tf"]["rms_ref"]
    ):
        # 1. Create sounds for inference if not already made
        seed = overwrite["seed"]
        settings_fn = os.path.join(
            os.environ["sound_dir"],
            audio_folder,
            f"cmr_expt1_settings_seed{seed}.npy"
        )
        if not os.path.isfile(settings_fn):
            print("Generating seed ", seed, flush=True)
            os.makedirs(os.path.join(
                os.environ["sound_dir"], audio_folder, ""
            ), exist_ok=True)
            _, settings = gen.expt1(
                audio_folder, overwrite=overwrite, seed=seed
                )
        else:
            settings = np.load(settings_fn, allow_pickle=True).item()
        if settings["audio_sr"] != context.audio_sr or settings["rms_ref"] != context.rms_ref:
            raise Exception("Conflicting config and settings for hypothesis definition.")

        # 2. Make caches for all (noise-only) sounds + (noise) hypothesis 
        print("Making cache", flush=True)
        cache = make_cache(audio_folder, config, settings, seed)

    # 3. Compile the stimuli, along with initial-hypotheses for each
    stimulus_tone_levels = settings["tone_levels"]
    possible_explanation_types = ["noise-only", "noise-tone"]
    experiments = {}
    for exemplar in range(settings["n_trials"]):
        for sound_type in ["mult", "rand"]:
            for bandwidth in settings["bandwidths"]:
                for tone_level_in_stimulus in sorted(stimulus_tone_levels): 
                    for explanation_type in possible_explanation_types:
                        if "tone" in explanation_type:
                            init_tone_levels_in_explanation = stimulus_tone_levels if tone_level_in_stimulus == 0 else [0.0, tone_level_in_stimulus]
                            for init_tone_level_in_explanation in init_tone_levels_in_explanation:
                                experiments.update(make(cache, exemplar, sound_type, bandwidth, tone_level_in_stimulus, explanation_type, settings, config, seed, init_tone_level_in_explanation=init_tone_level_in_explanation))
                        else:
                            experiments.update(make(cache, exemplar, sound_type, bandwidth, tone_level_in_stimulus, explanation_type, settings, config, seed))

    return experiments

def make(cache, exemplar, sound_type, bandwidth, tone_level_in_stimulus, explanation_type, settings, config, seed, init_tone_level_in_explanation=None):
    """ Gets an initial explanation for a single stimulus and explanation """
    if tone_level_in_stimulus == 0:
        sound_name = f"{sound_type}_noTone_bw{bandwidth}_seed{seed}-{exemplar}"
    else:
        sound_name = f"{sound_type}_tone_bw{bandwidth}_toneLevel{tone_level_in_stimulus}_seed{seed}-{exemplar}"
    if sound_type == "mult":
        cache_key = f"mult_noTone_bw{bandwidth}_seed{seed}-{exemplar}"
    elif sound_type == "rand":
        cache_key = f"rand_noTone_bw{bandwidth}_seed{seed}-{exemplar}"
    
    cached_explanation = cache[cache_key]
    if explanation_type == "noise-only":
        # Returns [Source]
        explanation = create_tone_absent_explanation(cached_explanation)
        return hutil.format(explanation, sound_name, "noise-only")
    elif explanation_type == "noise-tone":
        # Returns [Source, dict]
        explanation = create_tone_present_explanation(init_tone_level_in_explanation, cached_explanation, settings,config)
        return hutil.format(explanation, sound_name, f"noise+tone-{init_tone_level_in_explanation}dB")
    elif explanation_type == "tone-only":
        # Returns [dict]
        explanation = [create_whistle_source(init_tone_level_in_explanation, settings, config)]
        return hutil.format(explanation, sound_name, f"tone-only-{init_tone_level_in_explanation}dB")


def make_cache(audio_folder, config, settings, seed):
    """Determine initialization for the CMR maskers alone and cache."""

    audio_sr = settings["audio_sr"]

    def make_sl_func(bw):
        def spectrum_level_function(f):
            bottom = (settings["cf"] - bw/2)/1.1
            top = (settings["cf"] + bw/2)*1.1
            if f < bottom:
                return 0
            elif f >= top:
                return -20
            elif bottom <= f < top:
                return settings["noise_spectrum_level"]
        return spectrum_level_function

    cache = {}
    # Make a cache for each mult_noise masker
    wavs = glob(os.path.join(
        os.environ["sound_dir"],
        audio_folder,
        f"mult_noTone*seed{seed}*.wav"
    ))
    for w in wavs:
        cache_key = w.split(os.sep)[-1][:-4]
        bandwidth = [float(sub_string[2:]) for sub_string in w.split("_") if "bw" in sub_string][0]
        lpn = np.load(w[:-4] + ".npy")
        slf = make_sl_func(bandwidth)
        erbs, spectrum, amplitude_mean = hutil.create_noise_latents(slf, config, audio_sr)
        amplitude = 10 * np.log10(1e-8 + lpn**2)
        amplitude = amplitude + amplitude_mean
        cache[cache_key] = create_noise_initialization(erbs, spectrum, amplitude, config, settings, audio_sr)

    #Make a cache for rand noises
    wavs = glob(os.path.join(
        os.environ["sound_dir"],
        audio_folder,
        f"rand_noTone*seed{seed}*.wav"
    ))
    for w in wavs:
        cache_key = w.split(os.sep)[-1][:-4]
        bandwidth = [float(sub_string[2:]) for sub_string in w.split("_") if "bw" in sub_string][0]
        slf = make_sl_func(bandwidth)
        erbs, spectrum, amplitude_mean = hutil.create_noise_latents(slf, config, audio_sr)
        amplitude = np.full(lpn.shape, amplitude_mean)
        cache[cache_key] = create_noise_initialization(erbs, spectrum, amplitude, config, settings, audio_sr)

    return cache


def create_noise_initialization(fs, spectrum, amplitude, config, settings, audio_sr):
    onset = settings["silence_padding"] 
    offset = onset + settings["noise_duration"]
    source, ts = create_source(onset, offset, "noise", config)
    amplitudeInterpol8r = scipy.interpolate.interp1d(
        np.arange(amplitude.shape[0])/audio_sr,
        amplitude
    )
    source["features"] = {
        "amplitude": {
            "x": ts,
            "y": amplitudeInterpol8r(ts)
        }, 
        "spectrum": {"x": fs, "y": spectrum}
    }
    return source


# Helper functions to create the scene hypotheses
def create_source(onset, offset, source_type, config):
    onset = onset + config["renderer"]["ramp_duration"]
    offset = offset - config["renderer"]["ramp_duration"]
    ts = np.arange(
        onset + 0.001,
        offset + config["hypothesis"]["delta_gp"]["t"],
        config["hypothesis"]["delta_gp"]["t"]
    )
    return {
        "source_type": source_type,
        "events": [{"onset": onset, "offset": offset}],
        "features": {}
    }, ts


def create_whistle_source(amplitude, settings, config):
    onset = settings["silence_padding"]
    # noise_duration = (steady_tone_duration=0.3 + 2*ramp_duration=2*0.050)
    offset = onset + settings["noise_duration"]
    s, ts = create_source(onset, offset, "whistle", config)
    s["features"]["f0"] = {
        "x": ts,
        "y": np.full(ts.shape, renderer.util.freq_to_ERB(settings["cf"]))
    }
    s["features"]["amplitude"] = {
        "x": ts,
        "y": np.full(ts.shape, amplitude)
    }
    return s


def create_tone_present_explanation(tone_amplitude, masker_cache, settings, config):
    return [masker_cache, create_whistle_source(tone_amplitude, settings, config)]


def create_tone_absent_explanation(masker_cache):
    return [masker_cache]
