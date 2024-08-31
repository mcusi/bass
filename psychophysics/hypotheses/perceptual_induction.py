import os
import yaml
import numpy as np
import scipy.signal

import psychophysics.hypotheses.hutil as hutil
import psychophysics.generation.perceptual_induction as gen
import psychophysics.generation.noises as noisegen
import psychophysics.generation.basic as basicgen
from renderer.util import freq_to_ERB
from util import context


def full_design(audio_folder, hypothesis_config_name, overwrite={}):

    with open(os.path.join(os.environ["config_dir"], hypothesis_config_name + '.yaml'), 'r') as f:
        hypothesis_config = yaml.load(f, Loader=yaml.FullLoader)
    
    with context(
        audio_sr=hypothesis_config["renderer"]["steps"]["audio_sr"],
        rms_ref=hypothesis_config["renderer"]["tf"]["rms_ref"]
    ):
        seed = overwrite.get("seed", 0)
        # 1. Create sounds for inference if not already made
        if not os.path.isfile(os.path.join(
            os.environ["sound_dir"],
            audio_folder,
            f"pi_expt3_settings_seed{seed}.npy"
        )):
            os.makedirs(os.path.join(os.environ["sound_dir"], audio_folder, ""), exist_ok=True)
            _, settings = gen.expt3(audio_folder, overwrite=overwrite)
        else:
            settings = np.load(os.path.join(os.environ["sound_dir"], audio_folder, f"pi_expt3_settings_seed{seed}.npy"), allow_pickle=True).item()
        if settings["audio_sr"] != context.audio_sr or settings["rms_ref"] != context.rms_ref:
            raise Exception("Conflicting config and settings for hypothesis definition.")

        maskerRMS = settings["rms_ref"]*np.power(10, settings["maskerLevel"]/20.)
        masker = noisegen.pink(settings["duration"], settings["maskerLevel"])
        b, a = scipy.signal.butter(
            settings["butter_ord"],
            [settings["fl"]/(settings["audio_sr"]/2.), settings["fh"]/(settings["audio_sr"]/2.)],
            btype='bandstop'
            )
        masker = scipy.signal.filtfilt(b, a, masker)
        masker = basicgen.raisedCosineRamps(
            masker*maskerRMS/np.sqrt(np.mean(np.square(masker))),
            settings["rampDuration"]
            )
        settings["xf"], Pxx_den = scipy.signal.periodogram(masker, fs=settings["audio_sr"])
        settings["Pxx_density"] = Pxx_den / hypothesis_config["renderer"]["tf"]["rms_ref"]**2

    experiments = {}
    for experiment_style in ["masking", "continuity"]:
        for f0 in settings["tone_freqs"]:
            for dB in settings["tone_levels"]:
                observation_name = f"{experiment_style}_f{f0:04d}_l{dB:03d}_seed{seed}"
                if experiment_style == "masking":
                    E1 = create_present_tones_hypothesis(dB, f0, settings, hypothesis_config)
                    E2 = create_absent_tones_hypothesis(settings, hypothesis_config)
                    experiments.update(hutil.format(E1, observation_name, "present"))
                    experiments.update(hutil.format(E2, observation_name, "absent"))
                elif experiment_style == "continuity":
                    E1 = create_continuous_tones_hypothesis(dB, f0, settings, hypothesis_config)
                    E2 = create_discontinuous_tones_hypothesis(dB, f0, settings, hypothesis_config)
                    experiments.update(hutil.format(E1, observation_name, "continue"))
                    experiments.update(hutil.format(E2, observation_name, "discontinue"))

    return experiments

# Helpers to generate hypotheses


def create_event(onset, offset, amplitude, config, f0=None):
    onset = onset + config["renderer"]["ramp_duration"]
    offset = offset - config["renderer"]["ramp_duration"]
    ts = np.arange(onset + 0.001, offset + config["hypothesis"]["delta_gp"]["t"], config["hypothesis"]["delta_gp"]["t"])
    feature_dict = {
        "amplitude": {
            "x": ts,
            "y": np.full(ts.shape, amplitude)
        }}
    if f0 is not None:
        feature_dict["f0"] = {"x": ts, "y": np.full(ts.shape, freq_to_ERB(f0))}
    return {
        "event": {"onset": onset, "offset": offset},
        "features": feature_dict
        }


def create_whistle_event(onset, offset, amplitude, f0, config):
    return create_event(onset, offset, amplitude, config, f0=f0)


def create_noise_event(onset, offset, amplitude, config):
    return create_event(onset, offset, amplitude, config)


def create_whistle_source(temporal, amplitude, f0, config):

    events = [create_whistle_event(t[0], t[1], amplitude, f0, config) for t in temporal]

    feature_dict = {}
    for feature in ["f0", "amplitude"]:
        feature_dict[feature] = {}
        for z in ["x", "y"]:
            feature_dict[feature][z] = np.concatenate([event["features"][feature][z] for event in events])

    source = {
        "source_type": "whistle",
        "events": [event["event"] for event in events],
        "features": feature_dict
        }

    return source


def masker_hypothesis(settings, config, n_events=None):
    def spectrum_level_function(f):
        center = np.argmin(np.abs(f - settings["xf"]))
        return 10*np.log10( np.mean(settings["Pxx_density"][center-3:center+3]) )
    erbs, spectrum, amplitude = hutil.create_noise_latents(
        spectrum_level_function, config, settings["audio_sr"]
        )
    d = {"x": erbs, "y": spectrum}
    if n_events is not None:
        d['x_events'] = [d['x'] for e in range(n_events)]
        d['y_events'] = [d['y'] for e in range(n_events)]
    return d, amplitude


def create_noise_source(temporal, config, gen_settings):

    spectrum_dict, amplitude = masker_hypothesis(gen_settings, config, n_events=len(temporal))
    events = [create_noise_event(t[0], t[1], amplitude, config) for t in temporal]

    feature_dict = {"amplitude": {}, "spectrum": {}}
    for z in ["x", "y"]:
        feature_dict["amplitude"][z] = np.concatenate([event["features"]["amplitude"][z] for event in events])
    feature_dict["spectrum"] = spectrum_dict

    source = {
        "source_type": "noise",
        "events": [event["event"] for event in events],
        "features": feature_dict
        }

    return source


"""
== Continuity ==
Explanation 1. Continuous tone through the noises
Explanation 2. Discontinuous tones alternating with noise

"""


def continuity_noise(event_duration, pad, config, gen_settings):
    onset_0 = pad
    temporal = []
    for i in range(gen_settings["nRepetitions"]+2):
        onset = onset_0+i*2*event_duration
        temporal.append((onset, onset+event_duration))
    return create_noise_source(temporal, config, gen_settings)


def create_continuous_tones_hypothesis(amplitude, f0, settings, config):
    event_duration = settings["duration"]
    pad = settings["pad"]
    onset = pad + 1.5*event_duration
    offset = onset + 6*event_duration
    temporal = [(onset, offset)]
    masked = create_whistle_source(temporal, amplitude, f0, config)
    masker = continuity_noise(event_duration, pad, config, settings)
    return [masked, masker]


def create_discontinuous_tones_hypothesis(amplitude, f0, settings, config):
    event_duration = settings["duration"]
    pad = settings["pad"]
    onset_0 = pad + 1.5*event_duration
    temporal = []
    for i in range(settings["nRepetitions"]+1):
        if i == 0:
            temporal.append((onset_0, onset_0 + 0.5*event_duration))
        elif i == settings["nRepetitions"]:
            onset = onset_0 + (i*2 - 0.5)*event_duration
            temporal.append((onset, onset + 0.5*event_duration))
        else:
            onset = onset_0 + (i*2 - 0.5)*event_duration
            temporal.append((onset, onset + event_duration))
    masked = create_whistle_source(temporal, amplitude, f0, config)
    masker = continuity_noise(event_duration, pad, config, settings)
    return [masked, masker]


"""
== Masking ==
Explanation 1. No tones
Explanation 2. Tones embedded in noise
"""


def masking_noise(event_duration, pad, config, gen_settings):
    temporal = [(pad, 6*event_duration + pad)]
    return create_noise_source(temporal, config, gen_settings)


def create_present_tones_hypothesis(amplitude, f0, settings, config):
    pad = settings["pad"]
    # See "Instead of random..." comment in gen.toneMasking for event_duration/2.
    temporal = []
    event_duration = settings["duration"] + 2*settings["rampDuration"]
    for i in [0, 2, 4]:
        onset = i*event_duration + pad + settings["duration"]/2.
        offset = (i+1)*event_duration + pad + settings["duration"]/2.
        temporal.append((onset, offset))
    masked = create_whistle_source(temporal, amplitude, f0, config)
    masker = masking_noise(settings["duration"], pad, config, settings)
    return [masked, masker]


def create_absent_tones_hypothesis(settings, config):
    event_duration = settings["duration"]
    pad = settings["pad"]
    return [masking_noise(event_duration, pad, config, settings)]
