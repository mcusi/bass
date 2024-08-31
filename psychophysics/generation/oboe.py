import os
import sys
import yaml
import numpy as np
import soundfile as sf

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic
from util import context

"""
Create the classic Reynolds-McAdams oboe demo and single interval variant
"""

def classic_settings(overwrite={}):
    """Settings to make classic Reynolds-McAdams Oboe demo"""
    settings = {
        "tone_type": "harmonic", #harmonic
        "duration": 5.0, #sec
        "f0": 200, #Hz
        "levels": np.array([-25, -37, -44, -48, -52, -52, -58, -54, -62, -55, -65, -61, -67, -65, -69, -70]), #dB, level of each harmonic
        "n_harmonics": 16,
        "steady_duration_1": 0.6, #sec - modulation starts to increase at this point
        "steady_duration_2": 2.75, #sec - modulation reaches its peak then maintains
        "max_delta_f": 7, #Hz
        "f_m": 5, #Hz, frequency of modulation
        "ramp_duration": 0.020, #s
        "pad_duration": 0.0
    }
    for k in settings.keys():
        if k in overwrite.keys():
            settings[k] = overwrite[k]
    if settings["n_harmonics"] > len(settings["levels"]):
        d = settings["n_harmonics"]-len(settings["levels"])
        settings["levels"] = np.concatenate((settings["levels"], np.full((d,), settings["levels"][-1])-np.arange(d)))
    settings["levels"] += 100
    settings["increment_duration"] = settings["duration"] - (settings["steady_duration_1"] + settings["steady_duration_2"])
    return settings


def classic(audio_folder, overwrite={}, save_wav=True):

    settings = classic_settings(overwrite)
    if save_wav:
        path = os.environ["sound_dir"] + audio_folder + "/"
        os.makedirs(path, exist_ok=True)
        settings["audio_sr"] = context.audio_sr
        settings["rms_ref"] = context.rms_ref
        np.save(path + "classic_settings.npy", settings)

    tone = basic.silence(settings["duration"])
    set1_tone = basic.silence(settings["duration"])
    set2_tone = basic.silence(settings["duration"])
    t = np.arange(0,settings["duration"], 1./context.audio_sr)
    for i in range(settings["n_harmonics"]):
        f_c = settings["f0"]*(i+1)
        level = settings["levels"][i]
        if i % 2 != 0:
            # Even harmonics have FM applied
            df1 = basic.silence(settings["steady_duration_1"])
            df2 = (i+1)*np.linspace(0, settings["max_delta_f"], int(settings["increment_duration"]*context.audio_sr))
            last_sample = df2[-1] if len(df2) > 0 else (i+1)*settings["max_delta_f"]  # len(df2) == 0 for single_interval
            df3 = last_sample*np.ones(len(t) - len(df1) - len(df2))
            delta_f = np.concatenate((df1, df2, df3))
            phi_c = 0
            phi_m = np.pi/2  # pi/2 allows a smooth transition in "classic", and to start harmonically in "single_interval"
            set1_only = tones.sinusoidalFM(settings["duration"], f_c, phi_c, settings["f_m"], phi_m, delta_f, level)

            set1_tone += set1_only
            tone += set1_only

        else: 
            # Odd harmonics          
            set2_only = tones.pure(settings["duration"], f_c, level=level, phase=2*np.pi*np.random.rand())  
            set2_tone += set2_only        
            tone += set2_only

    tone = basic.linearRamps(tone, settings["ramp_duration"])
    set1_tone = basic.linearRamps(set1_tone, settings["ramp_duration"])
    set2_tone =  basic.linearRamps(set2_tone, settings["ramp_duration"])
    if settings["pad_duration"] > 0.0:
        tone = basic.addSilence(tone, settings["pad_duration"])
        set1_tone = basic.addSilence(set1_tone, settings["pad_duration"])
        set2_tone = basic.addSilence(set2_tone, settings["pad_duration"])

    if save_wav:
        fn = 'reynolds-oboe_f{:04d}_fm{:02d}_df{:04d}_nh{}_'.format(settings["f0"], settings["f_m"], settings["max_delta_f"], settings["n_harmonics"])
        sf.write(path + fn + 'mix.wav', tone, context.audio_sr)
        sf.write(path + fn + 'pm1.wav', set1_tone, context.audio_sr)
        sf.write(path + fn + 'pm2.wav', set2_tone, context.audio_sr)

    return (tone, set1_tone, set2_tone), settings


def single_interval(audio_folder, duration, overwrite={}):
    overwrite_duration = {
        "duration":duration,
        "steady_duration_1": 0.0,
        "steady_duration_2": duration
    }
    overwrite_duration.update(overwrite)
    return classic(audio_folder, overwrite_duration)


if __name__ == "__main__":

    config_name = sys.argv[1]
    sound_group = sys.argv[2]
    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with context(
            audio_sr=config["renderer"]["steps"]["audio_sr"],
            rms_ref=config["renderer"]["tf"]["rms_ref"]
    ):
        single_interval(
            sound_group,
            1.0,
            overwrite={
                "f0": 300,
                "f_m": 2,
                "n_harmonics": int(12),
                "max_delta_f": 70,
                "pad_duration": 0.050
            }
        )
