import os
import yaml
import sys
import numpy as np
import soundfile as sf

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic  
from util import context

"""
Cancelled harmonics demonstration
"""

def classic_settings(overwrite={}):
    """Original settings from the acoustical society of america disk"""
    settings = {
        # Temporal settings
        "duration_full": 7.0,  # sec
        "duration_tone": 1.0,  # sec
        "n_tones": 4,
        # Spectral settings
        "f0": 200,  # Hz
        "n_harmonics": 20,
        "cancelled_harmonic_idxs": range(1, 10),
        "level": 70,
        "equal_amplitude_harmonics": False,
        # Basic settings
        "ramp_duration": 0.010,
        "silence_pad": 0.050
    }
    for k in settings.keys():
        if k in overwrite.keys():
            settings[k] = overwrite[k]
    settings["duration_silence"] = (settings["duration_full"] - (settings["duration_tone"]*settings["n_tones"]))/(settings["n_tones"]-1)
    assert settings["duration_silence"] > 0
    if settings["equal_amplitude_harmonics"]:
        settings["levels"] = np.full((settings["n_harmonics"]), settings["level"])
        settings["tone_level"] = settings["level"]
    else:
        settings["levels"] = -np.arange(settings["n_harmonics"]) + settings["level"]
    return settings

def hartmann_settings():
    #Enhancing and unmasking the harmonics of a complex tone (2006)
    overwrite = {
        "duration_full": 9.171,
        "duration_tone": 1.31072,  # sec
        "n_tones": 4,
        "f0": lambda: np.random.randint(int(200*0.95), int(200*1.05)),  # Hz, but randomized over a 5% range
        "n_harmonics": 30,
        "cancelled_harmonic_idxs": range(1, 21),  # 1 means fundamental
        "level": 45,
        "equal_amplitude_harmonics": True,
        "phase_difference": 0  # components added in sine phase
    }
    return overwrite

def generate(audio_fp, settings_name, overwrite={}, save_wav=True):

    if settings_name == "hartmann":
        hartmann_overwrite = hartmann_settings()
        hartmann_overwrite.update(overwrite)
        settings = classic_settings(overwrite=hartmann_overwrite)
    else:
        settings = classic_settings(overwrite=overwrite)

    for harmonic_idx in settings["cancelled_harmonic_idxs"]:
        # Update settings
        settings["use_harmonics"] = np.ones((settings["n_harmonics"],))
        settings["use_harmonics"][harmonic_idx-1] = 0
        if not settings["equal_amplitude_harmonics"]:
            settings["tone_level"] = settings["levels"][harmonic_idx-1]

        # Create the harmonic
        H = tones.harmonic(settings["duration_full"], settings["f0"], settings["use_harmonics"], levels=settings["levels"])
        H = basic.raisedCosineRamps(H,settings["ramp_duration"])
        # Create the tone
        T_base = tones.pure(settings["duration_tone"], settings["f0"]*harmonic_idx, level=settings["tone_level"])
        T_base = basic.raisedCosineRamps(T_base,settings["ramp_duration"])
        T = T_base.copy()
        # Create the silence
        S = np.zeros((int(np.round(context.audio_sr*settings["duration_silence"])),))
        # Put it all together
        for tone_idx in range(settings["n_tones"]-1):
            if tone_idx == settings["n_tones"] - 2:
                S = np.zeros( ( len(H) - len(T) - len(T_base), ) )
            T = np.concatenate((T, S, T_base))
        cancelled_harmonic_sound = T+H
        cancelled_harmonic_sound = basic.addSilence(cancelled_harmonic_sound,[settings["silence_pad"], settings["silence_pad"]])
        if save_wav:
            sf.write(
                os.path.join(audio_fp, f"cancelled_f{settings['f0']:04d}_n{harmonic_idx:02d}_durf{int(1000*settings['duration_full']):04d}_durt{int(settings['duration_tone']*1000):04d}_nt{settings['n_tones']:02d}.wav"), 
                cancelled_harmonic_sound, context.audio_sr
                )

    if save_wav:
        settings["audio_sr"] = context.audio_sr
        settings["rms_ref"] = context.rms_ref
        np.save(os.path.join(audio_fp, "classic_settings.npy"), settings)

    return cancelled_harmonic_sound, settings


if __name__ == "__main__":
    if len(sys.argv[1:]) == 2:
        config_name = sys.argv[1]
        sound_group = sys.argv[2]
        with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
            hypothesis_config = yaml.load(f, Loader=yaml.FullLoader)
            audio_sr = hypothesis_config["renderer"]["steps"]["audio_sr"]
            rms_ref = hypothesis_config["renderer"]["tf"]["rms_ref"]
        audio_fp = os.environ["sound_dir"] + sound_group + "/"
    else:
        audio_fp = sys.argv[1]
        audio_sr = 20000
        rms_ref = 1e-6
    os.makedirs(audio_fp, exist_ok=False)
    
    with context(audio_sr=audio_sr, rms_ref=rms_ref):
        f0s = np.linspace(int(200*0.95), int(200*1.05), 5)
        for f0 in f0s:
            overwrite = {
                    "duration_full": 0.75,
                    "duration_tone": 0.1,
                    "n_tones": 5,
                    "f0": int(f0),
                    "cancelled_harmonic_idxs": [1, 2, 3, 10, 11, 12, 18, 19, 20],
                    "level": 70
                }
            generate(audio_fp, "hartmann", overwrite=overwrite)
