import os
import numpy as np
import soundfile as sf
import sys

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic
from util import context

"""
Stimuli from: Moore, B.C.J., Glasberg, B. R., and Peters, R. W. (1986). 
Thresholds for hearing mistuned partials as separate tones in harmonic complexes. 
The Journal of the Acoustical Society of America, 80, 479.
Also used in Moore, Peters, & Glasberg (1985b); Hartmann, McAdams & Smith (1990)

Naming format
`f{fundamental frequency in Hz}_dur{duration in milliseconds}_harm{index of mistuned harmonic}_p{mistuned percent of f0}_dB{level of mistuned harmonic}.wav`
Index of mistuned harmonic starts at 1, referring to the fundamental frequency
For mistuned percent of f0, `0` means perfectly harmonic

"""


def mistunedComplex(
        f0,
        duration,
        n_harmonics,
        mistunedIdx,
        mistunedPercent,
        ramp_duration,
        harmonicLevel,
        mistunedLevel
        ):
    """ Create mistuned harmonic tone """
    harmonics = np.ones(n_harmonics)
    harmonics[mistunedIdx-1] = 0
    harmonic = tones.harmonic(
        duration + ramp_duration*2,
        f0,
        harmonics,
        levels=harmonicLevel*np.ones(n_harmonics)
    )
    mistunedFreq = mistunedIdx*f0 + mistunedPercent*f0
    mistuned = tones.pure(
        duration + ramp_duration*2,
        mistunedFreq,
        level=mistunedLevel
    )
    complexTone = basic.raisedCosineRamps(harmonic + mistuned, ramp_duration)
    mistunedOnly = basic.raisedCosineRamps(mistuned, ramp_duration)
    harmonicOnly = basic.raisedCosineRamps(harmonic, ramp_duration)
    return complexTone, mistunedOnly, harmonicOnly


def expt_settings(overwrite={}):
    settings = {}
    settings["fundamentals"] = [100, 200, 400]  # Hz
    settings["n_harmonics"] = [12, 12, 10]  # for each respective fundamental
    settings["durations"] = [400]  # ms, steady state duration
    # percentages: moore, glasberg, peters 1985 a)
    settings["mistuned_percents"] = [0, 5, 10, 20, 30, 40, 50]
    settings["harmonic_level"] = 60
    settings["mistuned_level"] = 60
    settings["ramp_duration"] = 0.010  # s
    settings["pad"] = 0.050  # s
    settings["mistuned_idxs"] = range(1, 4)
    for k in settings.keys():
        if k in overwrite.keys():
            settings[k] = overwrite[k]
    return settings


def expt(audio_folder, overwrite={}, save_wav=True, return_all_components=False):

    settings = expt_settings(overwrite)
    if save_wav:
        path = os.path.join(os.environ["sound_dir"], audio_folder, "")
        os.makedirs(os.environ["sound_dir"] + audio_folder, exist_ok=True)

    expt_dict = {}
    all_components_dict = {}
    for i in range(len(settings["fundamentals"])):
        f0 = settings["fundamentals"][i]
        nH = settings["n_harmonics"][i]
        for d in settings["durations"]:
            duration = d/1000.
            # tested thresholds for first 6 harmonics
            for mistunedIdx in settings["mistuned_idxs"]:
                for mtP in settings["mistuned_percents"]:
                    mistunedPercent = mtP/100.
                    mistunedLevel = settings["mistuned_level"]
                    s, mistuned_tone_only, harmonic_only = mistunedComplex(
                        f0, duration, nH, mistunedIdx, mistunedPercent,
                        settings["ramp_duration"], settings["harmonic_level"],
                        mistunedLevel
                    )
                    s = basic.addSilence(s, settings["pad"])
                    mistuned_tone_only = basic.addSilence(
                        mistuned_tone_only, settings["pad"]
                    )
                    harmonic_only = basic.addSilence(harmonic_only, settings["pad"])
                    fn = f'f0{f0}_dur{d}_harm{mistunedIdx}_p{mtP}_dB{mistunedLevel}'
                    if save_wav:
                        sf.write(path + fn + ".wav", s, context.audio_sr)
                    expt_dict[fn] = s
                    all_components_dict[fn] = (s, mistuned_tone_only, harmonic_only)
    if save_wav:
        settings["audio_sr"] = context.audio_sr
        settings["rms_ref"] = context.rms_ref
        np.save(path + "/mh_expt_settings.npy", settings)

    if return_all_components:
        return all_components_dict, settings
    else:
        return expt_dict, settings


if __name__ == "__main__":
    print('Moore, Glasberg, and Peters (1986)')
    audio_fp = sys.argv[1]
    os.makedirs(audio_fp, exist_ok=False)
    with context(audio_sr=20000, rms_ref=1e-6):
        expt(audio_fp)
    print('~~~~')
