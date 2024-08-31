import os
import sys
import numpy as np
import soundfile as sf

import psychophysics.generation.basic as basic  
import psychophysics.generation.noises as noises
from util import context, manual_seed

"""
McDermott, J. H., & Oxenham, A. J. (2008).
Spectral completion of partially masked sounds.
Proceedings of the National Academy of Sciences, 105(15), 5939-5944.
"""


def bp(band, spectrumLevel, duration, pad):
    """
    stimuli gated on and off with 10ms raised cosine ramps
    """
    noise = noises.bpWhiteSpecLevel(duration, spectrumLevel, band)
    noise = basic.raisedCosineRamps(noise, 0.010)
    noise = basic.addSilence(noise, pad)
    return noise


def combine(freqs, SLs, durations, pads, fn=""):

    if type(pads[0]) is float:
        totalDur = durations[0] + 2*pads[0]
    elif len(pads[0]) == 2:
        totalDur = durations[0] + sum(pads[0])
    noise = basic.silence(totalDur)

    # Create full sound
    for i in range(len(freqs)):
        part = bp(freqs[i], SLs[i], durations[i], pads[i])
        noise += part
    
    if len(fn) > 0:
        sf.write(f'{fn}.wav', noise, context.audio_sr)

    return noise, context.audio_sr


def fig1_settings(overwrite={}):
    settings = {
        "SL": 20,  # dB/hz
        "lowBand": [100, 500],
        "midBand": [500, 2500],
        "highBand": [2500, 7500],
        "maskerDur": 0.750,
        "maskerPad": 0.1,
        "tabDur": 0.150,
        "tabPad": 0.4,
        "middleSLs": np.arange(-15, 40, 2.5),
        "ii": {"midBandLevel": 30},
        "iv": {
            "bottomMidBand": [500, 600],
            "topMidBand": [2080, 2500],
            "SLshift": np.log10(2000./520.)
            },
        "v": {"duration": 0.300}
    }
    for k in settings.keys():
        if k in overwrite.keys():
            settings[k] = overwrite[k]
    settings["stdFreqs"] = [
        settings["lowBand"], settings["midBand"], settings["highBand"]
    ]
    settings["fullDur"] = settings["maskerDur"] + 2*settings["maskerPad"]
    return settings


def fig1(path, overwrite={}):

    print('Figure 1')
    seed = overwrite.get("seed", 0)
    manual_seed(seed)
    s = fig1_settings(overwrite)
    sr = context.audio_sr
    rms_ref = context.rms_ref

    # Stimulus i
    freqs = [s["lowBand"], s["highBand"]]
    SLs = [s["SL"], s["SL"]]
    durations = [s["tabDur"], s["tabDur"]]
    pads = [s["tabPad"], s["tabPad"]]
    combine(freqs, SLs, durations, pads, os.path.join(path, f'sc_1i_seed{seed}'))

    # ii
    SLs = [s["SL"], s["ii"]["midBandLevel"], s["SL"]]
    durations = [s["tabDur"], s["tabDur"], s["tabDur"]]
    pads = [s["tabPad"], s["tabPad"], s["tabPad"]]
    combine(s["stdFreqs"], SLs, durations, pads, os.path.join(path, f'sc_1ii_seed{seed}'))

    # iii
    SLs = [s["SL"], s["SL"], s["SL"]]
    durations = [s["tabDur"], s["maskerDur"], s["tabDur"]]
    pads = [s["tabPad"], s["maskerPad"], s["tabPad"]]
    combine(s["stdFreqs"], SLs, durations, pads, os.path.join(path, f'sc_1iii_seed{seed}'))
    
    # iv
    freqs = [s["lowBand"], s["iv"]["bottomMidBand"], s["iv"]["topMidBand"], s["highBand"]]
    SLs = [s["SL"], s["SL"] + s["iv"]["SLshift"], s["SL"] + s["iv"]["SLshift"], s["SL"]]
    durations = [s["tabDur"], s["maskerDur"], s["maskerDur"], s["tabDur"]]
    pads = [s["tabPad"], s["maskerPad"], s["maskerPad"], s["tabPad"]]
    combine(freqs, SLs, durations, pads, os.path.join(path, f'sc_1iv_seed{seed}'))

    # v
    freqs = [s["lowBand"], s["midBand"], s["midBand"], s["highBand"]]
    SLs = [s["SL"], s["SL"], s["SL"], s["SL"]]
    durations = [s["tabDur"], s["v"]["duration"], s["v"]["duration"], s["tabDur"]]
    pads = [s["tabPad"], [s["maskerPad"], s["fullDur"] - s["maskerPad"] - s["v"]["duration"]], [s["fullDur"] - s["maskerPad"] - s["v"]["duration"], s["maskerPad"]], s["tabPad"]]
    combine(freqs, SLs, durations, pads, os.path.join(path, f'sc_1v_seed{seed}'))

    s["audio_sr"] = sr
    s["rms_ref"] = rms_ref
    np.save(os.path.join(path, f"sc_fig1_settings_seed{seed}.npy"), s)

    return s


def fig2_settings(overwrite={}):
    fns = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
    allSLs = [[5, 35, 5], [10, 30, 10], [15, 25, 15], [20, 20, 20], [25, 15, 25], [30, 10, 30]]
    settings = {
        "lowBand": [100, 500],
        "midBand": [500, 2500],
        "highBand": [2500, 7500],
        "maskerDur": 0.750,
        "maskerPad": 0.1,
        "tabDur": 0.150,
        "tabPad": 0.4,
        "SLs": {fns[i]: allSLs[i] for i in range(len(fns))},
        "comparisonSLs": np.arange(-5, 25, 2.5)
    }
    for k in settings.keys():
        if k in overwrite.keys():
            settings[k] = overwrite[k]
    settings["freqs"] = [settings["lowBand"], settings["midBand"], settings["highBand"]]
    settings["fullDur"] = settings["maskerDur"] + 2*settings["maskerPad"]
    return settings


def fig2(path, overwrite={}):
    print('Figure 2')
    seed = overwrite.get("seed", 0)
    manual_seed(seed)
    settings = fig2_settings(overwrite)
    sr = context.audio_sr
    rms_ref = context.rms_ref

    # Experiment stimuli
    durations = [settings["tabDur"], settings["maskerDur"], settings["tabDur"]]
    pads = [settings["tabPad"], settings["maskerPad"], settings["tabPad"]]
    for k, v in settings["SLs"].items():
        combine(settings["freqs"], v, durations, pads, fn=f'{path}sc_2{k}_seed{seed}')
    settings["audio_sr"] = sr
    settings["rms_ref"] = rms_ref
    np.save(os.path.join(path, f"sc_fig2_settings_seed{seed}.npy"), settings)
    return settings


if __name__ == "__main__":

    print('McDermott & Oxenham (2008)')
    print('Spectral completion of partially masked sounds.')
    basepath = sys.argv[1]
    os.makedirs(basepath)

    fig1(basepath)
    fig2(basepath)

    print('~~~~~~~~~~~`')
