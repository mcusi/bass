import os
import sys
import numpy as np
import soundfile as sf

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic  
from util import context

"""
Bregman, A. S., & Rudnicky, A. I. (1975). Auditory segregation: stream or streams?
Journal of Experimental Psychology: Human Perception and Performance, 1(3), 263.
"""


def expt1(path, overwrite={}, save_wav=True):

    print('Experiment 1 (no standard)')
    freq_a = 2200
    level_a = overwrite.get("level_a", 60)
    freq_b = 2400
    level_b = overwrite.get("level_b", 60)
    freq_distractor = 1460
    level_distractor = overwrite.get("level_distractor", 60)
    freq_captors = [590, 1030, 1460]
    level_captors = overwrite.get("level_captors", [63, 60, 65])
    silenceDurationTarget = [0.009, 0]
    silenceDurationCaptors = [0.009, 0.064]
    toneDuration = overwrite.get("tone_duration", 0.045)
    rampDuration = [0.007, 0.005]

    A = tones.pure(toneDuration, freq_a, level=level_a)
    A = basic.linearRamps(A, rampDuration)
    A = basic.addSilence(A, silenceDurationTarget)
    B = tones.pure(toneDuration, freq_b, level=level_b)
    B = basic.linearRamps(B, rampDuration)
    B = basic.addSilence(B, silenceDurationTarget)
    D = tones.pure(toneDuration, freq_distractor, level=level_distractor)
    D = basic.linearRamps(D, rampDuration)
    D1 = basic.addSilence(D, silenceDurationTarget)
    D2 = basic.addSilence(D, silenceDurationCaptors)

    standard = {}
    standard['up'] = np.concatenate((A, B))
    standard['down'] = np.concatenate((B, A))

    comparison = {}
    comparison['up'] = np.concatenate((D1, A, B, D2))
    comparison['down'] = np.concatenate((D1, B, A, D2))

    captors = []
    for i in range(len(freq_captors)):
        c = tones.pure(toneDuration, freq_captors[i], level=level_captors[i])
        c = basic.linearRamps(c, rampDuration)
        c = basic.addSilence(c, silenceDurationCaptors)
        captors.append(c)

    freq_captors.append('none')
    sounds_dict = {}
    for i in range(len(captors) + 1):
        if i == len(captors):
            beginning = np.zeros(len(beginning))
            end = np.zeros(len(end))
        else:
            c = captors[i]
            beginning = np.tile(c, 3)
            end = np.tile(c, 2)
        for comparisonDirection in ['up', 'down']:
            sequence = np.concatenate((
                basic.silence(0.1),
                beginning,
                comparison[comparisonDirection],
                end
            ))
            fn = f'{path}comp{comparisonDirection}_f{freq_captors[i]}.wav'
            if save_wav:
                sf.write(fn, sequence, context.audio_sr)
            else:
                sounds_dict[fn] = sequence
    
    return sounds_dict, context.audio_sr

if __name__ == "__main__":
    audio_fp = sys.argv[1]
    print('Bregman & Rudnicky (1975)')
    print('Auditory segregation: stream or streams?')
    os.makedirs(audio_fp, existok=False)
    with context(audio_sr=20000, rms_ref=1e-6):
        overwrite = {"tone_duration": 0.070}
        expt1(audio_fp, overwrite=overwrite)
