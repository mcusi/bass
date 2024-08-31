import os
import sys
import numpy as np
import soundfile as sf

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic  
from util import context

"""
Tougas, Y. & Bregman, A. S. (1985). Crossing of Auditory Streams.
Journal of Experimental Psychology: Human Perception and Performance, 11(6), 788-798.
"""


def demo(path):

    print('Experiment 2: harmonics vs. pure tones')
    sr = context.audio_sr
    duration = 0.100
    rampDuration = 0.008
    frequencies = [504, 1600, 635, 1270, 800, 1008, 1008, 800, 1270, 635, 1600, 504]
    pureLevels = {str(int(k)): 73 for k in np.unique(frequencies)}
    richLevels = {str(int(k)): 67 for k in np.unique(frequencies)}

    def makeTone(freq, kind):
        if kind == 'pure':
            a = tones.pure(duration, freq, level=pureLevels[str(freq)])
        elif kind == 'rich':
            a = tones.harmonic(
                duration,
                freq,
                [1, 1, 1, 1],
                levels=[richLevels[str(freq)] for i in range(4)]
            )
        a = basic.linearRamps(a, rampDuration)
        return a
    s = basic.silence(0.100)

    # Full sequence, pure tone only
    kind = "pure"
    sequence = []
    for i in range(len(frequencies)):
        freq = frequencies[i]
        sequence = np.concatenate((sequence, makeTone(freq, kind)))
    sf.write(os.path.join(path, '1.wav'), sequence, sr)

    # Low bounce, pure tone only
    sequence = []
    for freq in frequencies:
        if freq <= 800:
            sequence = np.concatenate((sequence, makeTone(freq, kind), s))
    sf.write(os.path.join(path, '1_lowbounce.wav'), sequence, sr)

    # High bounce, pure tone only
    sequence = []
    for freq in frequencies:
        if freq >= 800:
            sequence = np.concatenate((sequence, makeTone(freq, kind), s))
    sf.write(os.path.join(path, '1_highbounce.wav'), sequence, sr)

    # Full sequence, pure and complex alternating
    sequence = []
    for i in range(len(frequencies)):
        freq = frequencies[i]
        kind = 'rich' if i % 2 == 0 else 'pure'
        sequence = np.concatenate((sequence, makeTone(freq, kind)))
    sf.write(os.path.join(path, '4.wav'), sequence, sr)

    # Low bounce, pure and complex alternating
    sequence = []
    for i in range(len(frequencies)):
        freq = frequencies[i]
        if freq <= 800:
            kind = 'rich' if i % 2 == 0 else 'pure'
            sequence = np.concatenate((sequence, makeTone(freq, kind), s))
    sf.write(os.path.join(path, '4_lowbounce.wav'), sequence, sr)

    # High bounce
    sequence = []
    for i in range(len(frequencies)):
        freq = frequencies[i]
        if freq >= 800:
            kind = 'rich' if i % 2 == 0 else 'pure'
            sequence = np.concatenate((sequence, makeTone(freq, kind), s))
    sf.write(os.path.join(path, '4_highbounce.wav'), sequence, sr)


if __name__ == "__main__":
    print('Tougas, Y. & Bregman, A. S. (1985)')
    print('Crossing of Auditory Streams.')
    basepath = sys.argv[1]
    os.makedirs(basepath)
    demo(basepath)
    print('~~~~~~~~~~~~~~~~~')
