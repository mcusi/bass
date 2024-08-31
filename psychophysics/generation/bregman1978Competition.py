import sys
import os
import numpy as np
import soundfile as sf

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic
from util import context

"""
Bregman, A. S. (1978). Auditory streaming: Competition among alternative organizations.
Perception & Psychophysics, 23(5), 391-398.
"""


def expt2(path, overwrite={}, save_wav=True):

    print('Experiment 2')
    toneDuration = 0.100
    toneSilenceDuration = 0.010
    toneRampDuration = 0.010
    levels = {
        k: 70 for k in ['2800', '2642', '1556', '1468', '600', '566', '333', '314']
    }
    condFreqs = [
        [2800, 1556, 600, 333],
        [600, 333, 2800, 1556],
        [2800, 2642, 1556, 1468],
        [333, 314, 600, 566],
        [2800, 1556, 2642, 1468],
        [600, 333, 566, 314],
        [2800, 600, 1468, 314]
    ]
    condNames = ['isolate'] * 4 + ['absorb'] * 3
    nRepetitions = overwrite.get("n_repetitions", 4)

    def makeTone(freq, level):
        A = tones.pure(toneDuration, freq, level=level)
        A = basic.sRamps(A, toneRampDuration)
        A = basic.addSilence(A, [0, toneSilenceDuration])
        return A

    def makeCycle(freqs):
        a = makeTone(freqs[0], levels[str(freqs[0])])
        b = makeTone(freqs[1], levels[str(freqs[1])])
        x = makeTone(freqs[2], levels[str(freqs[2])])
        y = makeTone(freqs[3], levels[str(freqs[3])])
        return np.tile(np.concatenate((a, b, x, y)), nRepetitions)

    # AB Isolation
    sound_dict = {}
    for i in range(len(condNames)):
        comparison = makeCycle(condFreqs[i])
        # Pad the entire sequence with a bit of silence
        # The last tone will already have it though
        comparison = basic.addSilence(comparison, [0.05, 0.05])
        if save_wav:
            fn = f'{path}{condNames[i]}_A{condFreqs[i][0]}_B{condFreqs[i][1]}_D1-{condFreqs[i][2]}_D2-{condFreqs[i][3]}.wav'
            sf.write(fn, comparison, context.audio_sr)
        sound_dict[f'{condNames[i]}_A{condFreqs[i][0]}_B{condFreqs[i][1]}_D1-{condFreqs[i][2]}_D2-{condFreqs[i][3]}'] = comparison

    if not save_wav:
        return sound_dict


if __name__ == "__main__":
    audio_fp = str(sys.argv[1])
    os.makedirs(audio_fp, existok=False)
    print('Bregman, A. S. (1978)')
    print('Auditory streaming: Competition among alternative organizations.')
    with context(audio_sr=20000, rms_ref=1e-6):
        expt2(audio_fp)
    print("~~~~")
