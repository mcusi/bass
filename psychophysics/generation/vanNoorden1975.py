import sys
import os
import numpy as np
import soundfile as sf

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic
from util import context

"""
Van Noorden, L. S. (1975). Temporal coherence in the perception of tone sequences. 
PhD thesis, Eindhoven University of Technology.
"""


def expt1(path, overwrite={}, save_wav=True):

    print('Experiment 1')
    repetitionTimes = [67, 83, 100, 117, 134, 150] if "dt" not in overwrite.keys() else overwrite["dt"]  # ms
    semitoneSteps = [1, 3, 6, 9, 12] if "df" not in overwrite.keys() else overwrite["df"]
    cf = overwrite.get("cf", 1000)
    toneDuration = 0.050  # s
    rampDuration = 0.01  # s
    level = overwrite.get("level", 70)
    levelA = overwrite.get("levelA", level)
    levelB = overwrite.get("levelB", level)
    nReps = overwrite.get("n_reps", [1, 2, 3, 4, 5])  # monotonically increasing
    all_same_len = True
    sr = context.audio_sr

    _A = tones.pure(toneDuration, cf, level=levelA)
    _A = basic.raisedCosineRamps(_A, rampDuration)
    expt_dict = {}
    for rep_idx, nRep in enumerate(nReps[::-1]):
        for ss in semitoneSteps:
            f = cf*(np.power(2., (ss/12.)))
            _B = tones.pure(toneDuration, f, level=levelB)
            _B = basic.raisedCosineRamps(_B, rampDuration)
            for time_idx, repTime in enumerate(repetitionTimes[::-1]):
                silenceDuration = repTime/1000. - toneDuration
                A = basic.addSilence(_A, [0, silenceDuration])
                B = basic.addSilence(_B, [0, silenceDuration])
                C = basic.silence(repTime/1000.)
                stimulus = np.concatenate((A, B, A, C))
                sequence = np.tile(stimulus, nRep)
                fn = os.path.join(path, f'df{ss}_dt{repTime}_rep{nRep}.wav')
                sequence = basic.addSilence(sequence, [silenceDuration, 0])
                if len(nReps) > 1:
                    len_match_flag = (time_idx == 0) if all_same_len else True
                    if rep_idx == 0 and len_match_flag:
                        total_len = len(sequence)
                    else:
                        remainder_len = total_len - len(sequence)
                        sequence = basic.addSilence(sequence, [0, remainder_len/sr])
                if save_wav:
                    sf.write(fn, sequence, sr)
                else:
                    expt_dict[fn] = sequence

    if not save_wav:
        return expt_dict, sr


if __name__ == "__main__":

    print('Van Noorden (1975)')
    print('Temporal coherence in the perception of tone sequences. PhD thesis')
    print('Experiments pertaining to figure 2.6, 2.7 and 2.9 (pages 12, 13, 15)')

    basepath = sys.argv[1]
    os.makedirs(basepath)
    expt1(basepath)

    print('~~~~~~~~~~~~~~~~~')
