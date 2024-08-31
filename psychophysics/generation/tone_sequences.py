import os

import psychophysics.generation.bregman1978Competition as compete
import psychophysics.generation.bregmanRudnicky1975 as captors
import psychophysics.generation.tougasBregman1985 as crossing
import psychophysics.generation.vanNoorden1975 as aba


def make_all(sound_group, overwrite={}):
    path = os.path.join(os.environ["sound_dir"], sound_group, "")
    if overwrite.get("compete", True) is not False:
        compete.expt2(path, overwrite.get("compete", {"n_repetitions": 4}))
    if overwrite.get("aba", True) is not False:
        aba.expt1(path, overwrite.get("aba", {}))
    if overwrite.get("cumulative", True) is not False:
        # cumul_settings creates Thompson et al 2012
        cumul_settings = {
            "cf": 500,
            "dt": [125],
            "df": [4, 8],
            "level": 55,
            "n_reps": [1, 2, 3, 4, 5, 6]
        }
        cumul_settings.update(overwrite.get("cumulative", {}))
        aba.expt1(path, overwrite=cumul_settings)
    if overwrite.get("captor", True) is not False:
        captors.expt1(path)
    if overwrite.get("bouncing", True) is not False:
        crossing.demo(path)