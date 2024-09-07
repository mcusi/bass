import os
import re
import numpy as np
import soundfile as sf

from renderer.util import ramp_sound


def get_paths(sound_group, config):
    """ Define functions to load sounds from FUSS
    FUSS naming conventions: example{:05d,0-19999}.wav
    """

    FUSS_folder = os.path.join("FUSS", "ssdata", "train", "")
    n_datapoints = 20000
    # Where is data stored
    naming = {}
    naming["metadata"] = lambda i: os.path.join(
        os.environ["sound_dir"], FUSS_folder, f"example{i:05d}.jams"
        )
    naming["scene_src"] = lambda i: os.path.join(
        os.environ["sound_dir"], FUSS_folder, f"example{i:05d}.wav"
        )

    def source_src(i, source_idx, m=None):
        if source_idx == 0:
            name = "background0"
        elif source_idx > 0:
            name = "foreground{}".format(source_idx - 1)
        return os.path.join(
            os.environ["sound_dir"],
            FUSS_folder,
            f"example{i:05d}_sources/{name}_sound.wav"
            )
    naming["source_src"] = source_src

    # Where to save data
    naming["scene_dest"] = lambda i: os.path.join(
        os.environ["sound_dir"], sound_group, f"FUSS-train_{i:05d}_scene.wav"
        )
    naming["source_dest"] = lambda i, source_idx: os.path.join(
        os.environ["sound_dir"], sound_group, "single_sources",
        f"FUSS-train_{i:05d}_source{source_idx:02d}.wav"
        )
    return "fuss", n_datapoints, naming


def get_first_foreground_onset(m):
    """ Return the onset of the first foreground premixture sound
        (in seconds from beginning of mixture clip)
    """
    try:
        return m["annotations"][0].data[1].time
    except IndexError:
        return False


def check_for_n_sounds_inside(m, t0, t1, n, d):
    """ Checks for sounds of duration=min(their full duration, d)
        sec within [t0, t1]

        Means that:
        Exactly n sounds have are ongoing for at least d seconds
        in the interval
        Exactly n-1 foreground sounds are ongoing [... d seconds ... ]

        Input
        -----
        m: jams annotation for sound
        t0, t1: time of beginning, end of interval in seconds
        n: number of sounds
        d: minimum duration in seconds

        Output
        ------
        bool: whether there are exactly n sounds as specified above
        within_interval: which premixture sounds are inside the interval
        event_timings: the onset and offset of the premixture sounds
            relative to the interval start and finish (t0, t1)
    """
    within_interval = []
    event_timings = []
    for event in m["annotations"][0].data:
        onset = event.time
        # event.duration gives the time after any time_stretching.
        offset = event.time + event.duration
        a = max([t0, onset])
        b = min([t1, offset])
        # If they don't overlap at all, a > b --> False
        within_interval.append(b - a >= min(d, event.duration))
        event_timings.append([max(0, onset-t0), min(t1, offset)-t0])
    return (sum(within_interval) == n), within_interval, event_timings


def clip_ramp_normalize(fn, t0, t1):
    """ Clip pre-mixture sounds to start at t0
        and end at t1 (in seconds). Apply ramp
        and return max amplitude.
    """
    x, sr = sf.read(fn)
    if len(x.shape) == 2:
        print("Found multichannel audio")
        x = x[:, 0]
    # FUSS premixture recordings are the same duration as the mixture
    clipped_x = x[int(sr*t0):int(sr*t1)]
    clipped_x = ramp_sound(clipped_x, sr, ramp_duration=0.005)
    # normalize sound
    clipped_x = clipped_x - clipped_x.mean()
    max_x = np.max(np.abs(clipped_x))
    return clipped_x, sr, max_x


def check_for_categories(m, source_idxs, categories_to_exclude):
    """ Get the categories of the sounds
        in a scene and check for any to exclude.
    """
    # FUSS does not contain the class labels in the
    # jams annotations so we need to backtrack them
    ground_truth = os.path.join(
        os.environ["sound_dir"], "FUSS",
        "soundbank", "FSD50K.ground_truth", "dev.csv"
        )
    source_files = {}
    for source_idx in source_idxs:
        event = m["annotations"][0].data[source_idx]
        # Gets raw number
        source_files[source_idx] = os.path.splitext(
            os.path.basename(event.value["source_file"])
            )[0]

    # Retrieve categories from FSD50K if possible
    categories = {}
    with open(ground_truth, "r") as f:
        for line in f:
            for source_idx, source_file in source_files.items():
                if re.search(source_file, line):
                    categories[source_idx] = line

    # Check if all of the source_idxs are included in some cateogry
    for source_idx in source_idxs:
        if source_idx not in categories.keys():
            return False, categories

    # check if excluded_category is a substring of category
    for source_idx, category in categories.items():
        for excluded_category in categories_to_exclude:
            if excluded_category in category:
                return False, categories

    # If successful return the categories
    return True, categories
