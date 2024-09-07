import os
import yaml
import numpy as np
import scipy.signal
from glob import glob

import torch

import renderer.cochleagrams as cgram
from renderer.util import freq_to_ERB

################
# Common interface to cochlegrams for dataloaders.py and dataset.py
################

#########
# Shared
#########


def create_gammatonegram(waves, gtg_module=None, self=None):
    """ Create Dan-Ellis gammatonegram
        (see renderer.cochleagrams for reference)
        Save gammatonegram creation module to save time

        Input
        -----
        wave: Tensor
            Shape [batch, sound_len]
        gtg_module: dict

        Returns
        -------
        output_dict: dict[str, Tensor]
    """
    if gtg_module is None:
        audio_sr = self.train_sr
        if self is not None and hasattr(self, "nfft"):
            # Creating gtg_module from params
            gtg_module = {
                "nfft": self.nfft,
                "nhop": self.nhop,
                "nwin": self.nwin,
                "gtm": self.gtm,
                "window": self.window,
                "rms_ref": self.gen_config["renderer"]["tf"]["rms_ref"]
                }
        else:
            # Neural network training, creating gtg_module from gen_config
            gtg_params = self.gen_config["renderer"]["tf"]["gtg_params"]
            gtg_module = {}
            gtg_module["nfft"], gtg_module["nhop"], gtg_module["nwin"], gtm = cgram.gtg_settings(
                    sr=audio_sr,
                    twin=gtg_params["twin"],
                    thop=gtg_params["thop"],
                    N=gtg_params["nfilts"],
                    fmin=gtg_params["fmin"],
                    fmax=audio_sr/2.0,
                    width=gtg_params["width"]
                )
            gtg_module["gtm"] = gtm[np.newaxis, :, :]
            gtg_module["window"] = torch.hann_window(gtg_module["nwin"])
            gtg_module["rms_ref"] = self.gen_config["renderer"]["tf"]["rms_ref"]
    grams = cgram.gammatonegram(
            waves,
            gtg_module["nfft"],
            gtg_module["nhop"],
            gtg_module["nwin"],
            gtg_module["gtm"],
            gtg_module["rms_ref"],
            gtg_module["window"]
        )
    output_dict = {"grams": grams}
    return output_dict


def create_all_cgm_representations(self, waves, audio_sr, cgm=None):
    """ Create cochleagram representations,
        including rectified subbands and envelopes

        Input
        -----
        wave: Tensor
            Shape [batch, sound_len]
        audio_sr: int
        cgm: dict

        Returns
        -------
        output_dict: dict[str, Tensor]
    """
    # Create cochleagram settings
    if cgm is None:
        cgm, _ = cgram.cgm_settings(
            waves.shape[1],
            audio_sr,
            self.gen_config["renderer"]["tf"]["cgm_params"]
            )
    # Create cochleagram with no thresholding
    subbands, recsubbands = cgram.subbands(
        waves,
        cgm["cochleagram"],
        cgm["rectify_subbands"],
        cgm["compression"],
        None,
        downsampling=None
        )
    envelopes, extracted_envs = cgram.envelopes(
        subbands,
        cgm["envelope_extraction_hilbert"],
        cgm["compression"],
        None,
        downsampling=None
        )
    grams = cgram.envelopes_to_cgm(
        extracted_envs,
        cgm["cochleagram"].downsampling,
        cgm["compression"],
        None
        )
    output_dict = {
        "subbands": recsubbands,
        "envelopes": envelopes,
        "grams": grams
        }
    return output_dict


def create_cochleagram_only(self, waves, audio_sr, cgm=None, already_recursed=False):
    """ Create cochleagram representation only

        Input
        -----
        wave: Tensor
            Shape [batch, sound_len]
        audio_sr: int
        cgm: dict

        Returns
        -------
        output_dict: dict[str, Tensor]
    """
    # Create cochleagram settings
    if cgm is None:
        cgm, _ = cgram.cgm_settings(
            waves.shape[1], audio_sr,
            self.gen_config["renderer"]["tf"]["cgm_params"]
            )
    try:
        subbands = cgram.subbands(
            waves, cgm["cochleagram"], cgm["rectify_subbands"],
            cgm["compression"], None, downsampling=None, subbands_only=True
            )
        extracted_envs = cgram.envelopes(
            subbands, cgm["envelope_extraction_hilbert"],
            cgm["compression"], None, downsampling=None,
            extracted_envs_only=True
            )
        grams = cgram.envelopes_to_cgm(
            extracted_envs, cgm["cochleagram"].downsampling,
            cgm["compression"], None
            )
    except RuntimeError as e:
        # May run into memory error. Try to loop instead
        # Use already_recursed to prevent infinite recursion
        if not already_recursed:
            print("Runtime Error: trying loop over \
                  waves to compute cochleagrams.")
            gram_list = [
                create_cochleagram_only(
                    self, waves[i, None, :], audio_sr, cgm=cgm,
                    already_recursed=True
                    )['grams'] for i in range(waves.shape[0])
                    ]
            grams = torch.cat(gram_list, dim=0)
        else:
            raise e
    output_dict = {"grams": grams}
    return output_dict


def pad_sound_for_cochleagram(self, waves, audio_sr):
    """ Pad sound to create correct shape for cochleagram downsampling

    Input
    -----
    wave: array
        Shape [batch, sound_len]
    audio_sr: int

    Returns
    -------
    padded_waves: array
    """
    seconds_per_frame = 1./self.gen_config["renderer"]["tf"]["cgm_params"]["sk_downsampling_kwargs"]["env_sr"]
    if waves.shape[1] % int(np.round(seconds_per_frame*audio_sr)) != 0:
        N = int(np.round(seconds_per_frame*audio_sr))
        diff = int(np.ceil(waves.shape[1] / N) * N) - waves.shape[1]
        padded_waves = np.concatenate(
            (waves, np.zeros((waves.shape[0], diff,))),
            axis=1)
        return padded_waves
    else:
        return waves

# Helper functions to apply the correct shapes to the various input types
def process_as_scene_dict(scene_dict):
    return {k: process_as_scene(v) for k, v in scene_dict.items()}


def process_as_scene(scene_gram):
    # b,f,t ==> t,f
    return scene_gram[0, :, :].transpose(0, 1).to(torch.float32)


def process_as_events_dict(events_dict):
    return {k: process_as_events(v) for k, v in events_dict.items()}


def process_as_events(events_gram):
    # sum(n_elements in each source), f, t ==> n_elements, t, f, n_elements
    return events_gram.transpose(1, 2).to(torch.float32)


##################
# Dataset creation
##################


class CgramError(Exception):
    pass


def make_tf_for_dreaming(self, audio_sr, config, gtg_module=None, cgm_module=None, use_cuda=None):
    """ Generate cochleagrams or gammatonegrams using settings
        in given modules, if they exist. Reusing modules saves
        significant rendering time.
    """
    if self.scene_wave.shape[0] > 1:
        raise Exception("Batch size should be equal to 1 for dreaming.")

    # Create scene gram
    # Create modules for computing representation if haven't already
    input_tensors = {}
    f_rep = {}
    if "gtg" in self.representations:
        if gtg_module is None:
            gtg_params = config["renderer"]["tf"]["gtg_params"]
            gtg_module = {}
            gtg_module["nfft"], gtg_module["nhop"], gtg_module["nwin"], gtm, f_rep["gtg"] = cgram.gtg_settings(
                sr=audio_sr,
                twin=gtg_params["twin"],
                thop=gtg_params["thop"],
                N=gtg_params["nfilts"],
                fmin=gtg_params["fmin"],
                fmax=audio_sr/2.0,
                width=gtg_params["width"],
                return_all=True
                )
            gtg_module["gtm"] = torch.Tensor(gtm[np.newaxis, :, :])
            gtg_module["window"] = torch.hann_window(gtg_module["nwin"])
            gtg_module["rms_ref"] = config["renderer"]["tf"]["rms_ref"]
        tf_dict = create_gammatonegram(self.scene_wave, gtg_module=gtg_module)
        input_tensors["gtg"] = process_as_scene(tf_dict["grams"])

    if "cgm" in self.representations:
        if cgm_module is None:
            cgm_module, _ = cgram.cgm_settings(
                self.scene_wave.shape[1],
                audio_sr,
                config["renderer"]["tf"]["cgm_params"]
                )
            if use_cuda:
                for k, v in cgm_module.items():
                    v.cuda()
        f_rep["cgm"] = torch.from_numpy(
            cgm_module["cochleagram"].numpy_coch_filter_extras["cf"][1:-1].astype(np.float32)
            )
        tf_dict = create_cochleagram_only(
            self, self.scene_wave, audio_sr, cgm=cgm_module
            )
        input_tensors["cgm"] = process_as_scene(tf_dict["grams"])

    # Check if there's anything wrong, and skip if so
    for rep in self.representations:
        nan_present = torch.any(torch.isnan(input_tensors[rep]))
        infinite_present = torch.any(~torch.isfinite(input_tensors[rep]))
        if nan_present or infinite_present:
            raise CgramError()

    # Create event cochleagrams
    event_grams = {}
    event_waves = torch.cat([
        self.off_ramp[None, :] * self.sources[i].event_waves[0, :, :] for i in range(self.n_sources)
        ], dim=0)
    if "gtg" in self.representations:
        tf_dict = create_gammatonegram(
            event_waves, gtg_module=gtg_module
            )
        event_grams["gtg"] = process_as_events(tf_dict["grams"])
    if "cgm" in self.representations:
        tf_dict = create_cochleagram_only(
            self, event_waves, audio_sr, cgm=cgm_module
            )
        event_grams["cgm"] = process_as_events(tf_dict["grams"])

    # Scene wave
    self.scene_wave = self.scene_wave[0, :].to(torch.float32)

    return input_tensors, event_grams, f_rep, gtg_module, cgm_module


def compute_ideal_mask(event_grams, threshold=None):
    """ Computes ideal binary masks of events
        from their time-frequency representation
    """

    # Transpose, then transpose it back at the end.
    event_grams = event_grams.permute(1, 2, 0).cpu().numpy()

    if threshold is not None:
        element_thresh = event_grams == threshold
    else:
        element_thresh = np.zeros(event_grams.shape, dtype=bool)

    silence = np.all(element_thresh, axis=2)  # shape: t, f
    sounding = np.invert(silence)
    IBM_idxs = np.argmax(event_grams, axis=2)
    t = range(IBM_idxs.shape[0])
    f = range(IBM_idxs.shape[1])
    t = np.repeat(t, IBM_idxs.shape[1])
    f = np.tile(f, IBM_idxs.shape[0])
    e_idxs = IBM_idxs[t, f][sounding[t, f]]
    t_idxs = t[sounding[t, f]]
    f_idxs = f[sounding[t, f]]
    IBMs = np.zeros(event_grams.shape, dtype=bool)
    IBMs[t_idxs, f_idxs, e_idxs] = True

    IBMs = torch.from_numpy(IBMs).permute(2, 0, 1)

    return {"IBM": IBMs}


#############
# Dataloaders
#############


def set_gen_config(self, dataset_name, network_tf_rep=None, dataset_config=None):
    """ For dataloader self, set attributes based on
        how the cochleagram inputs were created.
    """

    # Get data_creation_config of how data was created
    # Choose the first folder of the dataset to get the config
    # (All should be the same)
    # See also inference/amortized/config/dataset.yaml
    if dataset_config is None:
        datasets_folder = os.environ["dream_dir"]
        fs = glob(os.path.join(
            datasets_folder, f"*{dataset_name}*", ""
        ))
        dataset_config = os.path.join(fs[0], "data-creation_config.yaml")
        with open(dataset_config, 'r') as f:
            self.gen_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        self.gen_config = dataset_config

    self.train_sr = self.gen_config["dream"]["audio_sr"]
    if network_tf_rep not in self.gen_config["dream"]["tf_representation"]:
        raise Exception(f"{network_tf_rep} was not created \
                        in dataset {dataset_name}")

    if "event_features" not in self.gen_config["dream"].keys():
        self.gen_config["dream"]["event_features"] = {
            "timing": self.gen_config["hyperpriors"]["source_type"]["args"],
            "f0": ["whistle", "harmonic"],
            "amplitude": ["whistle"],
            "spa": ["harmonic", "noise"]
        }

    # This is the home of these shared parameters!
    self.abs_threshold = 20.
    self.rel_threshold = 60.
    self.silence_threshold = 15

    # If generating gammatonegrams
    self.rms_ref = self.gen_config["renderer"]["tf"]["rms_ref"]
    if network_tf_rep == "gtg":
        gtg_params = self.gen_config["renderer"]["tf"]["gtg_params"]
        self.nfft, self.nhop, self.nwin, gtm = cgram.gtg_settings(
            sr=self.train_sr,
            twin=gtg_params["twin"],
            thop=gtg_params["thop"],
            N=gtg_params["nfilts"],
            fmin=gtg_params["fmin"],
            fmax=self.train_sr/2.0,
            width=gtg_params["width"]
            )
        self.gtm = torch.Tensor(gtm[np.newaxis, :, :])
        self.window = torch.hann_window(self.nwin)
        # f_coch currently in scale, not Hz
        self.f_coch = freq_to_ERB(torch.from_numpy(cgram.gtg_settings(
            sr=self.gen_config["dream"]["audio_sr"],
            twin=gtg_params["twin"],
            thop=gtg_params["thop"],
            N=gtg_params["nfilts"],
            fmin=gtg_params["fmin"],
            fmax=self.gen_config["dream"]["audio_sr"]/2.0,
            width=gtg_params["width"], return_freqs=True).copy()
            ))
    elif network_tf_rep == "cgm":
        audio_sr = self.gen_config["dream"]["audio_sr"]
        n_samples = audio_sr
        cgm, _ = cgram.cgm_settings(
            n_samples,
            audio_sr,
            self.gen_config["renderer"]["tf"]["cgm_params"]
            )
        # f_coch currently in scale, not Hz
        self.f_coch = freq_to_ERB(torch.from_numpy(
            cgm["cochleagram"].numpy_coch_filter_extras["cf"][1:-1].astype(np.float32)
            ))


def input_threshold(self, scene, scene_max=None, use_rel_threshold=True):
    """ Threshold on cochleagram generated with
        chcochleagram - relative threshold reduces skirts
    """
    scene_max = scene.max() if scene_max is None else scene_max
    if use_rel_threshold:
        cutoff = max(scene_max - self.rel_threshold, self.abs_threshold)
    else:
        cutoff = self.abs_threshold
    # Send to 'silence'
    scene[scene < cutoff] = self.abs_threshold
    return scene


def apply_silence_mask(self, ims, max_of_event_grams, set_bool=None):
    """ Mask out parts of ims that are silent for all events """
    silenced_ims = ims * 1.0*(max_of_event_grams > self.abs_threshold)
    if set_bool is True:
        return silenced_ims.to(bool)
    elif set_bool is False:
        return silenced_ims
    else:
        raise Exception("Need set_bool argument")


def resample_sound(self, waves, audio_sr):
    """ Resample sound to the sampling rate of the neural network training data

        Input
        -----
        wave: array
            Shape [batch, sound_len]
        audio_sr: int

        Returns
        -------
        waves: array
        audio_sr: int
        resampled: bool
    """
    if audio_sr != self.train_sr:
        waves = scipy.signal.resample_poly(
            waves, self.train_sr, audio_sr, axis=1
            )
        return waves, self.train_sr, True
    else:
        return waves, audio_sr, False


def trim_sound(self, waves, audio_sr, trim_start=0.0):
    """ Trim sound to the  max length
        of the training data for this neural network

        Input
        -----
        wave: array
            Shape [batch, sound_len]
        audio_sr: int

        Returns
        -------
        waves: array
        trimmed: bool
    """
    scene_duration = waves.shape[1]/audio_sr
    if self.trim_to_max_duration and scene_duration > self.max_train_duration:
        start_idx = int(trim_start*audio_sr)
        waves = waves[:, start_idx:start_idx+int(audio_sr*self.max_train_duration)]
        return waves, True
    else:
        return waves, False


def make_tf_for_segmentation(self, waves, audio_sr, trim_start=0.0):
    """ Process sounds for dataloader """
    # Trim sounds to max train duration
    waves, _ = trim_sound(self, waves, audio_sr, trim_start=trim_start)
    waves, audio_sr, _ = resample_sound(self, waves, audio_sr)
    if self.tf_representation == "gtg":
        tf_dict = create_gammatonegram(torch.Tensor(waves), self=self)
    elif self.tf_representation == "cgm":
        waves = pad_sound_for_cochleagram(self, waves, audio_sr)
        tf_dict = create_all_cgm_representations(
            self, torch.Tensor(waves), audio_sr
            )
    return process_as_scene_dict(tf_dict), waves, audio_sr


def event_threshold(event_grams):
    """ Threshold events """
    event_grams = (event_grams - 20)/50
    event_grams[event_grams < 0] = 0
    return event_grams
