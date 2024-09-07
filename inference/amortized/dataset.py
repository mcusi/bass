import os
import yaml
import sys
import dill
import numpy as np
from glob import glob

import torch

from util.sample import manual_seed
from util.cuda import cuda_reset, cuda_robust
from util.context import context
from model.scene import Scene
import inference.amortized.cgram_util as cgram_util
from renderer.util import freq_to_ERB, freq_to_octave

#####################
# Sample from the generative model to create
# datasets for neural network training
#####################


def dream(config_name, parallel, seed=0):
    """ Sample several Scene objects to create dataset
        for amortized inference network

        Saves files "chunk*.tar" where:
            - chunk: list[scene_dict]

            - scene_dict: {"info": bin_info, "ims": ims_dict, "scene": Scene}

            - bin_info: {
                    "t_coch": torch.Tensor[n_cochleagram_timepoints]  timing of each timebin of cochleagram                (same across chunk)
                    "t_audio": torch.Tensor[n_audio_timepoints]       timing of each timebin of audio                      (same across chunk)
                    "f": torch.Tensor[n_cochleagram_frequencies]      frequency of each frequencybin of tf_representation  (same across dataset)
                    "scene_duration": float    length of scene in seconds
                }
            - ims_dict: {
                    "audio": {
                        "IBM": torch.Tensor(n_events, n_audio_timepoints, n_cochleagram_frequencies),
                        ...
                    },
                    "grams": {
                        # Everything here is a Tensor(n_events, n_cochleagram_timepoints, n_cochleagram_frequencies)
                        "events": event cochleagrams
                        "IBM":    ideal binary mask
                        ...
                    }
                }

    """

    # Load config and get location for saving dreamed samples
    savepath, config, n_chunks_so_far, use_cuda = load_config_for_dreaming(config_name, parallel)

    # Define parameters for tf representation
    tf_representations = config["dream"]["tf_representation"]
    rep_params = {}
    if "gtg" in tf_representations:
        rep_params["gtg"] = config["renderer"]["tf"]["gtg_params"]
    if "cgm" in tf_representations:
        rep_params["cgm"] = config["renderer"]["tf"]["cgm_params"]

    if use_cuda: cuda_reset()
    gtg_module = None

    for chunk_idx in range(n_chunks_so_far, config["dream"]["n_chunks"]):
        manual_seed(seed*10000000 + chunk_idx)

        # Sample a duration for this chunk
        scene_duration, seconds_per_frame = sample_scene_duration(
            tf_representations, rep_params, config, chunk_idx
            )
        # If using cochleagram representation,
        # need to redo cgm_module for each scene_duration.
        # If gtg, will just remain None
        # Reusing across sounds of the same duration
        # helps to save rendering time.
        cgm_module = None

        stored_vals = []
        for sample_idx in range(config["dream"]["chunk_size"]):
            # Generate trace
            with context(config, batch_size=1), torch.no_grad():
                try:
                    full_data, gtg_module, cgm_module = sample(
                        config,
                        scene_duration,
                        seconds_per_frame,
                        use_cuda,
                        gtg_module=gtg_module,
                        cgm_module=cgm_module
                        )
                    stored_vals.append(full_data)
                except cgram_util.CgramError:
                    gtg_module = None
                    cgm_module = None
                    print(f"Cgram error on chunk {chunk_idx}, sample {sample_idx}. Skipping.")
                except RuntimeError as e:
                    gtg_module = None
                    cgm_module = None
                    if "cuda" in str(e).lower() or "cufft" in str(e).lower():
                       print(e)
                       print(f"Double-cuda error on chunk {chunk_idx}, sample {sample_idx}. Skipping.")
                    else:
                        raise e

                if use_cuda: cuda_reset()

        # Save all the outputs
        f = os.path.join(
            savepath, f"chunk{chunk_idx:08d}_t{int(scene_duration*1000):05d}.tar"
            )
        print(f, flush=True)
        # torch.save won't save functions, use dill
        torch.save(stored_vals, f, pickle_module=dill)

    print("Complete!")
    return


def sample(config, scene_duration, seconds_per_frame, use_cuda, gtg_module=None, cgm_module=None):
    """Sample a single Scene object and generate associated dataset entry"""

    freq_to_scale = {'ERB': freq_to_ERB, "octave": freq_to_octave}[
        config["renderer"]["steps"]["scale"]
    ]

    # Sample a scene wave
    scene = cuda_robust(
        lambda: Scene.sample(
            config["hyperpriors"],
            audio_sr=config["dream"]["audio_sr"],
            scene_duration=scene_duration
            )
    )

    # Create the cochleagrams needed for training networks.
    def on_cuda_error(migrate):
        migrate(scene)
        scene.scene_wave = migrate(scene.scene_wave)
        for source in scene.sources:
            source.event_waves = migrate(source.event_waves)

    input_tensors, event_grams, f_rep, gtg_module, cgm_module = cuda_robust(
        lambda: cgram_util.make_tf_for_dreaming(
            scene,
            config["dream"]["audio_sr"],
            config,
            cgm_module=cgm_module,
            use_cuda=use_cuda
            ),
        on_cuda_error
    )

    # Create spectrum_plus_amplitude, aka "spa" representation
    for source in scene.sources:
        if source.source_type == "noise" or source.source_type == "harmonic":
            source.shifted_spa, source.shift_masks = estimation_grid(
                source.source_type,
                source.renderer.AM_and_filter,
                source.renderer.trimmer,
                source.sequence.events
                )

    # Axes of tf representation
    t_rep = {}
    for rep in scene.representations:
        t_rep[rep] = seconds_per_frame[rep]*torch.arange(
            input_tensors[rep].shape[0], dtype=torch.float32
            )
    # We compute binning of latents at training time,
    # so here we keep all necessary info to do this.
    bin_info = {
        "t_coch": t_rep,
        "f": {k: freq_to_scale(v) for k, v in f_rep.items()},
        "scene_duration": scene_duration
        }

    # Compute ideal masks and then make the dict to save
    ims_dict = {}
    for rep in scene.representations:
        ideal_masks_gram = cgram_util.compute_ideal_mask(event_grams[rep], threshold=None)
        ims_dict[rep] = {
            **ideal_masks_gram, "events": event_grams[rep]
            }

    scene.clear_renderer()
    full_data = {"info": bin_info, "scene": scene, "input": input_tensors, "ims": ims_dict}

    return full_data, gtg_module, cgm_module


def load_config_for_dreaming(config_name, parallel):
    """load config specifically for dreaming"""
    # Load config
    with open(os.environ["config_dir"] + config_name + '.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    savepath = os.path.join(
        os.environ["dream_dir"], f"{config_name}_{parallel}", ""
        )
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    cfile = os.path.join(savepath, 'data-creation_config.yaml')
    # Save config for completeness
    if not os.path.isfile(cfile):
        print("Saving config: ", cfile)
        with open(cfile, 'w') as f:
            yaml.dump(
                config, stream=f,
                default_flow_style=False,
                sort_keys=False
                )
    print(savepath, flush=True)
    # Check how many chunks we already made
    chunks = glob(os.path.join(savepath, "chunk*.tar"))
    n_chunks_so_far = len(chunks)
    print("Number of chunks so far: " + str(n_chunks_so_far), flush=True)
    # Check if there's a cuda device available
    print("Cuda visible devices: ", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    use_cuda = torch.cuda.is_available()
    return savepath, config, n_chunks_so_far, use_cuda


def sample_scene_duration(tf_representations, rep_params, config, chunk_idx):
    """Randomly sample a scene duration to be generated"""
    seconds_per_frame = {}
    if "cgm" in tf_representations:
        seconds_per_frame["cgm"] = 1./rep_params["cgm"]["sk_downsampling_kwargs"]["env_sr"]
        b = config["dream"]["dur"][1]/seconds_per_frame["cgm"]
        a = config["dream"]["dur"][0]/seconds_per_frame["cgm"]
        scene_duration = seconds_per_frame["cgm"] * np.random.randint(a, high=b+1)
        scene_duration = np.round(scene_duration, decimals=5)
    else:
        scene_duration = np.round(
            (config["dream"]["dur"][1]-config["dream"]["dur"][0])*np.random.random_sample() + config["dream"]["dur"][0],
            decimals=3
            )
    if "gtg" in tf_representations:
        seconds_per_frame["gtg"] = rep_params["gtg"]["thop"]
    print(f"Next chunk {chunk_idx} with scene_duration={scene_duration}", flush=True)
    return scene_duration, seconds_per_frame


def estimation_grid(source_type, am_and_filter, trimmer, events, VminSpa=-100.0):
    """ Creates spectrum_plus_amplitude representation
        to be estimated by amortized inference network
    """

    # Shift grid takes batch, channels, windows
    if source_type == "harmonic":
        grid_mask = (am_and_filter.shift_E(
            torch.ones(
                am_and_filter.extended_spectrum.shape,
                device=am_and_filter.extended_spectrum.device
            ),
            am_and_filter.f0_for_shift
            ) > 0
        )*1.
        L = am_and_filter.shift_E(
            am_and_filter.tf_grid.permute(0, 2, 1),
            am_and_filter.f0_for_shift,
            V=VminSpa
            )
    elif source_type == "noise":
        grid_mask = torch.ones(
            am_and_filter.tf_grid.permute(0, 2, 1).shape,
            device=am_and_filter.tf_grid.device
            )
        L = am_and_filter.tf_grid.permute(0, 2, 1)

    # Pad to latent takes batch windows channels
    if trimmer.trim_events and trimmer.trim_this_iter:
        full_timing_level = trimmer.pad_to_latent(L.permute(0, 2, 1), events, VminSpa)
        grid_mask = trimmer.pad_to_latent(grid_mask.permute(0, 2, 1), events, 0.0)
    else:
        onsets = torch.stack([e.onset.timepoint for e in events], -1)
        offsets = torch.stack([e.offset.timepoint for e in events], -1)
        event_active = ((trimmer.gp_t_rs >= onsets.T)*(trimmer.gp_t_rs <= offsets.T))
        full_timing_level = L.permute(0, 2, 1)[:, 1:-1, :] * event_active[:, :, None]
        full_timing_level[~event_active[:, :, None].expand(-1, -1, full_timing_level.shape[2])] = VminSpa
        grid_mask = grid_mask.permute(0, 2, 1)[:, 1:-1, :] * event_active[:, :, None]

    return full_timing_level, grid_mask


def delete_corrupted(data_location):
    folders = glob(data_location)
    for folder in folders:
        files = glob(os.path.join(folder, "*.tar"))
        for file in files:
            try:
                _ = torch.load(file, pickle_module=dill)
                print("Success: ", file)
            except:
                print("Failure: ", file)
                os.remove(file)


if __name__ == '__main__':
    if str(sys.argv[1]) == "clean":
        data_location = sys.argv[2]
        delete_corrupted(data_location)
    else:
        parameter_file = str(sys.argv[2])
        parallel = int(sys.argv[3])
        dream(parameter_file, parallel, seed=parallel)
