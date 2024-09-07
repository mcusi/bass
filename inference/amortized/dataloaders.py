import os
import numpy as np
from glob import glob
import dill
import soundfile as sf
import scipy.interpolate

import torch
from torch.utils.data import DataLoader
from detectron2.engine import DefaultTrainer
from detectron2.structures import Instances, Boxes, BitMasks
import detectron2.utils.comm as comm

from inference.amortized.soft_masks import SoftMasks
import inference.amortized.cgram_util as cgram_util

################
# Dataloaders for training and inference
# with detectron2 segmentation network
################


class GeneratedScenesDataset(torch.utils.data.IterableDataset):
    """Generative model samples for segmentation."""

    def __init__(self, dataset_name, mask_types=None, batch_size=None, im_type=None, test=False, n_gpus=1, gpu_id=1, cfg=None):
        super().__init__()

        # Options
        self.im_type = "IBM" if im_type is None else im_type
        self.batch_size = batch_size or cfg.BASA.IMS_PER_BATCH
        self.use_real_timepoints = cfg.BASA.USE_REAL_TIMEPOINTS
        self.box_height_mask = cfg.BASA.BOX_HEIGHT_MASK
        self.tf_representation = cfg.BASA.TF_REPRESENTATION
        self.im_cutoff_ceil = cfg.BASA.IM_CUTOFF_CEIL
        self.use_rel_threshold = cfg.BASA.USE_REL_THRESHOLD
        self.mask_types = mask_types

        # Get paths of data files
        datasets_folder = os.environ["dream_dir"]
        fs = glob(os.path.join(
            datasets_folder, f"*{dataset_name}*", ""
            ))
        if not test:
            this_dataset_folders = [f for f in fs if "test" not in f]
        else:
            this_dataset_folders = [f for f in fs if "test" in f]

        chunks_in_folders = [
            glob(os.path.join(f, "chunk*.tar")) for f in this_dataset_folders
            ]
        self.chunk_paths = [item for sublist in chunks_in_folders for item in sublist]

        # Get data_creation_config of how data was created
        cgram_util.set_gen_config(
            self, dataset_name, network_tf_rep=self.tf_representation
            )

        self.n_gpus = n_gpus
        self.gpu_id = gpu_id

    def __iter__(self):
        """ Return an iterator of batches in this dataset,
            accounting for parallelization and batch size
        """
        worker_info = torch.utils.data.get_worker_info()
        workers_per_gpu = 1 if worker_info is None else worker_info.num_workers
        worker_id_on_gpu = 1 if worker_info is None else worker_info.id
        num_workers = workers_per_gpu * self.n_gpus
        worker_id = 0 if worker_info is None else worker_id_on_gpu + workers_per_gpu * self.gpu_id
        batch_size = self.batch_size if self.n_gpus == 0 else self.batch_size // self.n_gpus
        if batch_size > self.gen_config["dream"]["chunk_size"]:
            raise Exception(f"Chunks of size \
                                {self.gen_config['dream']['chunk_size']} \
                                do not contain one batch of size {batch_size}")
        elif batch_size <= 0:
            raise Exception("Batch size must be strictly postiive.")

        print(f"Starting worker {worker_id} of {num_workers}. Batch size per worker: {batch_size}")
        for i, chunk_path in enumerate(self.chunk_paths):
            if self.n_gpus > 0:
                if i % num_workers != worker_id:
                    continue

            try:
                chunk = torch.load(
                    chunk_path, pickle_module=dill, map_location="cpu"
                    )
            except Exception as e:
                print(f"Warning: could not load {chunk_path}")
                print(e)
                continue

            try:
                records = self.make_records(chunk, chunk_path)
            except Exception as e:
                print(f"{self.__class__} failed on chunk: {chunk_path}")
                raise e

            for i in range(0, len(records), batch_size):
                if len(records[i:i+batch_size]) == batch_size:
                    yield records[i:i + batch_size]

    def make_records(self, chunk, chunk_path):
        """ Create appropriately formatted detectron2
            inputs from chunks saved in dataset.py
        """

        records = []
        for scene_dict_idx, scene_dict in enumerate(chunk):

            # Obtaining necessary information from generative model
            # Obtain scene cochleagram and ideal masks
            if self.tf_representation == "cgm":
                scene_gram = cgram_util.input_threshold(
                    self, scene_dict["input"][self.tf_representation],
                    use_rel_threshold=self.use_rel_threshold
                    )
            elif self.tf_representation == 'gtg':
                scene_gram = scene_dict["input"][self.tf_representation]
            ideal_masks = cgram_util.apply_silence_mask(
                self,
                scene_dict["ims"][self.tf_representation][self.im_type],
                scene_dict["ims"][self.tf_representation]["events"].max(0).values,
                set_bool=True
                )
            to_keep = ideal_masks.sum((1, 2)) > self.silence_threshold
            ideal_masks = ideal_masks[to_keep]
            if len(ideal_masks) == 0:
                continue
            # Obtain event cochlegrams
            event_grams = cgram_util.event_threshold(
                scene_dict["ims"][self.tf_representation]["events"][to_keep]
            )

            # Obtain events
            events = [(source, event_idx, event) for source in scene_dict['scene'].sources for event_idx, event in enumerate(source.sequence.events)]
            events = [x for i, x in enumerate(events) if to_keep[i]]

            # Obtain event timings
            t_coch = scene_dict['info']['t_coch'][self.tf_representation]
            event_timings = torch.Tensor([
                (event.onset.timepoint, event.offset.timepoint)
                for (_, _, event) in events]
                )
            event_timing_idxs = (event_timings[..., None]-t_coch).abs().argmin(-1)

            # Obtain sound type labels
            source_type_labels = torch.LongTensor([int(source.source_type_idx.item()) for (source, _, _) in events])

            # Format the Gaussian Process latent variables
            # from the generative model into targets for detectron2
            latents = []
            # f_coch: currently in scale units
            f_coch = scene_dict['info']['f'][self.tf_representation]
            for (source, event_idx, _), (onset_idx, offset_idx), event_gram in zip(events, event_timing_idxs, event_grams):
                event_latents = self.get_targets_from_event(
                    source, event_idx, onset_idx, offset_idx,
                    t_coch, f_coch, event_gram
                    )
                latents.append(event_latents)
            latents = {k: torch.stack([x[k] for x in latents]) for k in latents[0].keys()}

            # Create data record to yield in detectron2 format
            record = {}

            # Add formatted sound input into record
            scene_gram = scene_gram.numpy()
            record["scene"] = scene_gram
            img, ft_shape = tf_to_image(self, scene_gram)
            record["image"] = torch.as_tensor(img.astype("float32"))

            # Create instances
            record["instances"] = Instances(ft_shape)
            record["instances"].gt_classes = source_type_labels
            record["instances"].gt_ibm = BitMasks(
                ideal_masks.permute(0, 2, 1).contiguous()
                )
            record["instances"].gt_eventgram = SoftMasks(
                event_grams.permute(0, 2, 1).contiguous()
                )
            for k, v in latents.items():
                if self.mask_types[k] == "bool":
                    setattr(
                        record["instances"],
                        f"gt_{k}",
                        BitMasks(v.permute(0, 2, 1).contiguous())
                        )
                else:
                    setattr(
                        record["instances"],
                        f"gt_{k}",
                        SoftMasks(v.permute(0, 2, 1).contiguous())
                        )

            # Bounding boxes for each instance
            boxes = []
            for (onset_idx, offset_idx), ideal_mask in zip(event_timing_idxs, ideal_masks):
                if self.box_height_mask == "ibm":
                    f = ideal_mask.nonzero()[:, 1].float()
                    fstop = max(f.quantile(0.99).long(), 1)
                    fstart = min(f.quantile(0.01).long(), fstop-1)
                elif self.box_height_mask == "padded":
                    f = ideal_mask.any(0).nonzero()
                    fstop = max(min(f.max()+5, ideal_mask.shape[1]-1), 1)
                    fstart = min(max(f.min()-5, 0), fstop-1)
                elif self.box_height_mask == "eventgram":
                    f = event_grams.sum(0).nonzero()
                    fstop = max(f.max(), 1)
                    fstart = min(f.min(), fstop-1)
                else:
                    raise NotImplementedError()

                if self.use_real_timepoints:
                    tstop = max(offset_idx, 1)
                    tstart = min(onset_idx, tstop-1)
                else:
                    t = ideal_mask.any(1).nonzero()
                    tstop = max(t.max(), 1)
                    tstart = min(t.min(), tstop-1)
                boxes.append([tstart, fstart, tstop, fstop])
            record["instances"].gt_boxes = Boxes(boxes)

            # Additional information
            record["sound_name"] = os.path.splitext(chunk_path)[0] + f"_{scene_dict_idx:08d}"
            record["sound_group"] = "generated"
            N, width, height = ideal_masks.shape
            record["height"] = height
            record["width"] = width

            records.append(record)

        return records

    def get_targets_from_event(self, source, event_idx, onset_idx, offset_idx, t_coch, f_coch, event_gram):
        """
        Transform latent variables from generative model into
        arrays which detectron2 can predict. Each latent should
        be represented as a cochleagram-size tensor (either bool
        or float).

        Return a zeros tensor if the source type doesn't include the latent
        See Appendix B > Event proposals via amortized inference - segmentation neural network > Training objective.

        Input
        -----
        source: Source object
        event: Event object
        event_idx: int
        t_coch: array[float], len = time dimension of cochleagram
        f_coch: array[float], len = frequency dimension of cochleagram
        scene_duration: float
        event_gram: Tensor

        Returns
        -------
        latents: dict[name, Tensor]
        """

        latent_types = self.gen_config["dream"]["event_features"]
        source_type = source.source_type

        def check_source_type(source_type, latent):
            return any(
                [source_type == latent_type for latent_type in latent_types[latent]]
            )

        # F0
        # Create binary image with 1
        # at the closest frequency for each timepoint
        f0 = torch.zeros((len(t_coch), len(f_coch)), dtype=bool)
        if check_source_type(source_type, "f0"):
            x = source.gps.f0.feature.gp_x[0, :, 0]
            y = source.gps.f0.feature.y_render[0, :, event_idx]
            y_interp = torch.from_numpy(np.interp(
                    t_coch[onset_idx:offset_idx+1],
                    x.numpy(),
                    y.numpy()
                ))
            closest_idx = (y_interp[None, :] - f_coch[:, None]).abs().argmin(dim=0)
            f0[range(onset_idx, offset_idx+1), closest_idx] = True

        # Amplitude
        amplitude_columns = torch.zeros((len(t_coch), len(f_coch)))
        # Multiply amplitude_columns by event_gram > threshold
        if check_source_type(source_type, "amplitude"):
            x = source.gps.amplitude.feature.gp_x[0, :, 0]
            y = source.gps.amplitude.feature.y_render[0, :, event_idx]
            amplitude = torch.from_numpy(np.interp(
                t_coch, x.numpy(), y.numpy(),
                left=y[0].item(), right=y[-1].item()
            )).to(torch.float32)
            amplitude_columns = amplitude[:, None].expand(-1, len(f_coch))

        # SPA: Spectrum plus amplitude (outer product)
        spectrum_plus_amplitude = torch.zeros((len(t_coch), len(f_coch)))
        if check_source_type(source_type, "spa"):

            # Get shifted spectrum/amplitude for this event
            shifted_spa = source.shifted_spa[event_idx, :, :].numpy()
            shift_mask = source.shift_masks[event_idx, :, :].numpy()

            # Interpolate to cochleagram times
            # f_gp: currently in scale, not Hz
            f_gp = source.gps.spectrum.feature.gp_x[0, :, 0].cpu().numpy()
            t_gp = source.gps.amplitude.feature.gp_x[0, :, 0].numpy()
            mask_interpol8r_in_time = scipy.interpolate.interp1d(
                t_gp, shift_mask, kind="nearest", fill_value='extrapolate', axis=0
                )
            spa_interpol8r_in_time = scipy.interpolate.interp1d(
                t_gp, shifted_spa, kind="nearest", fill_value='extrapolate', axis=0
                )
            shift_mask_repeat = mask_interpol8r_in_time(t_coch)  # shape: t, f
            shifted_spa_repeat = spa_interpol8r_in_time(t_coch)

            # If harmonic, account for interpolation process
            if source_type == "harmonic":
                # -100 is hardcoded in the estimation_grid code in dataset.py
                interp_acct = (1.0*(shifted_spa_repeat > -100)).argmax(1)
                # Assume interp_acct is corrupted by our interpolation
                # process in dataset.py, so use interp_acct+1
                copy_to = np.arange(shifted_spa_repeat.shape[1])[None] < interp_acct[:, None]+1
                copy_val = shifted_spa_repeat[
                    np.arange(shifted_spa_repeat.shape[0]),
                    np.clip(interp_acct+1, None, shifted_spa_repeat.shape[1]-1)
                    ]
                shifted_spa_repeat = copy_to*copy_val[:, None] + (1-copy_to)*shifted_spa_repeat

            # Interpolate to cochleagram frequencies
            mask_interpol8r = scipy.interpolate.interp1d(
                f_gp, shift_mask_repeat,
                kind='nearest', fill_value='extrapolate'
                )
            spa_interpol8r = scipy.interpolate.interp1d(
                f_gp, shifted_spa_repeat,
                kind='nearest', fill_value='extrapolate'
                )
            _shift_mask = torch.Tensor(
                mask_interpol8r(f_coch)[onset_idx:offset_idx+1, :]
                )
            _shifted_spa = torch.Tensor(
                spa_interpol8r(f_coch)[onset_idx:offset_idx+1, :]
                )
            # It's possible that the first part of the sound will get cut off
            # by the mask if it lies between two of the GP points.
            # fill it in with the mask at the first non-cutoff point
            first_val = (1.0*(_shift_mask.sum(1) > 0)).argmax()
            for vvidx in range(first_val):
                _shift_mask[vvidx, :] = _shift_mask[first_val, :]
                _shifted_spa[vvidx, :] = _shifted_spa[first_val, :]
            # Same with offset
            last_val = (1.0*(_shift_mask.sum(1) > 0)).flip([0]).argmax()
            for vvidx in range(-last_val, 0):
                _shift_mask[vvidx, :] = _shift_mask[-last_val-1, :]
                _shifted_spa[vvidx, :] = _shifted_spa[-last_val-1, :]

            # Place inside latent matrix
            XX, YY = np.meshgrid(range(len(f_coch)), range(onset_idx, offset_idx+1))
            spectrum_plus_amplitude[YY.ravel(), XX.ravel()] = _shifted_spa.flatten()

            # Before and after the event, use the first and last frame
            spectrum_plus_amplitude[:onset_idx, :] = _shifted_spa[0,  None, :]
            spectrum_plus_amplitude[offset_idx+1:, :] = _shifted_spa[-1, None, :]

        # Scale the amplitude and spa to appropriate scale for detectron2
        latents = {
            "f0": f0,
            "amplitude": amplitude_columns/100,
            "spa": spectrum_plus_amplitude/100
            }

        return latents


class BassTrainer(DefaultTrainer):
    """ Train on auditory scenes synthesized by bayesian model """

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls,cfg):
        n_gpus = torch.cuda.device_count()
        gpu_id = comm.get_rank()
        mask_types = dict([*cfg.BASA.MASK_TYPES, *cfg.BASA.TALLMASK_TYPES])

        print(f"Building loader on gpu {gpu_id}/{n_gpus}")
        for epoch_idx in range(50):
            print(f"Starting epoch {epoch_idx}")
            dataset = GeneratedScenesDataset(
                cfg.DATASETS.TRAIN,
                mask_types=mask_types,
                n_gpus=n_gpus,
                gpu_id=gpu_id,
                cfg=cfg
                )
            data_loader = DataLoader(
                dataset,
                batch_size=None,
                num_workers=cfg.DATALOADER.NUM_WORKERS
                )
            yield from data_loader

    @classmethod
    def build_test_loader(cls, cfg):
        mask_types = dict([*cfg.BASA.MASK_TYPES, *cfg.BASA.TALLMASK_TYPES])
        dataset = GeneratedScenesDataset(
            cfg.DATASETS.TEST, mask_types=mask_types, test=True, cfg=cfg
            )
        return DataLoader(dataset, batch_size=None)


class AudioDataset(torch.utils.data.IterableDataset):
    """Dataset of sound files to be segmented at inference time."""

    def __init__(self, dataset_name, demo_folder, extra_path="", batch_size=1, im_type=None, test=True, dataset_config=None, tf_rep=None, cfg=None):
        super().__init__()

        # Options
        self.im_type = "IBM" if im_type is None else im_type
        self.batch_size = batch_size

        # Get data_creation_config of how data was created
        self.tf_representation = cfg.BASA.TF_REPRESENTATION
        self.im_cutoff_ceil = cfg.BASA.IM_CUTOFF_CEIL
        self.use_rel_threshold = cfg.BASA.USE_REL_THRESHOLD

        cgram_util.set_gen_config(
            self, dataset_name,
            dataset_config=dataset_config,
            network_tf_rep=self.tf_representation
            )
        self.max_train_duration = self.gen_config["dream"]["dur"][-1]
        self.trim_to_max_duration = True
        self.default_trim_start = 0.0

        # Sounds to segment
        self.sound_group = demo_folder
        sound_folder = os.path.join(
            os.environ["sound_dir"], demo_folder, extra_path
            )
        self.sound_paths = glob(os.path.join(sound_folder, "*.wav"))

    def __iter__(self):
        for sound_path in self.sound_paths:
            print(sound_path, flush=True)
            scene_wave, audio_sr = sf.read(sound_path)
            scene_wave = self.check_shape(scene_wave)
            tf_dict, scene_wave, audio_sr = cgram_util.make_tf_for_segmentation(
                self, scene_wave, audio_sr
                )
            record = self.make_record(
                tf_dict, scene_wave, audio_sr, sound_path
                )
            yield record

    def make_record(self, tf_dict, scene_wave, audio_sr, sound_path):
        record = {}
        record["inputs"] = tf_dict
        record["scene_wave"] = scene_wave
        record["audio_sr"] = audio_sr
        record["sound_name"] = os.path.splitext(os.path.basename(sound_path))[0]
        record["sound_group"] = self.sound_group
        if self.tf_representation == "cgm":
            scene_gram = cgram_util.input_threshold(
                self, tf_dict["grams"],
                use_rel_threshold=self.use_rel_threshold
                )
        else:
            scene_gram = tf_dict["grams"]
        scene_gram = scene_gram.numpy()
        img, _ = tf_to_image(self, scene_gram)
        record["image"] = img.transpose(1, 2, 0).astype(np.float32)
        return record

    def check_shape(self, wave):
        wave = np.squeeze(wave)
        if len(wave.shape) > 1:
            raise Exception("Multichannel audio?")
        wave = wave[None, :]
        return wave


def tf_to_image(self, scene):
    """Add channels to basic cochleagram to make up
        for differences with (visual) images

       See Appendix B > Event proposals via amortized inference - segmentation neural network > Training dataset
    """

    im_cutoff_ceil = self.im_cutoff_ceil
    if im_cutoff_ceil == -1:
        im_cutoff_ceil = 120.0 if self.tf_representation == "cgm" else 180.0

    img = scene.transpose(1, 0)
    ft_shape = img.shape
    img = img[None, :, :]

    mask = (img > self.abs_threshold).astype(img.dtype)

    img = img - self.abs_threshold
    img = (img / im_cutoff_ceil)*255.0

    boundaries = np.ones((1, *img.shape[1:]), dtype=img.dtype)
    # full_img: channels=3, freqs, time
    full_img = np.concatenate((img, mask, boundaries), axis=0)

    return full_img, ft_shape
