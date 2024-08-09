
import io
import os
import dill
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions.poisson import Poisson
from torch.distributions.categorical import Categorical 
import numpy as np
import soundfile as sf

from model.source import Source
from model.scene_module import SceneModule
import renderer.cochleagrams as cgram
import renderer.util
from inference.proposals_meta import SceneMeta
from util.sample import rsample
from util.context import context


class Scene(nn.Module):
    """ Full description of auditory scene """

    def __init__(self, hyperpriors):
        super().__init__()
        # - Set any constants, etc.
        # - Set (fixed) prior over discrete latents
        if hyperpriors["n_sources"]["dist"] == "poisson":
            self.register_buffer("n_sources_lambda", torch.Tensor([hyperpriors["n_sources"]["args"]]))
            self.n_sources_dist = Poisson
        elif hyperpriors["n_sources"]["dist"] == "uniform_discrete":
            self.register_buffer("n_sources_probs", torch.full([1+hyperpriors["n_sources"]["args"]], (1/(1+hyperpriors["n_sources"]["args"][1]))))
            self.n_sources_dist = Categorical(probs=self.n_sources_probs)
        elif hyperpriors["n_sources"]["dist"] == "categorical":
            self.register_buffer("n_sources_probs", torch.Tensor(hyperpriors["n_sources"]["args"]))
            self.n_sources_dist = Categorical(probs=self.n_sources_probs)

    @property
    def cuda_device(self):
        return self.off_ramp.device

    @classmethod
    def hypothesize(cls, hypothesis, hyperpriors, likelihood, observation, audio_sr):
        """
        Creates a new instance of Scene where the variational distribution is initialized based on a hypothesis
        
        Parameters
        ----------
        hypothesis: dict
            Defines the variational distribution, that is:
            - Settings for all discrete latent variables
            - A hypothesis for continuous latent variables (onset, offset, gp ys)
        hyperpriors: dict
            Defines the model, i.e., the parameters of hyperpriors e.g. concentration0, a0_f0
        likelihood: dict
            config that defines gaussian noise likelihood on cochleagram
        observation: numpy array
            signal to condition on, typically of size (signal_length,)
        audio_sr: int
            sampling rate of observation
        """

        # Instantiate a module
        #     Sets prior over discrete latent variables
        self = cls(hyperpriors)
        self.meta = SceneMeta()
        
        # Set constants
        #     In scene: observe sound 
        if len(observation.shape) == 1:
            observation = observation[None, :]
        self.audio_sr = audio_sr

        with context(scene=self):
            # Initialize the observation and necessary matrices for sound synthesis
            self.init_cochleagram(observation)
            self.observe_sound(likelihood, observation)
            self.r = RenderingArrays()

            # Set all discrete latent variables at this level, based on hypothesis
            self.set_discrete(hypothesis)

            # Instantiate all submodules with `hypothesize`: discrete, p, q.
            #     In scene: sources
            # self.sources = nn.ModuleList([Source.hypothesize(hypothesis[source_idx], hyperpriors, self.r) for source_idx in range(self.n_sources.item())])
            sources = []
            for source_idx in range(self.n_sources.item()):
                if type(hypothesis[source_idx]) is dict:
                    new_source = Source.hypothesize(hypothesis[source_idx], hyperpriors, self.r)
                elif isinstance(hypothesis[source_idx],Source):
                    new_source = deepcopy(hypothesis[source_idx])
                    new_source.r = self.r
                    new_source.build_renderer()
                sources.append(new_source)
                self.meta.add_source(new_source)
            self.sources = nn.ModuleList(sources)

        return self
    
    @classmethod
    def sample(cls, hyperpriors, audio_sr, scene_duration):
        """
        Creates a new instance of Scene by sampling a scene description
        
        Parameters
        ----------
        hyperpriors: dict
            - Defines the model
            - Parameters of hyperpriors e.g. concentration0, a0_f0
        audio_sr: int
            sampling rate of sampled sounds
        scene_duration: float
            duration in seconds of sampled sounds
        """
        # Instantiate Scene - sets prior over discrete latent variables
        self = cls(hyperpriors)
        self.audio_sr = audio_sr; self.scene_duration = scene_duration;

        with context(scene=self):
            # Set/sample constants
            self.init_cochleagram(sampling=True)
            self.r = RenderingArrays(persistent=False)

            # Sample all discrete latent variables at this level, number of sources
            self.sample_discrete(hyperpriors)

            # Instantiate all submodules with `sample`: sample discrete, initialize p, sample from p
            self.sources = nn.ModuleList([Source.sample(hyperpriors, self.r) for source_idx in range(self.n_sources)])

            # Create scene sound from rendered source sounds
            self.scene_wave = torch.sum(torch.stack([source.source_wave*self.off_ramp[None,:] for source in self.sources]),dim=0)

        return self

    def set_discrete(self, hypothesis):
        """ Set the number of sources depending on the hypothesis """
        self.register_buffer("n_sources", torch.Tensor([len(hypothesis)]).long())

    def sample_discrete(self, hyperpriors):
        """ Sample the number of sources """
        if self.n_sources_dist is Poisson:
            self.n_sources = 0
            while self.n_sources == 0:
                self.n_sources = self.n_sources_dist(self.scene_duration*self.n_sources_lambda).sample().long()
        else: #Categorical()
            self.n_sources = self.n_sources_dist.sample()

    def init_p(self, hyperpriors):
        pass

    def init_q(self, hypothesis, hyperpriors):
        pass

    def log_p_discrete(self):
        """ Score number of sources under the prior
            See p(n) in Appendix A > section A.1, Generative model: Overview and Table A.1
        """
        if self.n_sources_dist is Poisson:
            lp = self.n_sources_dist(self.n_sources_lambda).log_prob(self.n_sources).expand(context.batch_size)
        else: #Categorical()
            lp = self.n_sources_dist.log_prob(self.n_sources).expand(context.batch_size)
        return lp

    def log_p(self):
        """ Sample from and score under variational distribution
            See p(n) in Appendix A > section A.1, Generative model: Overview and Table A.1
        """
        lp = self.log_p_discrete()
        for i in range(self.n_sources):
            lp = lp + self.sources[i].log_p().to(self.cuda_device)
        return lp

    def log_q(self, sample=rsample):
        """ Score sample under prior """
        lq = 0
        for i in range(self.n_sources):
            lq = lq + self.sources[i].log_q(sample).to(self.cuda_device)
        return lq

    ## Rendering
    def build_renderer(self):
        with context(scene=self):
            self.r = RenderingArrays()
            for source in self.sources:
                source.r = self.r
                source.build_renderer()

    def clear_renderer(self):
        del self.r
        for source in self.sources:
            del source.r
            del source.renderer

    def init_cochleagram(self, observation=None, sampling=False):
        """ Initialize the matrices required to compute a cochleagram. See Appendix A, Generative model: likelihood """
        if observation is not None:
            scene_len = observation.shape[1]
        else:
            scene_len = int(np.round(self.scene_duration*self.audio_sr))

        self.rms_ref = context.renderer["tf"]["rms_ref"]
        scene_offset_ramp_duration = context.renderer["tf"]["ramp"]
        if sampling: 
            #if sampling, use cgram_util instead of putting it in the scene
            #allows us to reuse the cgm/gtg design from sample to sample
            #also saves memory because these matrices aren't saved with the Scene. 
            self.representations = context.dream["tf_representation"]
        else:
            self.representation = context.renderer["tf"]["representation"]
            if self.representation == "gtg":
                #Gammatonegrams with Dan Ellis' approximation
                gtg_params = context.renderer["tf"]["gtg_params"]
                self.dB_threshold = gtg_params["dB_threshold"]
                self.nfft, self.nhop, self.nwin, gtm, self.gtg_center_freqs = cgram.gtg_settings(sr=self.audio_sr,twin=gtg_params["twin"],thop=gtg_params["thop"],N=gtg_params["nfilts"],fmin=gtg_params["fmin"],fmax=self.audio_sr/2.0,width=gtg_params["width"], return_all=True)
                self.thop = gtg_params["thop"]
                self.register_buffer("gtm", torch.Tensor(gtm[np.newaxis,:,:]))
                self.register_buffer("window", torch.hann_window(self.nwin))
            elif self.representation == "cgm":
                #Cochleagrams (Feather et al., 2022; chcochleagram)
                cgm_params = context.renderer["tf"]["cgm_params"]
                cochleagram_ops, self.cgm_threshold = cgram.cgm_settings(scene_len, self.audio_sr, cgm_params)
                self.cgm = nn.ModuleDict(cochleagram_ops)
        
        #Ramp to make sure that sounds don't have artifacts at the end of the scene
        off_ramp = torch.Tensor(renderer.util.hann_ramp(self.audio_sr, ramp_duration = scene_offset_ramp_duration))
        self.register_buffer("off_ramp", torch.cat((torch.ones(scene_len-off_ramp.shape[0]),off_ramp),dim=0))

    def observe_sound(self, likelihood, observation):
        """ Register an observation so the model can be conditioned on that observed signal during inference."""
        self.register_buffer("likelihood_sigma", torch.Tensor([likelihood["args"]]))
        if self.representation == "gtg":
            C_obs = cgram.gammatonegram(torch.Tensor(observation)*self.off_ramp, self.nfft, self.nhop, self.nwin, self.gtm, self.rms_ref, self.window, dB_threshold=self.dB_threshold)
            self.register_buffer("C_obs", C_obs)
        elif self.representation == "cgm":
            C_obs = self.cgm_threshold(self.cgm.cochleagram(torch.Tensor(observation)*self.off_ramp))
            self.register_buffer("C_obs", C_obs)
        self.scene_duration = observation.shape[-1]/self.audio_sr
        
    def generate(self, start_time=None, end_time=None, dream=False):
        """ Generate source waves and sum them to produce the scene wave."""
        for i in range(self.n_sources):
            source_wave = self.sources[i].generate(start_time, end_time, dream=dream).to(self.cuda_device)
            #Apply off-ramp to avoid edge artefacts
            if i == 0:
                self.scene_wave = source_wave * self.off_ramp[None,:]
            else:
                self.scene_wave = self.scene_wave + source_wave * self.off_ramp[None,:]
        return self.scene_wave
    
    @property
    def C_scene(self):
        """ Returns the rendered cochleagram """
        if self.representation == "gtg":
            C = cgram.gammatonegram(self.scene_wave, self.nfft, self.nhop, self.nwin, self.gtm, self.rms_ref, self.window, dB_threshold=getattr(self,"dB_threshold",20))
            return C       
        elif self.representation == "cgm":
            C = self.cgm_threshold(self.cgm.cochleagram(self.scene_wave))
            return C

    def log_likelihood(self, start_idx=None, end_idx=None):
        """ Computes the log likelihood from the observed and rendered cochleagrams. See Appendix A, Generative model: likelihood """
        if start_idx is None:
            C_scene = self.C_scene
            C_obs = self.C_obs
        else:
            C_scene = self.C_scene[:, :, start_idx:end_idx]
            C_obs = self.C_obs[:, :, start_idx:end_idx]

        C_scene = C_scene.reshape(context.batch_size,-1)
        #sI = self.likelihood_sigma.expand(context.batch_size)[:,np.newaxis]
        C_obs = C_obs.repeat(context.batch_size, 1, 1).reshape(context.batch_size, -1)
        
        # ll = Normal(C_scene, sI).log_prob(C_obs).sum(1) 
        # Note: removing a constant (given sound duration) so that you don't 'pay extra' for silence on longer sounds
        ll = -0.5 * (C_scene-C_obs).square().sum(1) / self.likelihood_sigma**2

        return C_scene, ll 

    ## Inference methods
    # Variational inference
    def pq_forward(self, sample=rsample):
        """ Computes the lower bound on the log marginal probability (see Appendix B Eqn 5) """
        self.lq = self.log_q(sample) #Takes a sample and evaluates log q
        self.lp = self.log_p() #Measures prior on sample
        self.scene_wave = self.generate() #Synthesize sampled sounds
        _, self.ll = self.log_likelihood() #Compute log likelihood 
        score = self.ll - self.lq + self.lp
        return score

    # Iterative inference with event proposals
    def update(self, event_proposal, hyperpriors):
        """Create a new source with event proposal"""
        with context(scene=self):
            new_source = Source.hypothesize(event_proposal, hyperpriors, self.r)
            self.sources.append(new_source)
            self.meta.add_source(new_source)
            self.n_sources += 1
        return new_source

    def check_iou(self, event_proposal):
        """ Checks whether new event proposal can be combined into this scene based on IOU """
        passes_iou_threshold = []
        for source in self.sources:
            passes_iou_threshold.append(source.sequence.check_iou(event_proposal))
        print("Passes IOU threshold: ", all(passes_iou_threshold))
        return all(passes_iou_threshold)

    def update_observation(self, observation, likelihood):
        """Updates observation, cochleagrams, and renderers to match, given a new sound to condition on."""
        self.init_cochleagram(observation)
        self.observe_sound(likelihood, observation)
        self.build_renderer()
        for source in self.sources:
            for gp in source.gps.values():
                # If scene duration increases, newly-active inducing points must be added to variational strategy
                gp.feature.update_variational_strategy() 

    def __deepcopy__(self, memo):
        self.__deepcopy__ = None
        with context(scene=True):
            x = deepcopy(self, memo)
        del x.__deepcopy__
        del self.__deepcopy__
        return x

    def clone(self):
        """Deepcopy doesn't work for scenes with non-leaf tensors. Use this instead."""
        buffer = io.BytesIO()
        torch.save(self, buffer, pickle_module=dill)
        return torch.load(io.BytesIO(buffer.getvalue()))


class RenderingArrays(SceneModule):
    """ Matrices required for rendering, which are constant given a sound of a certain duration. """
    
    def __init__(self, persistent=True):
        super().__init__()

        #Defining time and frequency points at which functions are sampled
        self.steps = context.renderer["steps"]
        self.rms_ref = context.renderer["tf"]["rms_ref"]
        self.rendering_constant = 1e-12

        self.register_buffer("scene_timepoints", torch.arange(0, self.scene_duration, 1./self.audio_sr), persistent=persistent)
        self.t = renderer.util.get_event_gp_times(0, self.scene_duration, self.steps["t"]) 
        self.dt = len(self.t)
        self.register_buffer("gp_t", torch.Tensor(np.reshape(self.t,(1,self.dt,1))), persistent=persistent)
        self.f = renderer.util.get_event_gp_freqs(self.audio_sr, self.steps)
        self.df = len(self.f)
        self.register_buffer("gp_f", torch.Tensor(np.reshape(self.f,(1,self.df,1))), persistent=persistent)
        self.win_pad = 1 #one on each side.

        #Filterbanks and window arrays based on sampled freq/time points
        win_arr, self.sig_len = renderer.util.make_windows_rcos_flat_no_ends(self.dt+(2*self.win_pad), self.steps["t"], self.audio_sr) 
        filterbank, bandwidthsHz, filt_freq_cutoffs, filt_freqs = renderer.util.make_overlapping_freq_windows( self.dt+(2*self.win_pad), self.steps, self.audio_sr)
        self.filterbank_numpy = filterbank
        self.register_buffer("filterbank", torch.Tensor(filterbank[np.newaxis,:,:]), persistent=persistent)
        self.register_buffer("bandwidthsHz", torch.Tensor(bandwidthsHz[np.newaxis, :, :]), persistent=persistent)
        self.register_buffer("filt_freqs", torch.Tensor(filt_freqs), persistent=persistent)

        self.excitation_duration = (1.0 * self.sig_len)/self.audio_sr
        self.extra_spacing = (self.excitation_duration-self.scene_duration)/2.
        #this could be a sample longer/shorter than win_arr due to floating point error
        new_timepoints = torch.arange(-self.extra_spacing, self.scene_duration+self.extra_spacing, 1./self.audio_sr)
        win_arr = win_arr[np.newaxis, :, :]
        if win_arr.shape[1] < new_timepoints.shape[0]:
            new_timepoints = new_timepoints[:win_arr.shape[1]] 
        elif win_arr.shape[1] > new_timepoints.shape[0]:
            win_arr = win_arr[:, :new_timepoints.shape[0], :]
        self.register_buffer("win_arr", torch.Tensor(win_arr), persistent=persistent)
        self.register_buffer("new_timepoints", new_timepoints, persistent=persistent)