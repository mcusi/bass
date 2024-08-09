from copy import deepcopy
import torch.nn as nn
import torch
import numpy as np

from model.scene import Scene, RenderingArrays
from model.source import Source
from util.context import context
from util.sample import sample_delta_nograd, rsample


######################
# Inference of meta-source parameters
# - We infer the meta-source parameters, as shared by multiple scenes
# - For tractability, we assume that each scene has only one source
# - However, the inferences are based on real sounds which may not be completely explained by a single source (e.g. due to background noise)
# - Therefore these classes include strategies to facilitate stable inference (e.g., conditioning on a subset of latent variables)
# See the Appendix A, section A.3, "Generative model: Source priors"
######################

class SceneShared(Scene):
    """ a Scene with optimizable meta-source parameters, which are common to other SceneShared objects in Scenes """

    def __init__(self, source_priors, scene_idx):
        super().__init__(source_priors)
        self.scene_idx = scene_idx

    @classmethod
    def hypothesize(cls, hypothesis, source_priors, likelihood, observation, audio_sr, observed_latents, scene_idx):
        """
        Creates a new instance of SharedScene where the variational distribution is initialized based on a hypothesis
        
        Parameters
        ----------
        hypothesis: dict
            Defines the variational distribution, that is:
            - Settings for all discrete latent variables
            - A hypothesis for continuous latent variables (onset, offset, gp ys)
        source_priors: dict
            Meta-source parameters define the model, i.e., the parameters of source priors such as concentration0, a0_f0
        likelihood: dict
            config that defines gaussian noise likelihood on cochleagram
        observation: numpy array
            signal to condition on, of size (signal_length,)
        audio_sr: int
            sampling rate of observation
        observed_latents: dict
            indicates whether latent variables are observed or not. {"events": "bool", "gp_feature": bool, ... }
        scene_idx: int
            index for this SharedScene within Scenes
        """

        # Sets prior over discrete latent variables
        self = cls(source_priors,scene_idx)

        # Set constants
        #     In SceneShared: observe sound 
        if len(observation.shape) == 1:
            observation = observation[None, :]
        self.audio_sr = audio_sr

        with context(scene=self):
            self.init_cochleagram(observation)
            self.observe_sound(likelihood, observation)
            self.r = RenderingArrays()

            # Set all discrete latent variables at this level, based on hypothesis
            self.set_discrete(hypothesis)

            # Instantiate all submodules with `hypothesize`: discrete, p, q.
            #     In SceneShared: Sources
            self.sources = nn.ModuleList([SourceShared.hypothesize(hypothesis[source_idx], source_priors, self.r, learn_hp=True) for source_idx in range(self.n_sources.item())])
            for source in self.sources:
                source.observe_latents(observed_latents)

        return self

    def learn_source_priors(self):
        """ Similar to pq_forward but makes use of learn_source_priors method in SourceShared """
        for i in range(self.n_sources):
            lq, lp = self.sources[i].learn_source_priors()
            self.lq = lq if i == 0 else self.lq + lq
            self.lp = lp if i == 0 else self.lp + lp
        self.scene_wave = self.generate()
        _, self.ll = self.log_likelihood()
        score = self.ll - self.lq + self.lp
        return score

    def learn_source_priors_from_events(self, gp_type):
        """ Learning metasource parameters conditioned on event-level variables, without rendering to audio """
        self.lq = self.sources[0].gps[gp_type].log_q_source()
        self.lp = self.sources[0].gps[gp_type].log_p()
        return -self.lq + self.lp

    def set_likelihood_mask(self):
        """ Limiting the observation to the loudest gammatonegram bin in order to ignore noise in observations """
        self.likelihood_mask = (self.C_scene[0] == self.C_scene[0].max(0, keepdim=True)[0]) * (self.C_scene[0] > self.C_scene[0].min())

    def log_likelihood(self, start_idx=None, end_idx=None):
        """ Computing log likelihood with the use of the likelihood mask """
        C_scene = self.C_scene * getattr(self, "likelihood_mask", 1)
        C_obs = self.C_obs * getattr(self, "likelihood_mask", 1)
        
        C_scene = C_scene.reshape(context.batch_size,-1)
        # sI = self.likelihood_sigma.expand(context.batch_size)[:,np.newaxis]
        C_obs = C_obs.repeat(context.batch_size, 1, 1).reshape(context.batch_size, -1)
        
        # ll = Normal(C_scene, sI).log_prob(C_obs).sum(1) 
        # Note: removing a constant (given sound duration) so that you don't 'pay extra' for silence on longer sounds
        ll = -0.5 * (C_scene-C_obs).square().sum(1) / self.likelihood_sigma**2

        return C_scene, ll 


class SourceShared(Source):
    """ Like Source, but with extra methods for controlling which latent variables are observed """

    def __init__(self,source_priors):
        super().__init__(source_priors)

    def observe_latents(self, observed_latents):
        """ Define which latents are observed (True) and which need to be inferred (False) """
        self.observed_latents = observed_latents

    def learn_source_priors(self):
        """ Similar to combining log_p and log_q from Source, but allows some latent variables to be observed """

        # Temporal
        # Actual onsets/offsets are already known
        # Only need to sample hyperparameters
        if self.observed_latents["events"]:
            temporal_q = self.sequence.log_q(sample=None, source_prior_sample=rsample, event_sample=sample_delta_nograd, detach_event_q=True)
        else:
            temporal_q = self.sequence.log_q()
        temporal_p = self.sequence.log_p()

        # Features
        gp_q = 0
        gps_to_use = self.gps.keys()
        for k in gps_to_use:
            if self.observed_latents[k]:
                #We observed this latent feature, and only need to sample hyperparameters
                gp_q += self.gps[k].log_q_source()
                self.gps[k].feature.set_x(self.sequence.events, self.r)
                gp_q += self.gps[k].feature.log_q(self.gps[k].mean_module, self.gps[k].covar_module, sample=sample_delta_nograd).detach()
            else:
                #We need to sample hyperparameters and the feature itself
                gp_q += self.gps[k].log_q(self.sequence.events, self.r)
        gp_p = 0
        for k in gps_to_use:
            gp_p += self.gps[k].log_p()

        return temporal_q + gp_q, temporal_p + gp_p


class Scenes(nn.Module):
    """ Contains several SceneShared in order to infer a common set of meta-source parameters """

    def __init__(self, n_scenes, observation_names, devices, max_per_device=15):
        super().__init__()
        self.n_scenes = n_scenes
        print("Using...", devices)
        max_per_device = max_per_device if "cuda" in devices[0] else n_scenes
        self.scenes_per_device = min(max_per_device,int(n_scenes/len(devices))) #divide scenes equally among devices
        self.scene_names = observation_names
        self.n_scenes = self.scenes_per_device * len(devices)
        device_assignments = [[devices[i] for j in range(self.scenes_per_device)] for i in range(len(devices))]
        self.device_assignments = [d for a in device_assignments for d in a]
        self.scene_assignments = np.array([[(i*self.scenes_per_device)+j for j in range(self.scenes_per_device)] for i in range(len(devices))]).transpose()
        print(f"Device assignments of {self.n_scenes} scenes: ", self.device_assignments)

    @classmethod
    def hypothesize(cls, hypotheses, observations, observation_names, audio_sr, config, devices):
        """
        Creates a new instance of Scenes where the variational distribution is initialized based on a set of hypotheses

        Parameters
        ----------
        hypotheses: list[dict]
            For each scene, defines the variational distribution
        observations: list[numpy array]
            for each scene, the signal to condition on, of size (signal_length,)
        observation_names: list[str]
            filename strings to aid in identifying the scenes
        audio_sr: int
            sampling rate of observation
        config: dict
            contains source priors, likelihood, and options for hierarchical inference
        devices: list[str] or str
            devices (cpu, cuda:0, etc.) for inference
        """
        
        if isinstance(devices,str): #cpu or cuda
            devices = [devices]
        self = cls(len(observations), observation_names, devices)
        #Define the relevant configs
        source_priors = config["hyperpriors"]
        likelihood = config["likelihood"]
        observed_latents = config["hierarchical"]["observed_latents"] 
        hp_to_learn = config["hierarchical"]["hp_to_learn"]

        for scene_idx in range(self.n_scenes):
            if len(hypotheses[scene_idx]) > 1:
                raise Exception("All scenes must have one source")

        #First source of first hypothesis gives the source type of all the scenes because we learn this separately
        self.source_type = hypotheses[0][0]["source_type"]
        source_priors = deepcopy(source_priors) #prevents yaml from being saved with buggy torch tensors in them.

        #Create the learnable hyperparameters shared by all scenes
        def is_learnable(source_prior_dict):
            if "learn" in source_prior_dict:
                return source_prior_dict['learn']
            elif ("bounds" not in source_prior_dict.keys()):
                return True
            else:
                return len(source_prior_dict["bounds"]) != 1
        if hp_to_learn["gp"]:
            self.feature_hp = nn.ModuleDict({feature: 
                    nn.ParameterDict({kernel_parameter: 
                        nn.Parameter(
                            torch.Tensor(source_priors[self.source_type][feature]["kernel"][kernel_parameter]["args"]).to(devices[0])
                        ) for kernel_parameter in source_priors[self.source_type][feature]["kernel"]["parametrization"] if is_learnable(source_priors[self.source_type][feature]["kernel"][kernel_parameter])   
                    })
                for feature in source_priors[self.source_type] if feature != "renderer"})
        if hp_to_learn["seq"]:
            self.sequence_hp = nn.ModuleDict({
                    interval: nn.ParameterDict({
                        k: nn.Parameter(torch.Tensor(source_priors[interval][k]["args"]).to(devices[0])) 
                    for k in ["mu", "precision"]}  ) 
                for interval in ["gap","duration"]})

        #Initialize the scenes with the learnable hyperparameters in the priors
        #Put the learnable parameters into the source prior dict used to initialize p
        self.scenes = []
        for scene_idx in range(self.n_scenes):
            if hp_to_learn["gp"]:
                for feature in self.feature_hp.keys():
                    for kernel_parameter in self.feature_hp[feature].keys():
                        source_priors[self.source_type][feature]["kernel"][kernel_parameter]["args"] = self.feature_hp[feature][kernel_parameter].to(self.device_assignments[scene_idx])
            if hp_to_learn["seq"]:
                for interval in ["gap", "duration"]:
                    for k in ["mu", "precision"]:
                        source_priors[interval][k]["args"] = self.sequence_hp[interval][k].to(self.device_assignments[scene_idx])
            self.scenes.append(SceneShared.hypothesize(hypotheses[scene_idx], source_priors, likelihood, observations[scene_idx], audio_sr, observed_latents, scene_idx
                ).to(self.device_assignments[scene_idx]))

        self.scenes = nn.ModuleList(self.scenes)
        scene_durations = [len(o)/(1.*audio_sr) for o in observations]
        self.register_buffer("scene_durations", torch.Tensor(scene_durations))

        return self

    def move_metasource_parameters(self, scene_idx):
        """ Transfer meta-source parameters to the correct device """
        source_idx = 0
        if hasattr(self, "feature_hp"):            
            for feature in self.feature_hp:
                for hp in self.feature_hp[feature].keys():
                    if "epsilon" in hp:
                        self.scenes[scene_idx].sources[source_idx].gps[feature].feature.epsilon = self.feature_hp[feature]["epsilon"].to(self.device_assignments[scene_idx])
                    else:
                        self.scenes[scene_idx].sources[source_idx].gps[feature].covar_module.hyperparams[hp].hp = self.feature_hp[feature][hp].to(self.device_assignments[scene_idx])

        if hasattr(self, "sequence_hp"):
            for interval in ["gap", "duration"]:
                for hp in ["mu", "precision"]:
                    setattr(getattr(self.scenes[scene_idx].sources[source_idx].sequence, interval), hp + "_hp", self.sequence_hp[interval][hp].to(self.device_assignments[scene_idx]) )

    def forward(self, scene_idxs):
        """ Return variational bound when conditioning on raw audio as observations """
        scores = []
        for scene_idx in scene_idxs:
            self.move_metasource_parameters(scene_idx)
        
        for scene_idx in scene_idxs:
            score = self.scenes[scene_idx].learn_source_priors()
            scores.append(score)

        scores = [score.to(self.device_assignments[0]) for score in scores]
        return torch.cat(scores)

    def event_forward(self, gp_type, scene_idxs):
        """ Return variational bound when conditioning on event-level variables as observations """
        scores = []
        for scene_idx in scene_idxs:
            self.move_metasource_parameters(scene_idx)

        for scene_idx in scene_idxs:
            score = self.scenes[scene_idx].learn_source_priors_from_events(gp_type)
            scores.append(score)

        scores = [score.to(self.device_assignments[0]) for score in scores]            
        return torch.cat(scores)