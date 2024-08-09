import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from util.sample import rsample
from util.context import context
from model.scene_module import SceneModule
from model.sequence import Sequence
from model.gaussian_processes import GP
from renderer.trimmer import SceneTrimmer, EventTrimmer
import renderer.excitation as excitation
import renderer.am_and_filters as am_and_filters


class Source(SceneModule):

    def __init__(self, source_priors):
        super().__init__()
        # - Set (fixed) prior over discrete latents
        self.register_buffer("source_type_probs", torch.full([
            len(source_priors["source_type"]["args"])],
            1/len(source_priors["source_type"]["args"])
            ))
        self.source_type_dist = Categorical
        self.source_types = source_priors["source_type"]["args"]

    @classmethod
    def hypothesize(cls, hypothesis, source_priors, rendering_arrays, learn_hp=False, **kwargs):
        """
        Creates a new instance of Source where the variational distribution
        is initialized based on a hypothesis
        
        Parameters
        ----------
        hypothesis: dict
            Defines the variational distribution, that is:
            - Settings for all discrete latent variables
            - A hypothesis for continuous latent variables (onset, offset, gp ys)
        source_priors: dict
            Defines the model, i.e., the "meta-source" parameters of
            source_priors e.g. concentration0, a0_f0
        rendering_arrays: RenderingArrays
            see scene.py
        """

        # Instantiate a module
        #     Sets prior over discrete latent variables
        self = cls(source_priors)

        # Set constants
        self.r = rendering_arrays

        # Set all discrete latent variables at this level, based on hypothesis
        self.set_discrete(hypothesis)

        # Instantiate all submodules with `hypothesize`: discrete, p, q.
        self.sequence = Sequence.hypothesize(
            hypothesis, source_priors, learn_hp=learn_hp
            )
        self.gps = nn.ModuleDict({
            gp_type: GP.hypothesize(
                hypothesis, source_priors[self.source_type][gp_type], gp_type, learn_hp=learn_hp
                ) for gp_type in source_priors[self.source_type].keys()
                })

        # Set up appropriate renderer based on source type
        self.batched_events = any([self.gps[k].non_stationary for k in self.gps.keys()])
        self.build_renderer()

        return self

    @classmethod
    def sample(cls, source_priors, rendering_arrays):
        """
        Creates a new instance of Source by sampling

        Parameters
        ----------
        source_priors: dict
            Defines the model, i.e., the "meta-source" parameters of source_priors e.g. concentration0, a0_f0
        rendering_arrays: RenderingArrays
            see scene.py
        """

        # Instantiate a module
        #     Sets prior over discrete latent variables
        self = cls(source_priors)

        # Set constants
        self.r = rendering_arrays

        # Sample all discrete latent variables at this level
        self.sample_discrete()

        # Instantiate all submodules with `hypothesize`: discrete, p, q.
        self.sequence = Sequence.sample(source_priors)
        self.gps = nn.ModuleDict({gp_type: GP.sample(
            source_priors[self.source_type][gp_type], gp_type, self.sequence.events, self.r
            ) for gp_type in source_priors[self.source_type].keys()
            })

        # Set up appropriate renderer based on source type, and generate sound
        self.batched_events = any([self.gps[k].non_stationary for k in self.gps.keys()])
        self.build_renderer()
        self.generate(dream=True)

        return self

    def set_discrete(self, hypothesis):
        """ Set source type to hypothesis """
        self.source_type = hypothesis["source_type"]
        self.register_buffer("source_type_idx", torch.Tensor(
            [[i for i, t in enumerate(self.source_types) if t == self.source_type][0]]))

    def sample_discrete(self):
        """ Sample source type from categorical distribution """
        self.register_buffer("source_type_idx", self.source_type_dist(probs=self.source_type_probs).sample())
        self.source_type = self.source_types[self.source_type_idx]

    def init_p(self, source_priors):
        pass

    def init_q(self, source_priors):
        pass

    def log_q(self, sample=rsample):
        """ Sample source latent variables under the variational distribution
            and return log probability
        """
        temporal_qs = self.sequence.log_q(sample)
        gp_qs = torch.zeros(temporal_qs.shape,device=temporal_qs.device)
        for k in self.gps.keys():
            gp_q = self.gps[k].log_q(self.sequence.events, self.r, sample)
            gp_qs = gp_q + gp_qs
        return temporal_qs + gp_qs

    def log_p(self):
        """ Return log probability of source latent variables under prior
            see expansion of p(source) in Eqn. 2 of Appendix A
        """
        temporal_p = self.sequence.log_p()
        discrete_p = self.log_p_discrete(temporal_p.shape)
        
        gp_ps = torch.zeros(temporal_p.shape, device=temporal_p.device)
        for k in self.gps.keys():
            gp_p = self.gps[k].log_p()
            gp_ps += gp_p

        return discrete_p + temporal_p + gp_ps
    
    def log_p_discrete(self, p_shape):
        """ Return log probability of source type under prior"""
        # Repeat because source type probability is equal for all samples in batch
        discrete_p = self.source_type_dist(probs=self.source_type_probs).log_prob(self.source_type_idx).repeat(p_shape[0])
        return discrete_p

    # Rendering
    def build_renderer(self):
        """ Set the type of the renderer """
        if self.source_type == "whistle":
            self.renderer = WhistleRenderer(self.batched_events, self.r, self.source_type, context.renderer["source"][self.source_type])
        elif self.source_type == "harmonic":
            self.renderer = HarmonicRenderer(self.batched_events, self.r, self.source_type, context.renderer["source"][self.source_type])
        elif self.source_type == "noise":
            self.renderer = NoiseRenderer(self.batched_events, self.r, self.source_type, context.renderer["source"][self.source_type])

    def generate(self, start_time=None, end_time=None, dream=False):
        """ Returns source wave based on sampled latent variables """
        events = self.sequence.events
        if start_time is not None:
            events = [el for el in events if max(el.offset.timepoint) > start_time - context.renderer["ramp_duration"]]
        if end_time is not None:
            events = [el for el in events if min(el.onset.timepoint) < end_time + context.renderer["ramp_duration"]]

        if len(events)==0:
            self.source_wave, self.event_waves = self.renderer.silence(context.batch_size, self.sequence.n_events)
        else:
            self.source_wave, self.event_waves = self.renderer(self.gps, events, self.r, dream=dream)
        return self.source_wave

    # Sequential inference
    def update(self, event_proposal, source_priors, config):
        """ Add new event to source during a new round of sequential inference """
        with context(scene=self.scene):
            new_event, updated_events = self.sequence.update(event_proposal["events"][0], source_priors)
            for gp in self.gps.values():
                gp.update(event_proposal, source_priors, updated_events, config)
        return new_event

    def check_fit(self, event_proposal):
        """Checks if source types match, and degree of overlap with previous events."""
        if event_proposal["source_type"] != self.source_type:
            return False
        else:
            return self.sequence.check_fit(event_proposal["events"][0])


class Renderer(SceneModule):
    """ Parent module for renderers -- see Appendix A, Generative model: likelihood  """

    def __init__(self, batched_events, r, source_type, trim_events=True):
        """
        Parameters
        ----------
        batched_events: bool
            Whether the source's gaussian processes use non-stationary events
        r: RenderingArrays
            see scene.py
        source_type: str
            options: whistle, noise, or harmonic
        trim_events: bool
            Saves memory by rendering all events on shortest arrays possible
        """
        super().__init__()
        self.ramps = excitation.Ramps(batched_events)
        if trim_events:
            self.trimmer = EventTrimmer(batched_events, r, source_type)
        else:
            self.trimmer = SceneTrimmer(batched_events, r, source_type)

        n_samples = int(np.round(self.audio_sr * self.scene_duration))
        self.register_buffer("zeros", torch.zeros(n_samples))

    def silence(self, batch_size, n_events):
        source_wave = self.zeros[None, :].expand(batch_size, -1)
        event_waves = self.zeros[None, None, :].expand(batch_size, n_events, -1)
        return source_wave, event_waves


class HarmonicRenderer(Renderer):
    """ Synthesize a harmonic source sound from sampled latent variables """

    def __init__(self, batched_events, r, source_type, renderer_options):
        super().__init__(batched_events, r, source_type, trim_events=renderer_options['trim'] if 'trim' in renderer_options else True)
        if renderer_options["source_colour"] == "pink":
            attenuation_constant = 0
        elif renderer_options["source_colour"] == "attenuated_pink":
            attenuation_constant = renderer_options["attenuation_constant"]
        else:
            raise Exception("Please use pink or attenuated_pink")
        self.excitation = excitation.Periodic(r, source_type, source_colour=renderer_options["source_colour"], attenuation_constant=attenuation_constant, n_harmonics=renderer_options.get("n_harmonics", 200))
        if renderer_options["panpipes"]:
            self.AM_and_filter = am_and_filters.Panpipes(r)
        else:
            self.AM_and_filter = am_and_filters.AMFilterHarmonic(r)

    def forward(self, gps, events, r, dream=False):
        FM_excitation = self.excitation(gps, events, r, self.trimmer)
        filtered_AM_wave = self.AM_and_filter(FM_excitation, gps, events, r, self.trimmer, f0=self.excitation.rendered_f0)
        source_wave, event_waves = self.ramps(filtered_AM_wave, events, r, self.trimmer, dream=dream)
        return source_wave, event_waves


class WhistleRenderer(Renderer):
    """ Synthesize a whistle source sound from sampled latent variables """

    def __init__(self, batched_events, r, source_type, renderer_options):
        super().__init__(batched_events, r, source_type, trim_events=renderer_options['trim'] if 'trim' in renderer_options else True)
        self.excitation = excitation.Periodic(r, source_type)
        self.AM = am_and_filters.AmplitudeModulation(r)

    def forward(self, gps, events, r, dream=False):
        FM_excitation = self.excitation(gps, events, r, self.trimmer)
        FM_AM_wave = self.AM(FM_excitation, gps, events, r, self.trimmer)
        source_wave, event_waves = self.ramps(FM_AM_wave, events, r, self.trimmer, dream=dream)
        return source_wave, event_waves


class NoiseRenderer(Renderer):
    """ Synthesize a noise source sound from sampled latent variables """

    def __init__(self, batched_events, r, source_type, renderer_options):
        super().__init__(batched_events, r, source_type, trim_events=renderer_options['trim'] if 'trim' in renderer_options else True)
        self.excitation = excitation.Aperiodic(r, renderer_options)
        self.AM_and_filter = am_and_filters.AMFilterNoise(r)

    def forward(self, gps, events, r, dream=False):
        excitation_wave = self.excitation(gps, events, r, self.trimmer)
        filtered_AM_wave = self.AM_and_filter(excitation_wave, gps, events, r, self.trimmer)
        source_wave, event_waves = self.ramps(filtered_AM_wave, events, r, self.trimmer, dream=dream)
        return source_wave, event_waves
