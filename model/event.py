import torch
import torch.nn as nn
from torch.distributions.log_normal import LogNormal
from torch.distributions.uniform import Uniform

from model.scene_module import SceneModule
from model.truncated_normal import TruncatedNormal
import renderer.util as dr
from util.diff import softplus, np_inv_softplus, softbound
from util.context import context
from util.sample import rsample


class Event(SceneModule):
    """ Latent variables representing the timing of an event """

    def __init__(self, event_idx):
        super().__init__()
        # Set (fixed) prior over discrete latents
        self.event_idx = event_idx
        self.onset = Switch("onset")
        self.offset = Switch("offset")

    @classmethod
    def hypothesize(cls, hypothesis, source_priors, event_idx):
        """
        Creates a new Event where variational distribution is initialized
        based on a hypothesis

        Parameters
        ----------
        hypothesis: dict[str, float]
            Defines the variational distribution via a hypothesis for
            continuous latent variables: {"onset": float, "offset": float}
        source_priors: dict[str, tuple]
            Parameters of source_priors that define the model:
            {"gap": (m, s), "duration": (m,s)}
        event_idx: int
            Index of event in Sequence
        """

        # Instantiate a module
        self = cls(event_idx)

        # Set constants: proposal rank and IOUs
        if "meta" in hypothesis.keys():
            import inference.proposals_meta
            self.meta = inference.proposals_meta.EventMeta(hypothesis)

        # Initialize p and q for continuous latent variables at this level
        self.init_p(source_priors, event_idx)
        self.init_q(hypothesis)

        return self

    @classmethod
    def sample(cls, source_priors, event_idx, last_offset):
        """
        Creates a new Event through sampling

        Parameters
        ----------
        source_priors: dict[str, tuple]
            Parameters of source_priors that define the model:
            {"gap": (m, s), "duration": (m,s)}
        event_idx: int
            Index of event in Sequence
        last_offset: float
            Time in seconds of the offset of the previous event
        """

        # Instantiate a module
        # Sets prior over discrete latent variables
        self = cls(event_idx)

        # Initialize p for continuous latent vars at this level then sample
        self.init_p(source_priors, event_idx)
        self.onset.sample(last_offset, source_priors["gap"])
        self.offset.sample(self.onset.timepoint, source_priors["duration"])

        return self

    # Initalize priors and variational distributions
    def init_p(self, source_priors, event_idx):
        """Initialize prior over Event"""
        self.onset.init_p(event_idx)
        self.offset.init_p(event_idx)

    def init_q(self, hypothesis):
        """Initialize guide distribution over Event"""
        self.onset.init_q(hypothesis["onset"])
        self.offset.init_q(hypothesis["offset"])

    # Sample and compute log probabilities
    def log_q(self, last_offset, sample=rsample):
        """ Returns log probability under the variational distribution,
            conditioned on the offset of the last event (in seconds)
        """
        lqon = self.onset.log_q(last_offset, sample)
        lqoff = self.offset.log_q(self.onset.timepoint, sample)
        return lqon + lqoff

    def log_p(self, gap_metasource_parameters, duration_metasource_parameters):
        """ Returns the log probability under the prior """
        lpon = self.onset.log_p(gap_metasource_parameters).squeeze()
        lpoff = self.offset.log_p(duration_metasource_parameters).squeeze()
        return lpon + lpoff

    # Rendering
    def onset_ramp(self, timepoints, last_offset=None):
        """ Synthesizes a smooth onset ramp associated with this Event

        Parameters
        ----------
        timepoints: torch.Tensor
            timepoints defining where to generate the ramp,
            sampled at audio sampling rate
            see scene_timepoints in RenderingArrays in scene.py
        """
        return dr.differentiable_ramp(
            timepoints, self.onset.timepoint[:, None], is_onset=True
            )

    def offset_ramp(self, timepoints, last_offset=None):
        """ Synthesizes a smooth offset ramp associated with this Event """
        return dr.differentiable_ramp(
            timepoints, self.offset.timepoint[:, None], is_onset=False
            )

    def onset_step_ramp(self, timepoints, last_offset=None):
        """ Synthesizes an onset step function associated with this event """
        return dr.step_ramp(
            timepoints, self.onset.timepoint[:, None], is_onset=True
            )

    def offset_step_ramp(self, timepoints, last_offset=None):
        """ Synthesizes an offset step function associated with this event """
        return dr.step_ramp(
            timepoints, self.offset.timepoint[:, None], is_onset=False
            )

    def ramps(self, timepoints, last_offset=None):
        """ Synthesizes a smooth onset/offset window associated with this event """
        return self.onset_ramp(timepoints) * self.offset_ramp(timepoints)

    def step_ramps(self, timepoints, last_offset=None):
        """ Synthesizes a step-function onset/offset window associated with this event """
        return self.onset_step_ramp(timepoints) * self.offset_step_ramp(timepoints)


class Switch(SceneModule):
    """ Module for an onset or an offset """
    INITIAL_SIGMA = 0.005  # 5ms
    TYPICAL_SIGMA = 0.01  # 10ms

    def __init__(self, switch_type):
        super().__init__()
        self.switch_type = switch_type  # onset or offset?

    # Initalize priors and variational distributions
    def init_p(self, event_idx):
        """ Initialize prior over Switch """
        self.register_buffer("event_idx", torch.Tensor([event_idx]))
        self.set_limits()

    def set_limits(self):
        """ Defines the shortest and longest that an event can be """
        if self.event_idx == 0 and self.switch_type == "onset":
            self.register_buffer("low_limit", torch.zeros(1))
        elif self.switch_type == "offset":
            # Without low_limit, there's a discontinuity in the gradients
            # because tones can suddenly disappear.
            self.register_buffer(
                "low_limit", torch.Tensor([context.renderer["steps"]["t"]])
                )
            self.register_buffer("_high_limit", torch.Tensor([100.]))
        else:
            self.register_buffer("low_limit", torch.Tensor([1e-10]))
            self.register_buffer("_high_limit", torch.Tensor([100.]))

    @property
    def high_limit(self):
        if self.event_idx == 0 and self.switch_type == "onset":
            return torch.Tensor([self.scene_duration]).to(self.low_limit.device)
        else:
            return self._high_limit

    def init_q(self, hypothesis):
        """ Initialize variational distribution (truncated normal)
            over Switch variable

            This function defines:
            _mu: Tensor[float]
                Real-valued variable reflecting the absolute location of the Switch (aka. mu)
                Within the valid range, mu and _mu are essentially equivalent
            _sigma: Tensor[float]
                Real-valued variable reflecting the standard deviation
                parameter of the truncated normal
        """
        self._mu = nn.Parameter(torch.Tensor([hypothesis]))
        self._sigma = nn.Parameter(torch.Tensor([
            np_inv_softplus(self.INITIAL_SIGMA / self.TYPICAL_SIGMA)
            ]))

    @property
    def sigma(self):
        """ Returns standard deviation parameter by transforming _sigma into a positive number """
        return softplus(self._sigma) * self.TYPICAL_SIGMA

    def make_relative_and_bound(self, reference):
        """Returns relative value (gap or duration), above low limit (but not high limit)"""
        return softbound(
            self._mu - reference, self.low_limit, self.high_limit, 0.005
            )

    # Sample and compute log probability
    def log_q(self, last_timepoint, sample=rsample):
        """ Sample from guide and returns log probability of sample under
            the guide distribution

        Parameters
        ----------
        last_timepoint: Tensor[float]
            If switch_type = onset, last_timepoint is the offset (in sec)
                of the previous Event (or zero, if event_idx=0)
            If switch_type = offset, last_timepoint is the onset (in sec)
                of the same Event
        sample: function
            How to sample from the guide distribution (see util.py)
        """

        self.last_timepoint = last_timepoint

        _loc = (self.make_relative_and_bound(self.last_timepoint)).expand(context.batch_size)
        interval_dist = TruncatedNormal(
            _loc, self.sigma.expand(context.batch_size),
            self.low_limit, self.high_limit
            )

        if sample is not None:
            self.interval = sample(interval_dist)
            self.timepoint = self.interval + self.last_timepoint
        lq = interval_dist.log_prob(self.interval)

        return torch.squeeze(lq)

    def log_p(self, metasource_parameters):
        """ Returns log probability of sample under the prior.
        See Appendix A > Appendix A.2, Generative model: Sources and events > Equation A.9 and A.10

        Parameters
        ----------
        metasource_parameters: tuple[Tensor]
            Parameters of log normal distribution, see metasource_parameter
                method of Sequence
            tensor.shape = (batch_size, 1)
        """
        if self.switch_type == "onset" and self.event_idx == 0:
            prior = Uniform(self.low_limit, self.high_limit.to(self.low_limit.device))
        else:
            prior = LogNormal(*[hp.squeeze(1) for hp in metasource_parameters])
        lp = torch.squeeze(prior.log_prob(self.interval))
        return lp

    def sample(self, last_timepoint, metasource_parameters):
        """ Sample from prior """
        if hasattr(context, "background"):
            if self.switch_type == "onset":
                self.interval = torch.zeros([context.batch_size])
            elif self.switch_type == "offset":
                self.interval = self.scene_duration*torch.ones([context.batch_size])
        elif self.switch_type == "onset" and self.event_idx == 0:
            prior = Uniform(self.low_limit, self.high_limit)
            self.interval = prior.rsample(sample_shape=torch.Size([context.batch_size]))[:, 0]
        else:
            prior = LogNormal(*metasource_parameters)
            self.interval = prior.rsample().clamp(min=self.low_limit.item(), max=self.high_limit.item())
        self.timepoint = (self.interval + last_timepoint)
        return
