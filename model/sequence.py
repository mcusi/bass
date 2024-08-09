import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.geometric import Geometric
import numpy as np

from util.diff import softplus, np_inv_softplus
from util.sample import rsample
from util.context import context
from model.event import Event
from model.scene_module import SceneModule
import inference.sequential_heuristics


class Sequence(SceneModule):
    """ Contains a set of Events (which are actually only the timing of events, i.e., onset and offset)"""

    def __init__(self, source_priors):
        super().__init__()
        # Create modules for sequence priors
        if source_priors["gap"]["precision"]["dist"] == "constant" or source_priors["gap"]["mu"]["dist"] == "constant":
            self.gap = ConstantSequenceSourcePrior("gap")
        else:
            self.gap = SequenceSourcePrior("gap")
        if source_priors["duration"]["precision"]["dist"] == "constant" or source_priors["duration"]["mu"]["dist"] == "constant":
            self.duration = ConstantSequenceSourcePrior("duration")
        else:
            self.duration = SequenceSourcePrior("duration")
        # Set (fixed) prior over n_events
        self.register_buffer("n_events_probs", torch.Tensor([source_priors["n_events"]["args"]]))
        self.n_events_dist = Geometric

    @classmethod
    def hypothesize(cls, hypothesis, source_priors, learn_hp=False):
        """
        Creates a new instance of Sequence where the variational distribution is initialized based on a hypothesis
        
        Parameters
        ----------
        hypothesis: dict
            Defines the variational distribution, that is:
            - Settings for all discrete latent variables
            - A hypothesis for continuous latent variables (onset, offset)
        source_priors: dict
            Defines the model, i.e., the parameters of source priors e.g. concentration0, a0_f0
        """

        # Instantiate a module
        #     Sets prior over discrete latent variables
        self = cls(source_priors)

        # Set all discrete latent variables (#Events), based on hypothesis
        self.set_discrete(hypothesis["events"])
            
        # Initialize p and q for continuous latent variables at this level
        self.init_p(source_priors, learn_hp=learn_hp)
        self.gap.init_q(hypothesis["events"], source_priors["gap"]) 
        self.duration.init_q(hypothesis["events"], source_priors["duration"])

        # Instantiate all Event submodule with `hypothesize`: discrete, p, q.
        self.events = nn.ModuleList([Event.hypothesize(hypothesis["events"][event_idx], source_priors, event_idx) for event_idx in range(self.n_events)])        
        if hasattr(self.scene, "meta"):
            for event in self.events:
                self.scene.meta.add_event(event)
        return self

    @classmethod
    def sample(cls, source_priors):
        """
        Creates a new instance of Sequence by sampling
        
        Parameters
        ----------
        source_priors: dict
            Defines the model, i.e., the "meta-source" parameters of source priors e.g. concentration0, a0_f0
        """
        # Instantiate a module
        #     Sets prior over discrete latent variables
        self = cls(source_priors)

        # Sample all discrete latent variables (#Events)
        self.sample_discrete()
    
        # Initialize p for continuous latent variables at this level, then sample
        self.init_p(source_priors)
        self.gap.sample(); self.duration.sample()

        # Instantiate all submodules with `sample`: sample discrete, initialize p, sample from p
        self.events = nn.ModuleList()
        for event_idx in range(self.n_events):
            last_offset = 0 if (event_idx == 0) else self.events[event_idx - 1].offset.timepoint
            if last_offset > self.scene_duration:
                self.n_events = torch.Tensor([len(self.events)]).long().to(last_offset.device)
                break
            self.events.append(Event.sample({"gap": self.gap.metasource_parameters(), "duration": self.duration.metasource_parameters()}, event_idx, last_offset))

        return self

    def set_discrete(self, hypothesis):
        self.register_buffer("n_events", torch.Tensor([len(hypothesis)]).long())

    def sample_discrete(self):
        self.n_events = self.n_events_dist(self.n_events_probs).sample().long() + 1

    def init_p(self, source_priors, learn_hp=False):
        """Initalize prior for all continuous latent variables at this level"""
        self.gap.init_p(source_priors["gap"], learn_hp=learn_hp)
        self.duration.init_p(source_priors["duration"], learn_hp=learn_hp)
        pass

    def log_p(self):
        """ Return the log probability of latent variables under the prior """
        return self.log_p_discrete() + self.log_p_temporal()

    def log_p_discrete(self):
        """ Return the log probability of the number of events under the prior """
        discrete_p = self.n_events_dist(self.n_events_probs).log_prob(self.n_events - 1).repeat(context.batch_size)
        return discrete_p

    def log_p_temporal(self):
        """ Return the log probability of the rest or active durations of each scene, under the prior"""
        gap_p = self.gap.log_p()
        duration_p = self.duration.log_p()
        event_ps = torch.zeros(gap_p.shape,device=gap_p.device)
        for i in range(self.n_events):
            event_p = self.events[i].log_p(self.gap.metasource_parameters(), self.duration.metasource_parameters())
            event_ps += event_p
        return gap_p + duration_p + event_ps 

    def log_q(self, sample=rsample, **kwargs):
        """ Sample the onsets and offsets of each event in the sequence from the variational distribute, and return the log probability
        Use kwargs for learning source_priors, when the event timings are known
        """
        s = sample if sample is not None else kwargs["source_prior_sample"]
        gap_q = self.gap.log_q(s)
        duration_q = self.duration.log_q(s)
        event_qs = torch.zeros(duration_q.shape,device=gap_q.device)
        s = sample if sample is not None else kwargs["event_sample"]
        for i in range(self.n_events):
            last_offset = 0.0 if (i == 0) else self.events[i-1].offset.timepoint
            event_q = self.events[i].log_q(last_offset, s)
            event_qs += event_q
        if "detach_event_q" in kwargs.keys():
            if kwargs["detach_event_q"]:
                event_qs = event_qs.detach()
        return gap_q + duration_q + event_qs 

    # Inference methods
    # Sequential inference
    def update(self, event_proposal, source_priors):
        """ """
        # See check_fit logic below
        if event_proposal["onset"] <= self.events[-1].offset._mu.item():
            self.events[-1].offset.init_q(event_proposal["onset"] - 0.001)
        # Add new event
        new_event = Event.hypothesize(event_proposal, source_priors, self.n_events.item())
        self.events.append(new_event)
        self.scene.meta.add_event(new_event)
        self.n_events += 1
        # Update source_priors
        updated_events = [{"onset": event.onset._mu.item(), "offset": event.offset._mu.item()} for event in self.events]
        self.gap.init_q(updated_events, source_priors["gap"]) 
        self.duration.init_q(updated_events, source_priors["duration"])
        return new_event, updated_events

    def check_iou(self, event_proposal):
        """ For sequential inference, check if the IOU of new event proposal with all other events in this sequence meets the threshold """
        passes_iou_threshold = []
        for event in self.events:
            passes_iou_threshold.append(event.meta.compare(event_proposal["events"][0]["meta"]))
        return all(passes_iou_threshold)

    def check_fit(self, event_proposal):
        """ For sequential inference, check new event proposal against heuristics """
        return inference.sequential_heuristics.check_event_fit(self, event_proposal, context.heuristics)


class SequenceSourcePrior(nn.Module):
    """ Gamma-Normal source prior for log-normal distributed gaps and durations - see Appendix A Table 1 """
    def __init__(self, interval_type):
        super().__init__()
        self.interval_type = interval_type

    ### Initialize probability distributions
    def relative_timings(self, events):
        """ Return relative timings from event hypotheses, which are defined with onset and offset"""
        relative_events = []
        last_offset = 0
        for e in events:
            gap = e["onset"] - last_offset
            duration = e["offset"] - e["onset"]
            last_offset = e["offset"]
            if self.interval_type == "gap":
                relative_events.append(max(0,gap)) 
            elif self.interval_type == "duration":
                relative_events.append(max(0,duration))
        if self.interval_type == "gap":
            relative_events = relative_events[1:]
        return relative_events

    def conjugate_prior(self, intervals, source_priors):
        """ Compute conjugate prior from event intervals in hypothesis
        
        Parameters
        ----------
        intervals: list[float]
            List of durations based on event hypothesis, in seconds
        source_priors: dict[str, dict[str, float or Tensor]]
            Meta-source parameters for Gamma-Normal source_prior 
            Values may be Tensor if learning meta-source parameters, or floats if they are fixed
        """
        if torch.is_tensor(source_priors["mu"]["args"]):
            #If learning metasource_parameters, use the shared nn.Parameters
            concentration_0 = source_priors["precision"]["args"][0].item()
            rate_0 = source_priors["precision"]["args"][1].item() 
            mu_0 = source_priors["mu"]["args"][0].item()
            kappa_0 = source_priors["mu"]["args"][1].item()
        else:
            concentration_0, rate_0 = source_priors["precision"]["args"]
            mu_0, kappa_0 = source_priors["mu"]["args"]
        if len(intervals) > 0:
            # precision ~ Gamma(concentration_n, rate_n); sigma = 1/sqrt(precision); mu ~ normal(mu_n, sigma/kappa_n)         
            # Conjugate prior: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf Eq 85 - 89
            log_intervals = np.log(np.array(intervals) + 1e-5)
            mu_x = np.mean(log_intervals)
            n = len(log_intervals)
            mu_n = ( kappa_0*mu_0 + n*mu_x )/(kappa_0 + n)
            kappa_n = kappa_0 + n
            concentration_n = concentration_0 + 0.5*n 
            rate_n = rate_0 + 0.5*np.sum( (log_intervals - mu_x)**2 ) + 0.5*(kappa_0*n*(mu_x - mu_0)**2)/(kappa_0 + n)
            return concentration_n, rate_n, mu_n, kappa_n
        else:
            return concentration_0, rate_0, mu_0, kappa_0

    def init_q(self, hypothesis, source_priors):
        """ Initialize variational distribution """
        intervals = self.relative_timings(hypothesis)
        concentration, rate, mu, kappa = self.conjugate_prior(intervals, source_priors)
        self._precision_concentration = nn.Parameter(
            torch.Tensor([np_inv_softplus(concentration)])
            )
        self._precision_rate = nn.Parameter(
            torch.Tensor([np_inv_softplus(rate)])
            )
        self.mu_mean = nn.Parameter(torch.Tensor([mu]))
        self.mu_sigma = nn.Parameter(torch.Tensor([np_inv_softplus(kappa)]))

    def init_p(self, source_prior, learn_hp=False):
        """ Initialize prior distribution """       
        p = source_prior["precision"]["args"]
        m = source_prior["mu"]["args"]
        if learn_hp:
            if isinstance(p, list):
                self.precision_hp = nn.Parameter(torch.Tensor(p))
                self.mu_hp = nn.Parameter(torch.Tensor(m))
            elif (p.requires_grad and p.is_leaf): #check that it's Parameter
                self.precision_hp = p 
                self.mu_hp = m 
        else:
            self.register_buffer("precision_hp", torch.Tensor(p)) #a_0, b_0
            self.register_buffer("mu_hp", torch.Tensor(m))

    # Sample and compute log probabilities
    def log_q(self, sample=rsample):
        """ Sample precision and mean source parameters from the variational distribution and return the log probability """
        precision_dist = Gamma(1e-5 + softplus(self._precision_concentration), 1e-5 + softplus(self._precision_rate))
        if sample:
            self.precision = sample(precision_dist, sample_shape=torch.Size([context.batch_size]))
        lq = precision_dist.log_prob(self.precision)

        mu_dist = Normal(self.mu_mean, 1e-5 + softplus(self.mu_sigma))
        if sample:
            self.mu = sample(mu_dist, sample_shape=torch.Size([context.batch_size]))
        lq += mu_dist.log_prob(self.mu)    

        return torch.squeeze(lq)

    @property
    def sigma(self):
        return 1.0/(torch.sqrt(1e-5 + self.precision))

    def log_p(self):
        """ Return the log probability of the precision and mean source parameters under the prior"""
        lp = Gamma(*self.precision_hp).log_prob(self.precision)
        lp += Normal(self.mu_hp[0], self.sigma/self.mu_hp[1]).log_prob(self.mu)
        return torch.squeeze(lp)

    def sample(self):
        """ Sample the precision and mean source parameters from the prior """
        precision_prior = Gamma(*self.precision_hp)
        self.precision = precision_prior.rsample(sample_shape=torch.Size([context.batch_size]))
        mu_prior = Normal(self.mu_hp[0], self.sigma/self.mu_hp[1])
        self.mu = mu_prior.rsample()
        return

    def metasource_parameters(self):
        return self.mu, self.sigma


class ConstantSequenceSourcePrior(nn.Module):
    """ Lesion of full model to test necessity of source_priors """

    def __init__(self, interval_type):
        super().__init__()
        self.interval_type = interval_type

    def init_q(self, hypothesis, source_priors):   
        """ Set values for source parameters (precision_val and mu_val)
            to be the mean values of the source parameter distributions
        """     
        concentration_0, rate_0 = source_priors["precision"]["args"]
        mu_0, kappa_0 = source_priors["mu"]["args"]
        self._precision_concentration = torch.Tensor([np_inv_softplus(concentration_0)])
        self._precision_rate = torch.Tensor([np_inv_softplus(rate_0)])
        self.mu_mean = torch.Tensor([mu_0])
        self.mu_sigma = torch.Tensor([np_inv_softplus(kappa_0)])
        self.register_buffer(
            "precision_val",
            Gamma(
                1e-5 + softplus(self._precision_concentration),
                1e-5 + softplus(self._precision_rate)
            ).mean
            )
        self.register_buffer("mu_val", self.mu_mean)

    def init_p(self, source_prior, learn_hp=False):   
        p = source_prior["precision"]["args"]
        m = source_prior["mu"]["args"]
        if learn_hp:
            if isinstance(p, list):
                self.precision_hp = nn.Parameter(torch.Tensor(p))
                self.mu_hp = nn.Parameter(torch.Tensor(m))
            elif (p.requires_grad and p.is_leaf):  # check that it's Parameter
                self.precision_hp = p
                self.mu_hp = m
        else:
            self.register_buffer("precision_hp", torch.Tensor(p))  # a_0, b_0
            self.register_buffer("mu_hp", torch.Tensor(m))

    # Sample and compute log probabilities
    def log_q(self, sample=rsample):
        """ Set source parameters to the mean of
            the meta-source distributions
        """
        self.precision = self.precision_val.expand(context.batch_size)[:, None]
        self.mu = self.mu_val.expand(context.batch_size)[:, None]
        return 0*self.mu.squeeze()

    def sample(self):
        self.log_q()
        return

    @property
    def sigma(self):
        return 1.0/(torch.sqrt(1e-5 + self.precision))

    def log_p(self):
        return 0*self.mu.squeeze()

    def metasource_parameters(self):
        return self.mu, self.sigma
