import torch
import torch.nn as nn
import numpy as np
import model.gp_hyperparameters as gp_hyperparameters
from model.feature import Feature
from util.sample import rsample

class GP(nn.Module):
    """ All gaussian process latent variables - mean and kernel source parameters + trajectory  """

    def __init__(self, hyperpriors, gp_type):
        super().__init__()
        # - Set any constants, etc.
        self.gp_type = gp_type

        mean_type = gp_hyperparameters.mean(hyperpriors["mu"])
        self.non_stationary, _, covar_type = gp_hyperparameters.kernel(hyperpriors["kernel"], gp_type)

        self.mean_module = mean_type()
        self.covar_module = covar_type(gp_type, hyperpriors["kernel"])

    @classmethod
    def hypothesize(cls, hypothesis, hyperpriors, gp_type, learn_hp=False):
        """
        Creates a new instance of GP where variational distribution is initialized based on a hypothesis

        Parameters
        ----------
        hypothesis: dict
            Defines the variational distribution via a hypothesis for continuous latent variables
            Format: {"events": List[Dict], "features":Dict{Dict}}
                events dict: {"onset": float, "offset": float}
                features dict: {gp_type: {"x": list, "y": list}, ... }
            For features dict, see use of "x_events" and "y_events" keys in the case that spectrum is non-stationary
        hyperprior: dict 
            Defines the model via parameters of hyperpriors
            Format: {"mu": float, "kernel": dict}
        gp_type: str
            Options: "amplitude", "spectrum", or "f0"
        learn_hp: bool
            Whether we are learning hyperpriors (True) or using them as fixed (False)
        """

        # Instantiate a module
        self = cls(hyperpriors, gp_type)        

        ### Initialize p and q for continuous latent variables at this level
        # Mean and Kernel
        self.init_p(hyperpriors, learn_hp=learn_hp)
        self.init_q(hypothesis)

        # Variational distribution over y 
        self.feature = Feature.hypothesize(hypothesis, hyperpriors, gp_type, learn_hp=learn_hp)

        return self

    @classmethod
    def sample(cls, hyperpriors, gp_type, events, r):
        """
        Creates a new instance of GP by sampling a scene description

        Parameters
        ----------
        hyperprior: dict 
            Defines the model via parameters of hyperpriors
            Format: {"mu": float, "kernel": dict}
        gp_type: str
            "amplitude", "spectrum" or "f0"
        events: List[Event]
            i.e., sequence.events
        r: RenderingArrays
            see scene.py    
        """

        # Instantiate a module
        #     Sets prior over discrete latent variables
        self = cls(hyperpriors, gp_type)

        ### Initialize p for continuous latent variables at this level, then sample
        # Mean and Kernel
        self.init_p(hyperpriors)
        self.mean_module.sample()
        self.covar_module.sample()

        # sample the trajectory itself
        self.feature = Feature.sample(hyperpriors, gp_type, self.mean_module, self.covar_module, events, r)

        return self

    def update(self, event_proposal, hyperpriors, updated_events, config):
        """Updates the variational distribution when a new event is added to a source"""
        if "spectrum" not in self.gp_type:
            #Update variational distribution
            inducing_points, inducing_values = self.feature.update(event_proposal)
            x = []; y = []; #c = [];
            for event_idx, event in enumerate(updated_events):
                on = np.full(inducing_points[:, 0].shape, event["onset"]) 
                off = np.full(inducing_points[:, 0].shape, event["offset"])
                b = (on < inducing_points[:, 0]) * (inducing_points[:, 0] < off)
                if inducing_points.shape[1] == 2:
                    b *= inducing_points[:, 1] == (event_idx + 1)
                    #c.append(inducing_points[b,1])
                x.append(inducing_points[b, 0]); y.append(inducing_values[b]); 

        if "spectrum" in self.gp_type and self.non_stationary:
            #Update variational distribution
            inducing_points, inducing_values = self.feature.update(event_proposal)
            x = []; y = []; #c = [];
            for event_idx, event in enumerate(updated_events):
                x.append(inducing_points[:, 0]); y.append(inducing_values[:]); 

        if config["heuristics"]["sequential"]["update_gp_hyperpriors"]:
            # update all hyperpriors with the inclusion of a new event
            self.init_q({"features":{self.gp_type:{"x":np.concatenate(x),"y":np.concatenate(y)}}})

    def init_p(self, hyperpriors, learn_hp=False):
        """ Initialize prior distribution over mean and covariance source parameters """
        self.mean_module.init_p(hyperpriors["mu"], learn_hp=learn_hp)
        self.covar_module.init_p(hyperpriors["kernel"], learn_hp=learn_hp)

    def init_q(self, hypothesis):
        """ Initialize variational distribution over mean and covariance source parameters """
        self.mean_module.init_q(hypothesis["features"][self.gp_type])
        self.covar_module.init_q(hypothesis["features"][self.gp_type])

    def log_q_source(self, sample=rsample):
        """ Sample mean and covariance source parameters from the variational distribution and return their log probability under this distribution """
        lqm = self.mean_module.log_q(sample); 
        lqc = self.covar_module.log_q(sample); 
        return torch.squeeze(lqm + lqc, dim=1) if (torch.is_tensor(lqm) or torch.is_tensor(lqc)) else lqm + lqc

    def log_q(self, events, r, sample=rsample):
        """ Sample the gaussian process from the variational distribution and return its log probability under this distribution

        Parameters
        ----------
        events: List[Event]
            i.e., sequence.events
        r: RenderingArrays
            see scene.py

        Returns
        -------
        lqs + lqy, Tensor[float]
            Log probability of the sampled gaussian process and its source parameters under the variational distribution
            shape: (batch_size,)
        """
        lqs = self.log_q_source(sample)
        self.feature.set_x(events, r)
        lqy = self.feature.log_q(self.mean_module, self.covar_module, sample=sample)
        return lqs + lqy

    def log_p_source(self):
        """ Return the log probability of the sampled mean and covariance source parameters under the prior, shape (batch_size,); see Appendix A Table 2 """
        lpm = self.mean_module.log_p()
        lpc = self.covar_module.log_p()
        return torch.squeeze(lpm + lpc, dim=1) if (torch.is_tensor(lpm) or torch.is_tensor(lpc)) else lpm + lpc

    def log_p(self):
        """ Return the log probability of the gaussian process under the prior, shape: (batch_size,) """
        lps = self.log_p_source()
        lpy = self.feature.log_p(self.mean_module, self.covar_module)
        return lps + lpy