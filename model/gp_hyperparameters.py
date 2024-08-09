import math
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.uniform import Uniform
from model.truncated_normal import TruncatedNormal
from torch.distributions.normal import Normal

import gpytorch
from gpytorch.means.mean import Mean
from gpytorch.kernels.kernel import Kernel
from gpytorch.utils.broadcasting import _mul_broadcast_shape 

from util.diff import np_inv_softplus, softplus, softbound
from util.sample import rsample
from util.context import context

################
# Setup functions for GP class in gaussian_processes.py
################


def mean(hyperpriors):
    """ Decide which mean module to use based on hyperpriors config """
    constant_mean = (hyperpriors['dist'] == 'constant')
    if constant_mean:
        mean_type = ZeroMean
    else:
        mean_type = SampledConstantMean
    return mean_type


def kernel(hyperpriors, gp_type):
    """ Decide which kernel to use based on hyperpriors config """

    # epsilon for stability during cholesky decomposition
    cholesky_stability = context.renderer["cholesky_stability"][gp_type]["basic"]
    if context.scene.scene_duration > context.renderer["cholesky_stability"]["long_duration"]:
        cholesky_stability += context.renderer["cholesky_stability"][gp_type]["add_to_long_sounds"]

    non_stationary = ("sigma_within" in hyperpriors["parametrization"] or "sigma_residual" in hyperpriors["parametrization"])
    if non_stationary:
        if gp_type == "spectrum":
            covar_type = ScaleMultispectrumSquaredExponential
        elif hyperpriors["type"] == "SE":
            covar_type = ScaleTemporalSquaredExponential
    else:
        if hyperpriors["type"] == "OU":
            covar_type = Matern
        elif hyperpriors["type"] == "SE":
            covar_type = ScaleSquaredExponential

    print("Creating covariance kernel of type", covar_type.__name__)
    return non_stationary, cholesky_stability, covar_type

##############################
#
# Means
#
##############################


class ZeroMean(gpytorch.means.zero_mean.ZeroMean):
    """ Constant zero mean """

    def __init__(self):
        #GPyTorch class simply extended with necessary methods
        super().__init__()

    def init_p(self, hyperprior, learn_hp=False):
        pass

    def init_q(self, init):
        pass

    def log_q(self, sample=rsample):
        return 0

    def log_p(self):
        return 0

    def sample(self):
        pass


class SampledConstantMean(Mean):
    """ Variational distribution over the mean of a gaussian process, with a uniform prior. """

    def __init__(self):
        super().__init__()

    def init_q(self, hypothesis):
        """ Initialize variational distribution based on hypothesis

        Parameters
        ----------
        hypothesis: dict[str,list[float]]
            hypothesis for feature trajectory for this event, format: {"x":list[float], "y":list[float]}
            x defines the time or frequency points where the hypothesis is defined, y defines the feature values
        """
        equally_spaced_xs = np.linspace(hypothesis["x"][0], hypothesis["x"][-1], num=100)
        equally_spaced_ys = np.interp(equally_spaced_xs, hypothesis["x"], hypothesis["y"])
        m = np.mean(equally_spaced_ys)
        self._mu_mu = nn.Parameter(torch.Tensor([m]))
        s = max(1.0, np.std(equally_spaced_ys))
        self._mu_sigma = nn.Parameter(torch.Tensor([np_inv_softplus(s)]))

    def init_p(self,hyperprior, learn_hp=None):
        """ Initialize the uniform prior distribution """
        self.register_buffer("mu_hp", torch.Tensor(hyperprior["args"]))
        self.register_buffer("mu_log_prior", -torch.log(torch.Tensor([self.mu_hp[1] - self.mu_hp[0]])))

    def log_q(self, sample=rsample):
        """ Sample mean from variational distribution and return log probability under this distribution, shape: (batch_size, 1) """
        _loc = softbound(self._mu_mu, self.mu_hp[0], self.mu_hp[1], (self.mu_hp[1]-self.mu_hp[0])/100)
        mu_dist = TruncatedNormal(_loc, 1e-5 + softplus(self._mu_sigma), self.mu_hp[0], self.mu_hp[1])
        if sample:
            self.mu = sample(mu_dist, sample_shape=torch.Size([context.batch_size]))
        lq = mu_dist.log_prob(self.mu)
        return lq

    def log_p(self):
        """ Return log probability of sampled mean under prior, shape: (batch_size, 1)"""
        return self.mu_log_prior[None, :].expand(*torch.Size([context.batch_size]), 1)

    def sample(self):
        """ Sample mean from prior distribution """
        self.mu = Uniform(*self.mu_hp).rsample(torch.Size([context.batch_size]))[:,None]
        return 

    def forward(self, input):
        """ Return sampled mean of appropriate shape """
        if input.shape[:-2] == torch.Size([context.batch_size]):
            return self.mu.expand(input.shape[:-1])
        else:
            return self.mu.expand(_mul_broadcast_shape(input.shape[:-1], self.mu.shape))

##############################
#
# Kernel hyperparameters
#
##############################


def initialize_hyperparameter_from_hypothesis(hypothesis, hp_name, gp_type, learn_hp=False):
    """ Returns an initialization for the hyperparameters based on a hypothesis. 

        Parameters
        ----------
        hypothesis (dict)
            hypothesis for feature trajectory for this event, format: {"x":list[float], "y":list[float]}
        hp_name (str)
            name of source parameter, e.g., "scale", "sigma"
        gp_type: str
            options: "amplitude", "spectrum" or "f0"
        learn_hp (bool)

        Returns
        -------
        m: float
            initial mean of variational distribution
        s: float
            initial standard deviation of variational distribution
    """

    if hp_name in context.hypothesis.get('initialization', {}).get(gp_type, {}):
        m = context.hypothesis['initialization'][gp_type][hp_name]["mu"]
        s = context.hypothesis['initialization'][gp_type][hp_name].get("sigma", 1.0)
        print(f"(Initializing hyperparameter {gp_type} {hp_name} at {m} ± {s})")
        return m, s

    if hp_name == "scale":
        if np.all(hypothesis["y"] == hypothesis["y"][0]):
            # Constant hypothesis
            m = 1.0
        else:
            # Scale has to be on the order of the inducing points for those to work
            sh = np.sort(hypothesis["x"])
            mean_distance_between_inducers = np.mean(sh[1:] - sh[:-1])
            m = mean_distance_between_inducers*2
            if gp_type != "spectrum":
                s = 0.1
                return m, s
    elif hp_name == "sigma":
        equally_spaced_xs = np.linspace(hypothesis["x"][0], hypothesis["x"][-1], num=100)
        equally_spaced_ys = np.interp(equally_spaced_xs, hypothesis["x"], hypothesis["y"])
        m = max(np.std(equally_spaced_ys),1.0)
    elif hp_name == "sigma_within":
        m = 1.0
    elif hp_name == "sigma_residual":
        m = 1.0
    else:
        raise NotImplementedError(f"Don't know how to intialize hyperparameter {gp_type} {hp_name}")

    s = 1.0
    return m, s


class ConstantHyperparameter(nn.Module):
    """ Hyperparameter that is a fixed value, not a random variable """

    def __init__(self, hp_name, gp_type):
        super().__init__()
        self.hp_name = hp_name
        self.gp_type = gp_type

    def init_q(self, hypothesis):
        pass

    def init_bounds(self, hyperprior):
        """ Initialize the bounds """
        if "bounds" in hyperprior.keys(): 
            if len(hyperprior["bounds"]) > 0:
                # Bounds should be given in linear scale, and this will transform with InvSoftplus.
                self.register_buffer("bounds", torch.Tensor(np.array(hyperprior["bounds"])))
            else:
                self.bounds = None
        else:
            self.bounds = None

    def init_p(self, hyperprior, learn_hp=False):
        if learn_hp:
            self.learn_hp = True
            if hyperprior.get("learn", True):
                self.hp = hyperprior["args"] #top level nn.Parameter from Scenes (see scene.py)
            else:
                self.learn_hp = False
                self.register_buffer("hp", torch.Tensor(hyperprior["args"]))
        else:
            self.learn_hp = False
            self.register_buffer("hp", torch.Tensor(hyperprior["args"]))
        self.init_bounds(hyperprior)
        return

    @property
    def mu(self):
        return self.hp[0]

    def log_q(self, sample=rsample):
        if sample:
            if self.bounds is None:
                self.val = self.mu.expand([context.batch_size,1])
            else:
                if len(self.bounds) == 2:
                    self.val = softbound(self.mu.expand([context.batch_size,1]), self.bounds[0], self.bounds[1], (self.bounds[1]-self.bounds[0])/100)
                elif len(self.bounds) == 1:
                    self.val = self.bounds[None, :].expand(*torch.Size([context.batch_size]), 1)
        return 0

    def log_p(self):
        return 0

    def sample(self):
        self.log_q()

    def forward(self):
        return self.val


class InvSoftplusNormalHyperparameter(nn.Module):
    """ Hyperparameter with InvSoftplusNormal prior. Softplus is more stable than Exp for inference. See Appendix A Table 2 """

    def __init__(self, hp_name, gp_type):
        super().__init__()
        self.hp_name = hp_name
        self.gp_type = gp_type

    def init_q(self, hypothesis):
        """ Initialize the mean and sigma of the variational distribution """
        # `hypothesis` should be given in linear scale, and this will transform with InvSoftplus
        m, s = initialize_hyperparameter_from_hypothesis(hypothesis, self.hp_name, self.gp_type)
        self.mu = nn.Parameter( torch.Tensor([np_inv_softplus(m)]) ) 
        self.sigma = nn.Parameter( torch.Tensor([np_inv_softplus(s)]) ) 

    def init_bounds(self, hyperprior):
        """ Initialize bounds of truncated distribution if applicable """
        if "bounds" in hyperprior.keys():
            # `bounds` should be given in linear scale, and this will transform with InvSoftplus. 
            if len(hyperprior["bounds"]) > 0:
                #If self.bounds = torch.Tensor([x]) (i.e., len=1), then treat this hyperprior like a constant
                #Otherwise treat as the bounds of a truncated normal
                self.register_buffer("bounds", torch.Tensor(np_inv_softplus(1e-10 + np.array(hyperprior["bounds"]))))
            else:
                self.bounds = None
        else:
            self.bounds = None

    def init_p(self, hyperprior, learn_hp=False):
        """ Initialize the prior """ 
        # `hyperprior`` should use values on linear scale
        if learn_hp:
            self.learn_hp = True
            if hyperprior.get("learn", True):
                self.hp = hyperprior["args"] #top level nn.Parameter from Scenes (see scene.py)
            else:
                self.learn_hp = False
                self.register_buffer("hp", torch.Tensor(hyperprior["args"]))
        else:
            self.learn_hp = False
            self.register_buffer("hp", torch.Tensor(hyperprior["args"]))
        self.init_bounds(hyperprior)
        return

    def log_q(self, sample=rsample):
        """ Sample from the variational distribution """
        if self.bounds is None:
            log_dist = Normal(self.mu, 1e-5 + softplus(self.sigma))
            if sample:
                self.inv_softplus_val = sample(log_dist, sample_shape=torch.Size([context.batch_size]))
            lq = log_dist.log_prob(self.inv_softplus_val)
        else:
            if len(self.bounds) == 2:
                _loc = softbound(self.mu, self.bounds[0], self.bounds[1], (self.bounds[1]-self.bounds[0])/100)
                log_dist = TruncatedNormal(_loc, 1e-5 + softplus(self.sigma), self.bounds[0], self.bounds[1])
                if sample:
                    self.inv_softplus_val = sample(log_dist, sample_shape=torch.Size([context.batch_size]))
                lq = log_dist.log_prob(self.inv_softplus_val)
            elif len(self.bounds) == 1:
                self.inv_softplus_val = self.bounds[None, :].expand(*torch.Size([context.batch_size]), 1)
                lq = 0
        return lq

    def log_p(self):
        """ Score the sample under the prior """
        if self.bounds is None:
            lp = Normal(self.hp[0], 1e-5 + softplus(self.hp[1])).log_prob(self.inv_softplus_val)
        else:
            if len(self.bounds) == 2:
                loc = softbound(self.hp[0], self.bounds[0], self.bounds[1], (self.bounds[1]-self.bounds[0])/100)
                lp = TruncatedNormal(loc, 1e-5 + softplus(self.hp[1]), self.bounds[0], self.bounds[1]).log_prob(self.inv_softplus_val)
            elif len(self.bounds) == 1:
                lp = 0
        return lp

    def sample(self):
        """ Sample a value from the prior """
        if self.bounds is None:
            self.inv_softplus_val = Normal(self.hp[0], 1e-5 + softplus(self.hp[1])).rsample(sample_shape=torch.Size([context.batch_size]))[:,None]
        else:
            if len(self.bounds) == 2:
                loc = softbound(self.hp[0], self.bounds[0], self.bounds[1], (self.bounds[1]-self.bounds[0])/100)
                self.inv_softplus_val = TruncatedNormal(loc, 1e-5 + softplus(self.hp[1]), self.bounds[0], self.bounds[1]).rsample(sample_shape=[context.batch_size])[:,None]
            elif len(self.bounds) == 1:
                self.inv_softplus_val = self.bounds[None, :].expand(*torch.Size([context.batch_size]), 1)

    def forward(self):
        """ transform into linear scale """ 
        return softplus(self.inv_softplus_val) + 1e-10 #avoid div by zero


class UniformHyperparameter(nn.Module):
    """ Hyperparameter with uniform prior. """

    def __init__(self, hp_name, gp_type):
        super().__init__()
        self.hp_name = hp_name
        self.gp_type = gp_type

    def init_q(self, hypothesis):
        """ Initialize q distribution """
        # `hypothesis` should be given in linear scale 
        m, s = initialize_hyperparameter_from_hypothesis(hypothesis, self.hp_name, self.gp_type)
        self.mu = nn.Parameter( torch.Tensor([m]) ) 
        self.sigma = nn.Parameter( torch.Tensor([s]) ) 

    def init_bounds(self, hyperprior):
        """ Initialize bounds of uniform prior """
        # `bounds`` should use values on a linear scale.
        self.register_buffer("bounds", self.hp)

    def init_p(self, hyperprior, learn_hp=False):
        """ Initialize prior """
        # `hyperprior` should use values on a linear scale.
        self.learn_hp=False
        self.register_buffer("hp", torch.Tensor(hyperprior["args"]))
        self.init_bounds(hyperprior)
        self.register_buffer("mu_log_prior", -torch.log(torch.Tensor([self.hp[1] - self.hp[0]])))
        return

    def log_q(self, sample=rsample):
        """ sample from variational distribution """
        _loc = softbound(self.mu, self.bounds[0], self.bounds[1], (self.bounds[1]-self.bounds[0])/100)
        log_dist = TruncatedNormal(_loc, 1e-5 + softplus(self.sigma), self.bounds[0], self.bounds[1])
        if sample:
            self.val = sample(log_dist, sample_shape=torch.Size([context.batch_size]))
        lq = log_dist.log_prob(self.val)
        return lq

    def log_p(self):
        """ score sample under prior uniform distribution """
        return self.mu_log_prior[None,:].expand(*torch.Size([context.batch_size]), 1)

    def sample(self):
        """ sample from uniform distribution """
        self.val = Uniform(self.hp[0], self.hp[1]).rsample(sample_shape=torch.Size([context.batch_size]))[:,None]

    def forward(self):
        return self.val + 1e-10 #avoid div by zero


class LogUniformHyperparameter(UniformHyperparameter):
    """ Sample hyperparameter from LogUniform distribution. No q distribution, used only for sampling. Used for domain randomization."""
    
    def __init__(self,hp_name, gp_type):
        super().__init__(hp_name, gp_type)

    def init_q(self, hypothesis):
        """ Initialize variational distribution with mean on log scale """ 
        # `hypothesis`` should be given in linear scale
        m, s = initialize_hyperparameter_from_hypothesis(hypothesis, self.hp_name, self.gp_type)
        self.mu = nn.Parameter( torch.Tensor(np.log([m])) ) 
        self.sigma = nn.Parameter( torch.Tensor([s]) ) 

    def init_p(self, hyperprior, learn_hp=False):
        """ Initialize prior distribution to be log uniform """
        # `hyperprior` should use linear-scale values. They will be log transformed here.
        self.learn_hp=False
        self.register_buffer("hp", torch.log(torch.Tensor(hyperprior["args"])))
        self.init_bounds(hyperprior)
        self.register_buffer("mu_log_prior", -torch.log(torch.Tensor([self.hp[1] - self.hp[0]])))
        return

    def forward(self):
        """ exponentiate sampled value """
        return torch.exp(self.val) + 1e-10

##############################
#
# Kernels
#
##############################


class BaseSampledKernel(Kernel):
    """ Method that combines several hyperparameters and implements a covariance kernel function, plus interfaces with gpytorch
        Note: gpytorch Kernel has a method called hyperparameters so make sure not to use that as an attribute name 
    """

    def __init__(self, hyperparams, gp_type, kernel_hyperpriors):
        super().__init__()
        ##Defining batch_shape for sampling
        hyperparameter_dict = {}
        for hp in hyperparams:
            if kernel_hyperpriors[hp]["dist"] == "inv_softplus_normal":
                hyperparameter_dict[hp] = InvSoftplusNormalHyperparameter(hp, gp_type)
            elif kernel_hyperpriors[hp]["dist"] == "uniform":
                hyperparameter_dict[hp] = UniformHyperparameter(hp, gp_type)
            elif kernel_hyperpriors[hp]["dist"] == "log_uniform":
                hyperparameter_dict[hp] = LogUniformHyperparameter(hp, gp_type)
            elif kernel_hyperpriors[hp]["dist"] == "constant":
                hyperparameter_dict[hp] = ConstantHyperparameter(hp, gp_type)
            else:
                raise Exception("Distribution {} for hyperparameter {} not implemented. Choose `inv_softplus_normal`, `uniform` or `log_uniform`.".format(kernel_hyperpriors[hp]["dist"], hp))
        self.hyperparams = nn.ModuleDict(hyperparameter_dict)
    
    def init_q(self, hypothesis):
        """ Initialize  variational distribution for each source parameter in this kernel """
        for k, hp_module in self.hyperparams.items():
            hp_module.init_q(hypothesis)

    def init_p(self, hyperprior, learn_hp=False):
        """ Initialize prior distribution for each source parameter in this kernel """
        for k, hp_module in self.hyperparams.items():
            hp_module.init_p(hyperprior[k], learn_hp=learn_hp) 

    def log_q(self, sample=rsample):
        """ Sample each source parameter in this kernel from the variational distribution and return log probability under this distribution """
        lq = 0
        for k, hp_module in self.hyperparams.items():
            lq += hp_module.log_q(sample)
        return lq

    def log_p(self):
        """ Return the log probability of each sampled source parameter in this kernel under the prior """
        lp = 0
        for k, hp_module in self.hyperparams.items():
            lp += hp_module.log_p()
        return lp

    def sample(self):
        """ Sample each source parameter in this kernel from the prior """
        for k, hp_module in self.hyperparams.items():
            hp_module.sample()

    def forward(self):
        pass


class ScaleSquaredExponential(BaseSampledKernel):
    """ Classic squared exponential kernel. Appendix A Eqn 15 """

    def __init__(self, gp_type, kernel_hyperparameters):
        super().__init__(["scale", "sigma"], gp_type, kernel_hyperparameters)

    @property
    def scale(self):
        return self.hyperparams["scale"]() 

    @property
    def sigma(self):
        return self.hyperparams["sigma"]() 

    def forward(self, x1, x2, diag=False, **params):
        """ Return prior covariance matrix for points at x1 and x2 """
        x1_ = x1.div(self.scale[:,None,:])
        x2_ = x2.div(self.scale[:,None,:])
        C = self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=gpytorch.kernels.rbf_kernel.postprocess_rbf, postprocess=True, **params
        )
        if diag:
            C = C.mul(self.sigma**2)
        else:
            C = C.mul(self.sigma[:,None,:]**2)
        return C


class Matern(BaseSampledKernel):
    """ Matern kernel with nu=0.5 as a default, aka Ornstein Uhlenbeck (OU) kernel. Appendix A Eqn. 20 """

    def __init__(self, gp_type, kernel_hyperparameters, nu=0.5):
        super().__init__(["scale", "sigma"], gp_type, kernel_hyperparameters)
        self.nu = nu

    @property
    def scale(self):
        return self.hyperparams["scale"]() 

    @property
    def sigma(self):
        return self.hyperparams["sigma"]() 

    def forward(self, x1, x2, diag=False, **params):
        """ Return prior covariance matrix for points at x1 and x2 """

        x1_ = x1.div(self.scale[:,None,:])
        x2_ = x2.div(self.scale[:,None,:])
        distance = self.covar_dist(
            x1_, x2_, square_dist=False, diag=diag, postprocess=False, **params
        )
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
        
        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        
        C = constant_component * exp_component
        if diag:
            C = C.mul(self.sigma**2)
        else:
            C = C.mul(self.sigma[:,None,:]**2)
        
        return C


class ScaleTemporalSquaredExponential(BaseSampledKernel):
    """ Non-stationary squared exponential kernel, with non-stationarity between events. Appendix A Eqn. 15 """

    def __init__(self, gp_type, kernel_hyperparameters):
        super().__init__(["scale", "sigma", "sigma_within"], gp_type, kernel_hyperparameters)
        self.gp_type = gp_type

    @property
    def scale(self):
        return self.hyperparams["scale"]()

    @property
    def sigma(self):
        return self.hyperparams["sigma"]() 

    @property
    def sigma_within(self):
        return self.hyperparams["sigma_within"]() 

    def forward(self, x1, x2, diag=False, **params):
        """ Return prior covariance matrix for points at x1 and x2

        This is called in multiple ways.

        - Covariance between gp points (for sampling and also scoring)
            section_mask: are these two timepoints either BOTH in the same element or BOTH silence
            silence_mask: are these two timepoints BOTH non-silence

        - Cross-covariance between inducers and gp points (for sampling)
            - each inducing is a (t, e) pair, where:
                - e is an element index (never silence)
                - element e DOES NOT need to contain timepoint t
            - each gp point is a (t, e) pair, where:
                - e is an element index OR SILENCE
                - element e contains timepoint t

            section_mask: is the gp timepoint located within the element that the inducing point represents

            1  1 111   1 1     1  --- inducers for event 1
                |-----------|    --- event 1
            x=0      x=1         --- gp timepoint

            silence mask: is the gp timepoint non-silence


        For the KL divergence, we only want to score gp timepoints that are non-silence
        So we want to:
            - remove ALL covariance between silence gp timepoints and any other points
            - variance for a silence point must be the same in log p and in log q

        """
        x1_ = x1[:,:,None,0].div(self.scale[:,None,:])
        x2_ = x2[:,:,None,0].div(self.scale[:,None,:])
        C = self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=gpytorch.kernels.rbf_kernel.postprocess_rbf, postprocess=True, **params
        )
        if diag:
            C = C.mul(self.sigma**2)
        else:
            C = C.mul(self.sigma[:,None,:]**2)
        section_mask = (x1[:, :, None, 1] == x2[:, None, :, 1]) #batch, len(xi1), len(xi2)
        silence_mask = (x1[:, :, None, 1] * x2[:, None, :, 1]) > 0 #silence will always be zero.
        elems_mask = section_mask*silence_mask*1.0
        if diag:
            elems_mask = torch.diagonal(elems_mask,dim1=-2,dim2=-1)
            silence_mask = torch.diagonal(silence_mask,dim1=-2,dim2=-1)
            s2 = self.sigma_within**2
        else:
            s2 = (self.sigma_within**2)[:,:,None]

        return C*silence_mask + s2*elems_mask


class ScaleMultispectrumSquaredExponential(BaseSampledKernel):
    """ Non-stationary squared exponential kernel, with non-stationarity for spectrum variable between events. Not used in paper. """

    def __init__(self, gp_type, kernel_hyperparameters):
        super().__init__(["scale", "sigma", "sigma_residual"], gp_type, kernel_hyperparameters)
        self.gp_type = gp_type

    @property
    def scale(self):
        return self.hyperparams["scale"]()

    @property
    def sigma(self):
        return self.hyperparams["sigma"]() 

    @property
    def sigma_residual(self):
        return self.hyperparams["sigma_residual"]() 

    def forward(self, x1, x2, diag=False, **params):
        """ Return prior covariance matrix for points at x1 and x2 """
        x1_ = x1[:,:,None,0].div(self.scale[:,None,:])
        x2_ = x2[:,:,None,0].div(self.scale[:,None,:])
        C = self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=gpytorch.kernels.rbf_kernel.postprocess_rbf, postprocess=True, **params
        )
        if diag:
            C = C.mul(self.sigma**2)
        else:
            C = C.mul(self.sigma[:,None,:]**2)
        silence_mask = (x1[:, :, None, 1] * x2[:, None, :, 1]) > 0 #silence will always be zero.
        
        if diag:
            silence_mask = torch.diagonal(silence_mask,dim1=-2,dim2=-1)
            s2 = self.sigma_residual**2
        else:
            s2 = (self.sigma_residual**2)[:, :, None]

        if x1.shape==x2.shape and (x1==x2).all():
            if diag:
                # Variance
                return C*silence_mask + s2*silence_mask
            else:
                # Covariance
                return C*silence_mask + s2*torch.eye(silence_mask.shape[1], device=silence_mask.device)
        else:
            # Cross covariance
            return C*silence_mask
