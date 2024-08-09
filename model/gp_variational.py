import torch
import math

from gpytorch.variational import _NaturalVariationalDistribution, MeanFieldVariationalDistribution, UnwhitenedVariationalStrategy
from gpytorch.module import Module
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import (
    CholLazyTensor,
    DiagLazyTensor,
    PsdSumLazyTensor,
    RootLazyTensor,
    ZeroLazyTensor,
)
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.utils.memoize import add_to_cache

from util.context import context

#################
# Integrates bass code with gpytorch--
# specifically, implementing a mean field variational distribution
# that can be initialized and updated based on hypotheses.
#################

class InitializableMeanFieldVariationalDistribution(MeanFieldVariationalDistribution):
    """ Mean field approximation that can be initialized using a hypothesis """

    def __init__(self, init_inducing_values, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3, **kwargs):
        _NaturalVariationalDistribution.__init__(self, num_inducing_points=num_inducing_points, batch_shape=torch.Size([context.batch_size]), mean_init_std=mean_init_std)

        mean_init = torch.Tensor(init_inducing_values)
        if len(mean_init.shape) == 1:
            mean_init = mean_init[None, :]
        covar_init = mean_init_std*torch.ones(num_inducing_points)

        self.register_parameter(name="_variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="_variational_stddev", parameter=torch.nn.Parameter(covar_init))

    @property
    def variational_mean(self):
        return self._variational_mean.repeat(*torch.Size([context.batch_size]), 1)

    @property
    def variational_stddev(self):
        mask = torch.ones_like(self._variational_stddev)
        return self._variational_stddev.mul(mask).abs().clamp_min(1e-8).repeat(*torch.Size([context.batch_size]), 1)

    def initialize_variational_distribution(self, prior_dist):
        self._variational_mean.data.copy_(prior_dist.mean[0,:])
        self._variational_mean.data.add_(torch.randn_like(prior_dist.mean[0,:]), alpha=self.mean_init_std)
        self._variational_stddev.data.copy_(prior_dist.stddev[0,:])



class MyUnwhitenedVariationalStrategy(UnwhitenedVariationalStrategy):
    """ Integrating gpytorch unwhitened variational strategy with our codebase """
    
    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        # If our points equal the inducing points, we're done
        if torch.equal(x, inducing_points):
            if variational_inducing_covar is None:
                raise RuntimeError
            else:
                return MultivariateNormal(inducing_values, variational_inducing_covar)

        # Otherwise, we have to marginalize
        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs,**kwargs)
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        # Mean terms
        test_mean = full_mean[..., num_induc:]
        induc_mean = full_mean[..., :num_induc]
        mean_diff = (inducing_values - induc_mean).unsqueeze(-1)

        # Covariance terms
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute Cholesky factorization of inducing covariance matrix
        if settings.fast_computations.log_prob.off() or (num_induc <= settings.max_cholesky_size.value()):
            induc_induc_covar = CholLazyTensor(self._cholesky_factor(induc_induc_covar))

        # If we are making predictions and don't need variances, we can do things very quickly.
        if not self.training and settings.skip_posterior_variances.on():
            self._mean_cache = induc_induc_covar.inv_matmul(mean_diff).detach()
            predictive_mean = torch.add(
                test_mean, induc_data_covar.transpose(-2, -1).matmul(self._mean_cache).squeeze(-1)
            )
            predictive_covar = ZeroLazyTensor(test_mean.size(-1), test_mean.size(-1))
            return MultivariateNormal(predictive_mean, predictive_covar)

        # Expand everything to the right size
        shapes = [mean_diff.shape[:-1], induc_data_covar.shape[:-1], induc_induc_covar.shape[:-1]]
        if variational_inducing_covar is not None:
            root_variational_covar = variational_inducing_covar.root_decomposition().root.evaluate()
            shapes.append(root_variational_covar.shape[:-1])
        shape = _mul_broadcast_shape(*shapes)
        mean_diff = mean_diff.expand(*shape, mean_diff.size(-1))
        induc_data_covar = induc_data_covar.expand(*shape, induc_data_covar.size(-1))
        induc_induc_covar = induc_induc_covar.expand(*shape, induc_induc_covar.size(-1))
        if variational_inducing_covar is not None:
            root_variational_covar = root_variational_covar.expand(*shape, root_variational_covar.size(-1))

        # Cache the kernel matrix with the cached CG calls
        if self.training:
            prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
            add_to_cache(self, "prior_distribution_memo", prior_dist)

        # Compute predictive mean
        if variational_inducing_covar is None:
            left_tensors = mean_diff
        else:
            left_tensors = torch.cat([mean_diff, root_variational_covar], -1)
        inv_products = induc_induc_covar.inv_matmul(induc_data_covar, left_tensors.transpose(-1, -2))
        predictive_mean = torch.add(test_mean, inv_products[..., 0, :])

        # Compute covariance
        if self.training:
            interp_data_data_var, _ = induc_induc_covar.inv_quad_logdet(
                induc_data_covar, logdet=False, reduce_inv_quad=False
            )
            data_covariance = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
        else:
            neg_induc_data_data_covar = torch.matmul(
                induc_data_covar.transpose(-1, -2).mul(-1), induc_induc_covar.inv_matmul(induc_data_covar)
            )
            data_covariance = data_data_covar + neg_induc_data_data_covar
        predictive_covar = PsdSumLazyTensor(RootLazyTensor(inv_products[..., 1:, :].transpose(-1, -2)), data_covariance)

        # Done!
        return MultivariateNormal(predictive_mean, predictive_covar)

class EventUnwhitenedVariationalStrategy(MyUnwhitenedVariationalStrategy):
    """ Variational strategy that can be updated with multiple rounds of inference """

    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=True):
        Module.__init__(self)

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        self.lil = learn_inducing_locations
        if learn_inducing_locations:
            self.register_parameter(name="_inducing_points", parameter=torch.nn.Parameter(inducing_points[:, None, 0]))
            self.register_buffer("element_idxs", inducing_points[:, None, 1])
        else:
            self.register_buffer("_inducing_points", inducing_points[:, None, 0])
            self.register_buffer("element_idxs", inducing_points[:, None, 1])

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(1))

    @property
    def inducing_points(self):
        return torch.cat((self._inducing_points, self.element_idxs), dim=1)

    def update(self, updated_inducing_points, updated_inducing_values, updated_inducing_stds):

        vm_shape = self._variational_distribution._variational_mean.shape #Should be len(vm_shape) = 2
        n_new_inducing_points = updated_inducing_points.shape[0] - vm_shape[1]
        assert n_new_inducing_points>=0

        if len(updated_inducing_values.shape) == 1:
            updated_inducing_values = updated_inducing_values[None, :]
            updated_inducing_stds = updated_inducing_stds[None, :]
        self._variational_distribution._variational_mean = torch.nn.Parameter(torch.Tensor(updated_inducing_values))
        self._variational_distribution._variational_stddev = torch.nn.Parameter(torch.Tensor(updated_inducing_stds))
            
        if self.lil:
            self._inducing_points = torch.nn.Parameter(torch.Tensor(
                    updated_inducing_points[:, None, 0]
                )
            )
            self.register_buffer("element_idxs", torch.Tensor(updated_inducing_points[:, None, 1]))
        else:
            self.register_buffer("_inducing_points", torch.Tensor(updated_inducing_points[:, None, 0]))
            self.register_buffer("element_idxs", torch.Tensor(updated_inducing_points[:, None, 1]))