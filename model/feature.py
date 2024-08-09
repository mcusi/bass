import torch
import torch.nn as nn
import numpy as np

import gpytorch

from util.diff import softplus
from util.sample import rsample
from util.context import context
from model.scene_module import SceneModule
from model.gp_variational import InitializableMeanFieldVariationalDistribution, MyUnwhitenedVariationalStrategy, EventUnwhitenedVariationalStrategy
import model.gp_hyperparameters as gp_hyperparameters


class Feature(SceneModule):
    """ Latent variables for spectrum or time-trajectories (amplitude or F0),
        integrated with variational inducing point framework for inference
    """

    def __init__(self, hyperpriors, gp_type):
        super().__init__()
        # Set any constants, etc.
        # Set (fixed) prior over discrete latents

        self.gp_type = gp_type
        self.pad_after_active = 0.050
        self.default_inducer_pad = 0.020

        self.initial_inducing_std = 1e-3

        self.learn_inducing_locations = hyperpriors["learn_inducing_locations"]
        self.non_stationary, self.cholesky_stability, covar_type = \
            gp_hyperparameters.kernel(hyperpriors["kernel"], gp_type)

    @classmethod
    def hypothesize(cls, hypothesis, hyperpriors, gp_type, learn_hp=False):
        """
        Creates a new instance of Feature where variational distribution
        is initialized based on a hypothesis

        Parameters
        ----------
        hypothesis: dict
            Defines the variational distribution via a hypothesis
                for continuous latent variables
            Format: {"events": List[Dict], "features":Dict{Dict}}
                events dict: {"onset": float, "offset": float}
                features dict: {gp_type: {"x": list, "y": list}, ... }
            For features dict, see use of "x_events" and "y_events" keys in
                the case that spectrum is non-stationary
        hyperprior: dict
            Defines the model via parameters of hyperpriors
            Format: {"mu": float, "kernel": dict}
        gp_type: str
            Options: "amplitude", "spectrum", or "f0"
        learn_hp: bool
            Whether we are learning hyperpriors (True) or using them as fixed (False)
        """

        # Instantiate a module
        #     Sets prior over discrete latent variables
        self = cls(hyperpriors, gp_type)

        # Initialize p and q for continuous latent variables at this level
        # p(y_spectrum) is handled by the parent, no init_p; except to set epsilon
        # q(y_spectrum) = MeanFieldVariationalDistribution(...)
        self.init_p(hyperpriors, learn_hp=learn_hp)
        self.init_q(hypothesis)

        return self

    @classmethod
    def sample(cls, hyperpriors, gp_type, mean_module, covar_module, events, r):
        """
        Creates a new instance of Feature by sampling a scene description

        Parameters
        ----------
        hyperprior: dict 
            Defines the model via parameters of hyperpriors
            Format: {"mu": float, "kernel": dict}
        gp_type: str
            "amplitude", "spectrum" or "f0"
        mean_module: SampledConstantMean or ZeroMean
            see gp_hyperparameters.py
        covar_module: module with BaseSampledKernel as parent
            see `kernel` in gp_hyperparameters.py
        events: List[Event]
            i.e., sequence.events
        r: RenderingArrays
            see scene.py
        """

        # Instantiate module
        self = cls(hyperpriors, gp_type)

        # P already initialized in parent module, reflected in mean
        # and covar_module; except to set epsilon
        # set_x: creates self.gp_x, self.event_masks
        self.init_p(hyperpriors)
        self.set_x(events, r)

        # Sample from p(y)
        prior = self.forward(
            self.gp_x, mean_module=mean_module, covar_module=covar_module
            )
        self.y = prior.rsample()

        # Prepare y for being rendered      
        if self.non_stationary and self.gp_type != "spectrum":
            self.y_render = self.pad_after_sample(self.y, self.event_masks)

        return self

    # Functions required for sampler
    def set_x(self, events, r):
        """ Get values for self.gp_x, i.e., timepoints or
            frequency points that indicate where to sample the GPs
        """
        if not self.non_stationary:
            if self.gp_type == "spectrum":
                self.gp_x = r.gp_f.expand(context.batch_size, r.gp_f.shape[1], 1) 
            else:
                self.gp_x = r.gp_t.expand(context.batch_size, r.gp_t.shape[1], 1)
        else:
            if self.gp_type == "spectrum":
                B, E, F, = context.batch_size, len(events), r.gp_f.shape[1]

                # r.gp_f: 1 * F * 1
                f = r.gp_f.view(F).expand(B, E, F)
                e = torch.arange(
                    1, len(events)+1, device=f.device
                )[:, None].expand(B, E, F)
                self.gp_x = torch.stack([f.reshape(B, E*F), e.reshape(B, E*F)], axis=2)
            else:
                is_elems = []
                for i in range(len(events)):
                    R = events[i].step_ramps(r.gp_t[:, :, 0])  # batch len(timepoints)
                    is_elems.append(R[:, :, None])
                self.event_masks = torch.cat(is_elems, 2)  # event_masks: batch, len(gp_points), n_events
                sound_mask = torch.sum(self.event_masks, 2)  # batch, len(gp_points)
                silence_mask = 1. - sound_mask  # batch, len(gp_points)
                all_masks = torch.cat((
                    silence_mask[:, :, None], self.event_masks), dim=2
                    )  # batch, len(gp_points), n_events+1
                mask_indexes = torch.sum(torch.arange(
                    0, all_masks.shape[2],device=all_masks.device
                    )[None, None, :] * all_masks, 2, keepdims=True)  # batch, len(gp_points), 1
                self.gp_x = torch.cat((
                    r.gp_t.repeat(mask_indexes.shape[0], 1, 1), mask_indexes
                    ), 2) #batch len(gp) 2

    def forward(self, x, mean_module=None, covar_module=None):
        """ Gpytorch integration, returns the prior of the values defined at x 
            This function needs to be called 'forward' for gpytorch usage
        """
        mean_x = mean_module(x)
        covar_x = covar_module(x)
        scaled_jitter = torch.diag_embed(
            self.epsilon.square() + (covar_module.sigma * self.cholesky_stability).repeat(1,x.shape[1])
            )
        covar_x = covar_x + scaled_jitter
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Inducing point framework
    def pad(self, _x, _y, event=None, event_idx=None):
        """ Pad events with feature values at ends to prevent boundary
            effects during rendering

        Parameters
        ----------
        _x: numpy array[float]
            if event is None, timepoints or frequency points that define an event hypothesis
                shape: (number of points in event hypothesis,)
            if event is not None, a longer set of timepoints or frequency points
        _y: numpy array[float]
            if event is None, feature values that define an event in the hypothesis
                shape: (number of points in event hypothesis,)
            if event is not None, a longer set of feature values
        event: dict[str, float]
            event timing, {"onset": float, "offset":float}
        event_idx: int
            index of event in Sequence

        Returns
        -------
        x_event: numpy array[float]
            padded array of timepoints or frequency points defining an event hypothesis
            shape: (number of points in event hypothesis + 2,)
        y_event: numpy array[float]
            padded array of feature values defining an event hypothesis
            shape: (number of points in event hypothesis + 2,)
        c_event: numpy array[int]
            array that will be used to indicate the index of event in sequence
            shape: (number of points in event hypothesis + 2,)
        """
        if event is not None:
            this_event_x = (np.full(_x.shape, event["onset"]) < _x) * (_x < np.full(_x.shape, event["offset"]))
            mx = _x[this_event_x]
            my = _y[this_event_x]
        else:
            mx = _x
            my = _y
        if len(mx) > 1:
            dm = mx[1:2] - mx[:1]
        elif len(mx) == 1:  # will only ever happen in time
            dm = self.default_inducer_pad
        x_event, y_event = (
            np.concatenate((mx[:1] - dm, mx, mx[-1:] + dm)),
            np.concatenate((my[:1], my, my[-1:]))
            )
        if event is None:
            c_event = np.zeros(x_event.shape)
        else:
            c_event = np.full(x_event.shape, event_idx + 1)
        return x_event, y_event, c_event

    def get_active(self, x):
        """ Figure out which timepoints are within the duration of
            scene under consideration (some events may extend longer)

        Parameters
        ----------
        x: numpy array[float]
            timepoints where scene is defined (in sec)
        
        Returns
        -------
        active: numpy array[bool]
            timepoints that can be optimized during inference
        """
        active = (x <= (self.scene_duration + self.pad_after_active))
        if hasattr(self, "active"):
            active[:len(self.active)] = active[:len(self.active)] | self.active
        self.active = active
        return active

    def create_inducers(self, hypothesis):
        """ Create inducing points that define the variational distribution,
            based on a hypothesis

        We use an inducing point framework for variational inference.
        This function defines the inducing points (x) and corresponding
        initial feature values (y) for these points. The first step is 
        to pad the hypotheses (either neural net outputs or hand-designed
        hypotheses), This prevents boundary artifacts during rendering.
        For temporal GPs, the second step is to determine which timepoints
        are currently being inferred.

        Input
        -----
        hypothesis: dict

        Returns
        -------
        init_inducing_values: Tensor[float]
            feature values for inducers
        inducing_points: Tensor[float]
            timepoints or frequency points for inducers
            (time in sec, frequency in ERB)
        """

        if self.gp_type == "spectrum":
            # 1. pad the hypotheses (either neural net outputs or hand-designed hypotheses)
            # Note: for multi-spectrum, _x and _y are a 2d arrays: n_events * n_freqs
            #       for all other cases, _x and _y are 1d arrays: n_timepoints
            if self.non_stationary:
                _x = np.array(hypothesis["features"][self.gp_type]["x_events"])
                _y = np.array(hypothesis["features"][self.gp_type]["y_events"])
                x = []
                y = []
                c = []
                assert len(_x.shape) == 2
                for event_idx in range(_x.shape[0]):
                    x_event, y_event, c_event = self.pad(_x[event_idx,:], _y[event_idx,:])
                    x.append(x_event)
                    y.append(y_event)
                    c.append(c_event + (event_idx+1))
                x = np.concatenate(x)
                y = np.concatenate(y)
                c = np.concatenate(c)
                xc = np.concatenate((x[:, None], c[:, None]), axis=1)

                inducing_points = torch.Tensor(xc)
                self.all_event_idxs = torch.Tensor(c)
                init_inducing_values = torch.Tensor(y)
                self.inactive_inducing_points = torch.zeros(0)
                self.inactive_inducing_values = torch.zeros(0)
                self.inactive_inducing_stds = torch.zeros(0)

            else:
                _x = np.array(hypothesis["features"][self.gp_type]["x"]) 
                _y = np.array(hypothesis["features"][self.gp_type]["y"])
                x, y, c = self.pad(_x, _y)
                inducing_points = torch.Tensor(x[:, None])
                self.all_event_idxs = c[:, None]
                init_inducing_values = torch.Tensor(y)
                self.inactive_inducing_points = torch.zeros(0)
                self.inactive_inducing_values = torch.zeros(0)
                self.inactive_inducing_stds = torch.zeros(0)

        else:
            # 1. pad the hypotheses
            _x = np.array(hypothesis["features"][self.gp_type]["x"]) 
            _y = np.array(hypothesis["features"][self.gp_type]["y"])
            if self.non_stationary:
                x = []; y = []; c = [];
                for event_idx, event in enumerate(hypothesis["events"]):
                    x_event, y_event, c_event = self.pad(_x, _y, event=event, event_idx=event_idx)
                    x.append(x_event); y.append(y_event); c.append(c_event)
                x = np.concatenate(x); y = np.concatenate(y); c = np.concatenate(c)
                xc = np.concatenate((x[:, None], c[:, None]), axis=1)
            else:
                x, y, c = self.pad(_x, _y)

            # 2. find active events for temporal GPs
            # Inactive inducers are needed for sequential inference.
            # E.g., if you have a long event that extends beyond the
            # current scene duration, the neural net hypothesis will
            # be ruined by optimizing when the latter half of the
            # sound is unseen. Therefore we exclude those points from
            # being optimized, but include them in the next round of
            # sequential inference.
            self.all_event_idxs = torch.Tensor(c)
            active = self.get_active(x)
            # y-values
            init_inducing_values = torch.Tensor(y[active])  # shape: (n_active_inducing_points,)
            self.inactive_inducing_values = y[~active]
            self.inactive_inducing_stds = np.full(
                self.inactive_inducing_values.shape, self.initial_inducing_std
                )
            # x-values
            if self.non_stationary:
                inducing_points = torch.Tensor(xc[active, :])  # shape: (n_inducing_points, 2)
                self.inactive_inducing_points = xc[~active, :]
            else:
                inducing_points = torch.Tensor(x[active, None])  # shape: (n_inducing_points, 1)
                self.inactive_inducing_points = x[~active, None]

        return init_inducing_values, inducing_points

    @property
    def full_inducing_points(self):
        IP = self.variational_strategy.inducing_points.detach().cpu().numpy()
        if self.inactive_inducing_points.shape[0] > 0:
            IP = np.concatenate((
                IP, self.inactive_inducing_points
            ), axis=0)
        if IP.shape[1] == 1:
            IP = np.concatenate((IP, self.all_event_idxs), axis=1)
        return IP

    @property
    def full_inducing_values(self):
        IV = self.variational_strategy._variational_distribution._variational_mean.detach().cpu().numpy()
        if len(IV.shape) == 2 and IV.shape[0] == 1:
            IV = IV[0, :]
        if self.inactive_inducing_values.shape[0] > 0:
            IV = np.concatenate((IV, self.inactive_inducing_values), axis=0)
        return IV

    @property
    def full_inducing_stds(self):
        IS = self.variational_strategy._variational_distribution._variational_stddev.detach().cpu().numpy()
        if len(IS.shape) == 2 and IS.shape[0] == 1:
            IS = IS[0, :]
        if self.inactive_inducing_stds.shape[0] > 0:
            IS = np.concatenate((IS, self.inactive_inducing_stds), axis=0)
        return IS

    def update(self, event_proposal):
        """ Update the inducing points during sequential inference,
            if a new proposal is added

        Parameters
        ----------
        event_proposal: dict
            event hypothesis with format: {
                "features":{gp_type:{"x": list[float], "y": list[float]}, ...},
                "events":list[{"onset":float,"offset:float}]
                }

        Returns
        -------
        inducing points and inducing values for this round of inference
            see update_variational_strategy below
        """
        # Do not alter the source spectrum based on an event proposal
        if ("spectrum" in self.gp_type) and not self.non_stationary:
            return

        if ("spectrum" in self.gp_type) and self.non_stationary:
            _x = np.array(event_proposal["features"][self.gp_type]["x"])
            _y = np.array(event_proposal["features"][self.gp_type]["y"])
            x_event, y_event, c_event = self.pad(_x, _y)
            xc = np.concatenate((
                x_event[:, None],
                self.all_event_idxs.max() + 1 + c_event[:, None]
                ), axis=1)
        elif "spectrum" not in self.gp_type:
            # Only pad the newest event
            _x = np.array(event_proposal["features"][self.gp_type]["x"]) 
            _y = np.array(event_proposal["features"][self.gp_type]["y"])
            x_event, y_event, c_event = self.pad(
                _x, _y, event_proposal["events"][0], self.all_event_idxs.max()
                )
            xc = np.concatenate((x_event[:, None], c_event[:, None]), axis=1)

        # ombine existing points with newest event
        existing_inducing_points = self.full_inducing_points
        existing_inducing_values = self.full_inducing_values
        existing_inducing_stds = self.full_inducing_stds
        updated_inducing_points = np.concatenate((existing_inducing_points, xc), axis=0)
        self.all_event_idxs = updated_inducing_points[:,1]
        if not self.non_stationary:
            updated_inducing_points = updated_inducing_points[:, 0]
        updated_inducing_values = np.concatenate((existing_inducing_values, y_event))
        updated_inducing_stds = np.concatenate((
            existing_inducing_stds,
            np.full(y_event.shape, self.initial_inducing_std)
            ))

        return self.update_variational_strategy(updated_inducing_points, updated_inducing_values, updated_inducing_stds)

    def update_variational_strategy(self, inducing_points=None, inducing_values=None, inducing_stds=None):
        """
        Determine which inducing points are within the current scene duration,
        and update the variational strategy to use only those inducing points.

        Parameters
        ----------
        inducing_points: numpy array[float]
            time or frequency values (x) defining the inducers
        inducing_values: numpy array[float]
            feature values (y) defining the inducers
        inducing_stds: numpy array[float]
            standard deviation on the feature values,
            defining the variational distribution 

        Returns
        -------
        inducing_points: numpy array[float]
            [Currently active] inducing points (i.e., time or frequency points)
        inducing_values: numpy array[float]
            [Currently active] inducing values (i.e., feature values)
        """

        if ("spectrum" in self.gp_type) and not self.non_stationary:
            return

        if inducing_points is None:
            inducing_points = self.full_inducing_points
            inducing_values = self.full_inducing_values
            inducing_stds = self.full_inducing_stds

        if "spectrum" in self.gp_type and self.non_stationary:
            self.variational_strategy.update(torch.Tensor(inducing_points), torch.Tensor(inducing_values), torch.Tensor(inducing_stds))
            return inducing_points, inducing_values

        elif "spectrum" not in self.gp_type:

            # Apply basic active constraint (based on current scene_duration)
            active = self.get_active(inducing_points[:, 0])  # Time of inducing points

            active_inducing_values = torch.Tensor(inducing_values[active])  # shape: (n_active_inducing_points,)
            active_inducing_stds = torch.Tensor(inducing_stds[active])

            self.inactive_inducing_values = inducing_values[~active]
            self.inactive_inducing_stds = inducing_stds[~active]
            if self.non_stationary:
                active_inducing_points = torch.Tensor(inducing_points[active, :])  # shape: (n_inducing_points, 2)
                self.inactive_inducing_points = inducing_points[~active, :]
            else:
                active_inducing_points = torch.Tensor(inducing_points[active, None])  # shape: (n_inducing_points, 1)
                self.inactive_inducing_points = inducing_points[~active, None]

            # Given how self.full_inducer functions work,
            # inducing_points = [
            #   old inducers that were already active (_variational_mean),
            #   old inducers that used to be inactive,
            #   new inducers that are active
            # ]
            self.variational_strategy.update(active_inducing_points, active_inducing_values, active_inducing_stds)
            return active_inducing_points.numpy(), active_inducing_values.numpy()

    # Define/initialize prior and variational distributions
    def init_q(self, hypothesis):
        """ Sets up gpytorch variational distribution and variational strategy """
        init_inducing_values, inducing_points = self.create_inducers(hypothesis)
        # Set up gpytorch variational distribution
        variational_distribution = InitializableMeanFieldVariationalDistribution(init_inducing_values, inducing_points.size(0), batch_shape=context.batch_size, mean_init_std=self.initial_inducing_std)
        # Strategy marginalizes out variational distribution to
        # give you predictive mean and covariance over given X
        if self.non_stationary:
            self.variational_strategy = EventUnwhitenedVariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=self.learn_inducing_locations
            )  # set initialized to one inside the init
        else:
            self.variational_strategy = MyUnwhitenedVariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=self.learn_inducing_locations
            )
            self.variational_strategy.variational_params_initialized.fill_(1)

    def init_p(self, hyperpriors, learn_hp=False):
        """ Initialize prior distribution """
        if learn_hp and (hyperpriors["kernel"]["epsilon"]["dist"] == "parameter"):
            # Epsilon is learned separately for each scene in the hierarchical Scenes object
            self._epsilon = nn.Parameter(hyperpriors["kernel"]["epsilon"]["args"]*torch.ones(1))
            self.learn_epsilon = True
        elif learn_hp and ("epsilon" in hyperpriors["kernel"]["parametrization"]):
            # Epsilon is shared across all scenes in the hierarchical Scenes object
            self._epsilon = hyperpriors["kernel"]["epsilon"]["args"]
            self.learn_epsilon = True
        else:
            self.register_buffer("_epsilon", torch.Tensor([hyperpriors["kernel"]["epsilon"]["args"]]))
            self.learn_epsilon = False

    @property
    def epsilon(self):
        if self.learn_epsilon:
            return softplus(self._epsilon) + 0.01
        else:
            return self._epsilon

    def log_q(self, mean_module=None, covar_module=None, sample=rsample):
        """ Sample from variational distribution and return
            log probability of sample under variational distribution

        Parameters
        ----------
        mean_module: SampledConstantMean or ZeroMean
            see gp_hyperparameters.py
        covar_module: module with BaseSampledKernel as parent
            see `kernel` in gp_hyperparameters.py
        sample: function
            How to sample from the guide distribution (see util.py)

        Returns
        -------
        lq: Tensor[float]
            log probability of sample under variational distribution
            shape: [batch,]
        """
        # Requires running self.set_x from a level up,
        # likely once before log_q
        predictive_MVN = self.variational_strategy(self.gp_x, mean_module=mean_module, covar_module=covar_module)
        if sample is not None:
            self.y = sample(predictive_MVN)
        if self.non_stationary and self.gp_type != "spectrum": self.y_render = self.pad_after_sample(self.y, self.event_masks)
        lq = predictive_MVN.log_prob(self.y)
        return lq

    def log_p(self, mean_module, covar_module):
        """ Returns log probability of sample under prior
            Equation A.13 if amplitude/f0
            Equation A.18 if spectrum
        """
        # Requires running self.set_x from a level up,
        # likely once before log_q
        prior_MVN = self.forward(self.gp_x, mean_module=mean_module, covar_module=covar_module)
        lpy = prior_MVN.log_prob(self.y)
        return lpy

    # Rendering
    def pad_after_sample(self, sampled_gp, event_masks):
        """ Split sampled gaussian process into its events and
            pad it, as input to the renderer

        Parameters
        ----------
        sampled_gp: Tensor[float]
            Shape: (batch, len(timepoints))
        event_masks: Tensor[int]
            Shape: (batch, len(timepoints), n_events)

        Returns
        -------
        gp_render: Tensor[float]
            Shape: (batch, len(timepoints), n_events)
        """

        idxs = torch.arange(event_masks.shape[1], device=event_masks.device)[None, :, None]
        countup = torch.arange(1, event_masks.shape[1]+1, device=event_masks.device)[None, :, None]
        countdown = torch.arange(event_masks.shape[1], 0, -1, device=event_masks.device)[None, :, None]

        up_idx = event_masks * countup
        last_idx = torch.argmax(up_idx, dim=1, keepdims=True)

        down_idx = event_masks * countdown
        first_idx = torch.argmax(down_idx, dim=1, keepdims=True)

        left_pad = torch.where(idxs < first_idx, torch.ones(idxs.shape, device=event_masks.device), torch.zeros(idxs.shape,device=event_masks.device))
        right_pad = torch.where(idxs > last_idx, torch.ones(idxs.shape, device=event_masks.device), torch.zeros(idxs.shape,device=event_masks.device))

        event_gps = sampled_gp[:, :, None] * event_masks
        padding = torch.gather(event_gps, 1, last_idx)*right_pad + torch.gather(event_gps, 1, first_idx)*left_pad
        gp_render = event_gps + padding

        return gp_render
