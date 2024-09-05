import inspect
import matplotlib
import numpy as np
import scipy.special
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from util.context import context

#############
# Different loss functions
#############


def importance_weighted_bound(scores):
    """ Returns the importance weighted bound from a set of scores output from Scene 
        Requires `with context(...):` at level above

        Corresponds to equation 8 in https://arxiv.org/pdf/1509.00519.pdf
        See Appendix B > Section B.1.1, Inference: Hypothesis optimization and scene selection > Equation B.4
    """
    assert (torch.all(~torch.isnan(scores)) and torch.all(~torch.isinf(scores)))
    cat_scores = scores.reshape([
        context.optimization["n_importance_samples"],
        int(context.batch_size/context.optimization["n_importance_samples"])
        ])
    loss = torch.mean(-torch.logsumexp(cat_scores,0) + np.log(cat_scores.shape[0]))
    return loss


def mean_score(scores):
    assert (torch.all(~torch.isnan(scores)) and torch.all(~torch.isinf(scores)))
    loss = torch.mean(-scores)
    return loss

#############
# Metrics logger
#############


def elbo(scores):
    return scipy.special.logsumexp(scores) - np.log(len(scores))


def ess(scores):
    #http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html
    return np.exp(  2. * scipy.special.logsumexp(scores) - scipy.special.logsumexp(2. * scores)  )


class BasicLogger():
    def __init__(self, checkpoint=None, track_best=True, metadata={}):
        # Initialize basic metrics
        if checkpoint is None:
            self.loss = []
            self.lp = []
            self.lq = []
            self.ll = []
            self.scores = []
            self.track_best = track_best
            self.metadata = {**metadata}
            
            # self.params = defaultdict(list)
            # self.grads = defaultdict(list)
            # self.params
            self.grad_mean = {}
            self.grad_sq_mean = {}
            self.grad_abs_mean = {}

            self.training_time = 0

            if self.track_best:
                # Initialize "best" tracking
                self.best_scene = None
        else:
            metrics = checkpoint["metrics"]               
            attributes = inspect.getmembers(metrics, lambda a: not(inspect.isroutine(a)))
            attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
            for k, v in attributes:
                setattr(self, k, v)

    def track_scene_with_best_loss(self, loss, scene, scores):
        # Track best scene in case optimization diverges
        if self.track_best:
            if (self.best_scene is None) or (loss <= np.min(self.loss)):
                self.best_scene = scene.clone()
                self.best_scene_scores = scores
                self.best_scene_idx = len(self.scores)-1
                self.best_loss = np.array(self.loss[-100:]).mean()
                self.best_score = np.concatenate(self.scores[-100:]).mean()
        return

    def update(self, scene, loss, scores, step_time, **kwargs):
        # Update basic metrics
        self.loss.append(loss.item())
        best_sample_idx = np.argmax(scores.detach().cpu().numpy())
        self.lp.append(scene.lp[best_sample_idx].item()) 
        self.lq.append(scene.lq[best_sample_idx].item()) 
        self.ll.append(scene.ll[best_sample_idx].item()) 
        self.scores.append(scores.detach().cpu().numpy()) 

        if hasattr(self, "training_time"):
            self.training_time += step_time

        self.track_scene_with_best_loss(loss.item(), scene, scores.detach().cpu().numpy())

        # Update additional metrics
        for key, value in kwargs.items():
            if hasattr(self, key):
                self[key].append(value)
            else:
                setattr(self, key, [value])

    def accumulate(self, scene, loss, scores, accumulation_idx):
        """ For use in gradient accumulation when there's a cuda error with bigger batch sizes """
        # Update basic metrics
        if accumulation_idx == 0:
            self.accumulate_losses = [loss.item()]
        else:
            self.accumulate_losses.append(loss.item())
        self.accumulate_scores = scores.squeeze().detach().cpu().numpy() if accumulation_idx == 0 else np.concatenate((self.accumulate_scores, scores.squeeze().detach().cpu().numpy()))
        self.accumulate_lp = scene.lp.squeeze().detach().cpu().numpy() if accumulation_idx == 0 else np.concatenate((self.accumulate_lp, scene.lp.squeeze().detach().cpu().numpy()))
        self.accumulate_lq = scene.lq.squeeze().detach().cpu().numpy() if accumulation_idx == 0 else np.concatenate((self.accumulate_lq, scene.lq.squeeze().detach().cpu().numpy()))
        self.accumulate_ll = scene.ll.squeeze().detach().cpu().numpy() if accumulation_idx == 0 else np.concatenate((self.accumulate_ll, scene.ll.squeeze().detach().cpu().numpy()))

    def accumulation_update(self, scene, step_time=None):
        """ For use in gradient accumulation when there's a cuda error with bigger batch sizes """
        #Update basic metrics
        loss = np.mean(self.accumulate_losses).item()
        self.loss.append(loss)
        best_sample_idx = np.argmax(self.accumulate_scores)
        self.lp.append(self.accumulate_lp[best_sample_idx].item())
        self.lq.append(self.accumulate_lq[best_sample_idx].item())
        self.ll.append(self.accumulate_ll[best_sample_idx].item())
        self.scores.append(self.accumulate_scores)
        if hasattr(self, "training_time"):
            self.training_time += step_time
        self.track_scene_with_best_loss(loss, scene, self.accumulate_scores)

    def elbo(self):
        return elbo(np.concatenate(self.scores))

    def ess(self):
        return ess(np.concatenate(self.scores))

    def summary_dict(self):
        return {
            'elbo': self.elbo(),
            'ess': self.ess(),
            'n_batch': len(self.scores),
            'n': len(np.concatenate(self.scores)),
            'best_iter': int(np.argmin(self.loss)),
            'total_iters': len(self.loss),
            'best_score': self.best_score.item() if hasattr(self, "best_score") else None,
            'best_loss': self.best_loss.item() if hasattr(self, "best_loss") else None,
            **self.metadata
        }

    def plot(self, savepath):
        plt.plot(self.loss); plt.savefig(savepath + "losses.png"); plt.close()
        plt.plot(self.lp); plt.savefig(savepath + "LQ.png"); plt.close()
        plt.plot(self.lq); plt.savefig(savepath + "LP.png"); plt.close()
        plt.plot(self.ll); plt.savefig(savepath + "LL.png"); plt.close()
        plt.plot(np.array(self.lq)-np.array(self.lp)); plt.savefig(savepath + "KL.png"); plt.close()


class SourcePriorLogger():

    def __init__(self, checkpoint=None):
        # Initialize basic metrics
        if checkpoint is None:
            self.loss = []
            self.epoch_loss = []
            self.scores = []
            self.feature_hps = []
            self.sequence_hps = []
        else:
            metrics = checkpoint["metrics"]      
            attributes = inspect.getmembers(metrics, lambda a:not(inspect.isroutine(a)))
            attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
            for k, v in attributes:
                setattr(self, k, v)

    def step_update(self, loss, scores, **kwargs):

        # Update basic metrics
        self.loss.append(loss.item()) 
        self.scores.append(scores.detach().cpu().numpy()) 

        # Update additional metrics
        for key, value in kwargs.items():
            if hasattr(self, key):
                self[key].append(value)
            else:
                setattr(self, key, [value])

    def epoch_update(self, scenes, n_chunks):

        # Keep track of hyperpriors
        if hasattr(scenes, "feature_hp"):
            S = {feature: {
                    kernel_name: param.clone().detach().cpu().numpy() for kernel_name, param in pdict.items()
                } for feature, pdict in scenes.feature_hp.items()
            }
            self.feature_hps.append(S)

        if hasattr(scenes, "sequence_hp"):
            S = {feature: {
                    hp_name: param.clone().detach().cpu().numpy() for hp_name, param in pdict.items()
                } for feature, pdict in scenes.sequence_hp.items()
            }
            self.sequence_hps.append(S)

        # Update epoch loss
        self.epoch_loss.append(np.mean(self.loss[-n_chunks:]))

    def plot(self, savepath):
        plt.plot(self.epoch_loss); plt.savefig(savepath + "epoch_losses.png"); plt.close()
