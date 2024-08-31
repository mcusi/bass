import torch
import numpy as np

from inference.amortized.cgram_util import create_gammatonegram
import renderer.cochleagrams as cgram

def make_gammatonegram(w, audio_sr):
    """ Convenience function for making gammatonegrams for preference function """
    gtg_params={
        "ref": 1.0e-06,
        "twin": 0.025,
        "thop": 0.01,
        "nfilts": 64,
        "fmin": 20,
        "width": 0.5,
        "log_constant": 1.0e-80,
        "dB_threshold": 20
        }
    gtg_module = {}
    gtg_module["nfft"], gtg_module["nhop"], gtg_module["nwin"], gtm = cgram.gtg_settings(
        sr=audio_sr, twin=gtg_params["twin"], thop=gtg_params["thop"],
        N=gtg_params["nfilts"], fmin=gtg_params["fmin"],
        fmax=audio_sr/2.0, width=gtg_params["width"], return_all=False
        )
    gtg_module["gtm"] = torch.Tensor(gtm[np.newaxis, :, :])
    gtg_module["window"] = torch.hann_window(gtg_module["nwin"])
    gtg_module["rms_ref"] = gtg_params["ref"]
    with torch.no_grad():
        tf_dict = create_gammatonegram(
            torch.Tensor(w[None, :]), gtg_module=gtg_module
            )
    return tf_dict["grams"]


def get_distance_metric(mode):
    return lambda v, network_output, mask: ((v - network_output)**2).mean().sqrt().item()


def combine_distances(distances, mode="rms"):
    return np.sqrt(np.mean(np.square(distances)))
