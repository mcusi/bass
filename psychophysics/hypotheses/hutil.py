import numpy as np
from copy import deepcopy

import renderer.util


def create_noise_latents(spectrum_level, config, audio_sr):
    """ Determine source latent variables from spectrum levels """
    bw_config = {"renderer": deepcopy(config["renderer"])}
    bw_config["renderer"]["steps"] = config["hypothesis"]["delta_gp"]
    bw = renderer.util.get_bandwidthsHz(audio_sr, bw_config)
    erbs = renderer.util.get_event_gp_freqs(
        audio_sr,
        config["hypothesis"]["delta_gp"],
        lo_lim_freq=config["renderer"]["lo_lim_freq"]
    )
    fs = renderer.util.ERB_to_freq(erbs)
    wattsPerHz = np.power(10., (np.array([spectrum_level(f) for f in fs])/10.))
    pink_wattsPerHz = 1 / (np.log((audio_sr/2)/config["renderer"]["lo_lim_freq"])*fs)
    power_density_ratio = wattsPerHz / pink_wattsPerHz
    spectrum = 10*np.log10(power_density_ratio / bw)
    amplitude = np.mean(spectrum)
    spectrum = spectrum - amplitude
    return erbs, spectrum, amplitude


def format(hypothesis_list_of_dicts, observation_name, hypothesis_name):
    k = (observation_name, hypothesis_name)
    return {k: hypothesis_list_of_dicts}