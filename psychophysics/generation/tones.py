import numpy as np

import psychophysics.generation.basic as basic
from util import context


def pure(duration, frequency, level=0, phase=0):
    """pure:
    generates a constant-ampiltude tone consisting of a single frequency
    
    Input
    -----
    duration (s)
    frequency (Hz)
    level: average (dB)
    phase (rads)
    sr (Hz)


    Returns
    -------
    tone
    """

    sr = context.audio_sr
    x_rms = context.rms_ref*np.power(10., level/20.)
    t = np.arange(0, duration, 1./sr)
    soundwave = np.sin(2*np.pi*frequency*t + phase)
    tone = (x_rms/np.sqrt(np.mean(np.square(soundwave))))*soundwave

    return tone


def harmonic(duration, f0, harmonics, levels=[], phases=[]):
    """harmonic:
    generates a constant-amplitude tone with frequencies 
    at integer multiples of fundamental frequency

    Input
    -----
    duration (s)
    f0: fundamental frequency (Hz)
    harmonics: list of 1's and 0's 
        1's indicate the presence of a harmonic 
        0's indicate skipping that harmonic

        harmonics[0]: 1st harmonic, i.e. fundamental frequency
        len(harmonics) + 1 is the highest harmonic
        (len(harmonics) + 1)*f0 must be less than Nyquist, sr/2. 

    levels: relative levels of each component (dB)
        if [0], equal level for all components
    phases: relative phases of each component  (rads)
        if [0], sinephase for all components
        if 'rand', randomize phase for each component
    sr (Hz)


    Returns
    -------
    tone
    """

    freq_idx = range(len(harmonics))
    if len(levels) == 0:
        levels = [0 for i in freq_idx]
    if len(phases) == 0:
        phases = [0 for i in freq_idx]
    elif phases == 'rand':
        phases = [2*np.pi*np.random.rand() for i in freq_idx]

    assert len(levels) == len(harmonics)
    assert len(phases) == len(harmonics)

    tone = basic.silence(duration)
    for i in freq_idx:
        if harmonics[i] == 0:
            continue
        elif harmonics[i] == 1:
            harmonic = pure(duration, f0*(i+1), level=levels[i], phase=phases[i])
        tone += harmonic

    return tone


def complex(duration, frequencies, levels=[], phases=[]):
    """complexTone:
    generates a constant-amplitude tone with specified frequencies  

    Input
    -----
    duration (s)
    f0: fundamental frequency (Hz)
    frequencies (Hz)
    levels: relative levels of each component (dB)
        if [0], equal level for all components
    phases: relative phases of each component  (rads)
        if [0], sinephase for all components
        if 'rand', randomize phase for each component
    sr (Hz)


    Returns
    -------
    tone
    """

    freq_idx = range(len(frequencies))
    if len(levels) == 0:
        levels = [0 for i in freq_idx]
    if len(phases) == 0:
        phases = [0 for i in freq_idx]
    elif phases == 'rand':
        phases = [2*np.pi*np.random.rand() for i in freq_idx]

    assert len(levels) == len(frequencies)
    assert len(phases) == len(frequencies)

    tone = basic.silence(duration)
    for i in freq_idx:
        overtone = pure(duration, frequencies[i], level=levels[i], phase=phases[i])
        tone += overtone

    return tone


def sinusoidalAM(duration, f_c, phi_c, f_m, phi_m, modulation_depth, level):
    """sinusoidalAM:
    generates a tone with constant sinusoidal amplitude modulation

    input
    -----
    duration (s)
    f_c (Hz), phi_c (rads): carrier parameters
    f_m (Hz), phi_m (rads): modulator parameters
    modulation_depth: must be in between 0 and 1. 
    level (dB): average level
    ref (Pa), sr (Hz)

    Hartmann pg 400 eqn 17-17
    """

    assert modulation_depth > 0 and modulation_depth < 1
    t = basic.t(duration)
    tone = (1.0 + modulation_depth * np.cos(2.0*np.pi*f_m*t + phi_m))*np.sin(2.0*np.pi*f_c*t + phi_c)
    desired_rms = context.rms_ref * np.power(10., level/20.)
    tone = (desired_rms/basic.rms(tone)) * tone

    return tone


def sinusoidalFM(duration, f_c, phi_c, f_m, phi_m, delta_f, level):
    """sinusoidalFM:
    generates a tone with sinusoidal frequency modulation

    input
    -----
    duration (s)
    f_c (Hz), phi_c (rads): carrier parameters
    f_m (Hz), phi_m (rads): modulator parameters
    delta_f (Hz) change in frequency
    level (dB)
    ref (Pa), sr (Hz)

    Hartmann pg 430 eqns 19.1-19.6
    """

    t = basic.t(duration)
    desired_rms = context.rms_ref*np.power(10., level/20.)
    tone = np.sin(2*np.pi*f_c*t + phi_c + delta_f/(1.*f_m)*np.sin(2*np.pi*f_m*t + phi_m))
    tone = (desired_rms/basic.rms(tone)) * tone

    return tone


def pureGlide(duration, f_init, f_final, level=0, phase=0, scale='log'):
    """pureGlide:
    generates a pure tone that changes linearly on log or ERB scale

    Input
    -----
    duration (s)
    f_init: initial frequency (Hz), f_final: final frequency (Hz)
    level (dB)
    phase (rads)
    sr (Hz)
    scale: 'log' or 'ERB' or 'linear'

    Returns
    -------
    glide
    """
    
    sr = context.audio_sr
    x_rms = context.rms_ref*np.power(10., level/20.)
    t = np.arange(0, duration, 1./sr)
    if scale == 'ERB':
        f = basic.erb2freq(np.linspace(basic.freq2erb(f_init), basic.freq2erb(f_final), num=len(t)))
    elif scale == 'log':
        f = np.exp(np.linspace(np.log(f_init), np.log(f_final), num=len(t)))
    elif scale == 'linear':
        f = np.linspace(f_init, f_final, num=len(t))
    dt = 1./sr
    glide = np.sin(2*np.pi*np.cumsum(f*dt) + phase)
    glide = x_rms*glide/np.sqrt(np.mean(np.square(glide)))

    return glide
