import numpy as np
from scipy.fftpack import fft, ifft

from util import context


def whiteGaussian(duration, p, units, imp=1):
    """whiteGaussian: 
    mimics MATLAB wgn function: 
    Generates Gaussian white noise. 
    The unit of measure for the output of the wgn function is Volts.
    For power calculations, it is assumed that there is a load of 1 Ohm.

    Input
    -----
    duration (s)
    p: power of the output noise, which is assumed to be referenced to an
    impedance of 1 Ohm, unless imp explicitly defines the impedance
    units: units of p
        'dBW': decibels (dB) reference to one Watt
        'dBm': decibels (dB) referenced to one milliWatt (mW)
        'linear': Watts
    sr: sampling rate (Hz)
    imp: impedance (Ohm)

    Returns
    -------
    noise
    """

    sr = context.audio_sr
    if units == "dBW":
        npow = np.power(10., p/10.)
    elif units == "dBm":
        npow = np.power(10, (p-30)/10.)
    elif units == "linear":
        npow = p

    noise = np.sqrt(imp*npow) * np.random.randn(int(duration*sr))

    return noise


def pink(duration, level, band=[]):
    """pink 
    generates pink noise
    power spectral density (energy or power per frequency interval)
    is inversely proportional to the frequency of the signal.
    In pink noise, each octave (halving/doubling in frequency)
    carries an equal amount of noise energy.

    Input
    -----
    level (dB rms), relative to context.rms_ref
    band ([Hz, Hz]):
        low and high limit of bandpass
        defaults to not bandpass
    duration (s)
    sr: sampling rate (Hz)

    Returns
    -------
    noise
    """

    sr = context.audio_sr
    # By default, non-zero amplitude spans up to Nyquist frequency
    if len(band) == 0:
        band = [0, int(sr/2.)]

    n = int(np.round(duration*sr))
    binFactor = n/(1.*sr)  # move between frequencies and bin indexes
    lowPassBin = int(np.round(band[0]*binFactor))
    highPassBin = int(np.round(band[1]*binFactor))
    x_real = np.zeros(n)
    x_imag = np.zeros(n)
    x_real[lowPassBin:highPassBin+1] = np.random.randn(1+highPassBin-lowPassBin)
    x_imag[lowPassBin:highPassBin+1] = np.random.randn(1+highPassBin-lowPassBin)
    spectrum = x_real + x_imag*1j

    # divide each element of the spectrum by f^(1/2).
    # Since power is the square of amplitude,
    # this means we divide the power at each component by f
    # if you want to include arbitrary beta parameter, use instead np.power(pinkWeights,beta/2.)
    # beta = 0: white noise,  beta = 2: red (Brownian) noise
    pinkWeights = binFactor*np.arange(1, n+1)
    pinkSpectrum = np.divide(spectrum, np.sqrt(pinkWeights))

    noise = np.real(ifft(pinkSpectrum))
    noise = noise[0:int(duration*sr)]
    noise = (context.rms_ref*np.power(10, level/20.) / np.sqrt(np.mean(np.square(noise))) ) * noise

    return noise


def bpWhite(duration, level, band=[]):
    """bpWhite
    bandpass white noise

    Input
    -----
    level (dB rms), relative to internal ref
    band ([Hz, Hz]):
        low and high limit of bandpass
        defaults to not bandpass
    duration (s)
    sr: sampling rate (Hz)

    Returns
    -------
    noise
    """

    sr = context.audio_sr
     #by default, non-zero amplitude spans up to Nyquist frequency
    if len(band) == 0:
        band = [0, int(sr/2.)]

    n = int(np.round(duration*sr))
    binFactor = n/(1.*sr)  # move between frequencies and bin indexes
    lowPassBin = int(np.round(band[0]*binFactor))
    highPassBin = int(np.round(band[1]*binFactor))

    x_real = np.zeros(n)
    x_imag = np.zeros(n)
    x_real[lowPassBin:highPassBin+1] = np.random.randn(1+highPassBin-lowPassBin)
    x_imag[lowPassBin:highPassBin+1] = np.random.randn(1+highPassBin-lowPassBin)
    spectrum = x_real + x_imag*1j

    noise = np.real(ifft(spectrum))
    noise = noise[0:int(duration*sr)]
    noise = context.rms_ref*np.power(10, level/20.) * (1./np.sqrt(np.mean(np.square(noise)))) * noise

    return noise


def bpWhiteSpecLevel(duration, spectrumLevel, band):
    """whiteSL:
    bandpass white noise with specified spectrum level

    Input
    -----
    spectrumLevel (dB/Hz)
    band ([Hz, Hz]): low and high limit of bandpass
    duration (s)
    sr: sampling rate (Hz)

    Returns
    -------
    noise

    Alternative way of getting spectrum level
    >> after sampling N(0,1) noise:
    n = int(duration*sr) #length of fft
    overall_level = spectral_level + 10*log10(band[1]-band[0]);
    noise = noise .* sqrt(2*n*(sr/2.)/(band[1]-band[0]));
    noise = noise .* 10^(overall_level/20);
    """
    
    sr = context.audio_sr
    wattsPerHz = np.power(context.rms_ref, 2)*np.power(10., (spectrumLevel/10.))
    watts = wattsPerHz * (sr/2.)

    gnoise = whiteGaussian(1, watts, 'linear')
    spectrum = np.abs(fft(gnoise))
    # low pass
    spectrum[0:band[0]] = 0
    spectrum[-band[0]:] = 0
    # high pass
    spectrum[band[1]:-band[1]] = 0

    N = len(spectrum)
    Nh = int(np.ceil(N/2.))
    phi = np.random.rand(N)*2*np.pi
    x_real = np.multiply(spectrum[0:Nh], np.concatenate(([0], np.cos(phi[1:Nh]))))
    x_imag = np.multiply(spectrum[0:Nh], np.concatenate(([0], np.sin(phi[1:Nh]))))

    x = np.zeros(N, dtype=np.complex128)
    if N % 2 == 0:
        x[0:Nh] = x_real + x_imag*1j
        y = np.concatenate((x_real[1:Nh] - x_imag[1:Nh]*1j, [1]))
        x[Nh:] = y[::-1]
    else:
        x[0:Nh] = x_real + x_imag*1j 
        y = x_real[1:Nh] - x_imag[1:Nh]*1j
        x[Nh:] = y[::-1]

    noise = np.real(ifft(x))
    noise = noise[:int(duration*sr)]

    return noise


def octave2freq(cf, nOctaves):
    """ n-octave range around centre frequency (cf) """
    return [cf * np.power(2., -nOctaves/2.), cf * np.power(2., nOctaves/2.)]
