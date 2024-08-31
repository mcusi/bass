import numpy as np
from util import context

def silence(duration):
    """silence
    Input
    -----
    duration (s), sr (Hz)

    Returns
    -------
    silence, vector of zeros
    """
    sr = context.audio_sr
    return np.zeros(int(duration*sr))

def addSilence(s, silenceDuration):
    """addSilence
    concatenate silence to another signal 

    Input
    -----
    s, signal to add silence to
    silenceDuration, float or 2-vector [sec sec]
        if float: assume want beginning and end silence to be the same
        otherwise:
        silenceDuration[0]: silence to add before s
        silenceDuration[1]: silence to add after s
    sr: sampling rate (Hz)

    Returns
    -------
    silence, vector of zeros
    """

    sr = context.audio_sr
    if type(silenceDuration) is float or type(silenceDuration) is int:
        silenceDuration = [silenceDuration, silenceDuration]

    silenceBefore = silence(silenceDuration[0])
    silenceAfter = silence(silenceDuration[1])
    s_padded = np.concatenate((silenceBefore, s, silenceAfter))

    return s_padded

def raisedCosineRamps(s,rampDuration):
    """raisedCosineRamps
    multiply beginning and end of signal with half a hann window

    Parameters
    ----------
    sr: Sampling frequency (Hz)
    rampDuration: quarter of a period of cosine (s)

    Returns
    -------
    ramped signal 
    """

    sr = context.audio_sr
    if type(rampDuration) is not float and len(rampDuration) == 2:
        t_off = np.arange(0,rampDuration[1]+1./sr,1./sr)
        offRamp = 0.5*(1 + np.cos(np.pi/rampDuration[1]*t_off))
        t_on = np.arange(0,rampDuration[0]+1./sr,1./sr)
        onRamp = 0.5*(1 + np.cos(np.pi/rampDuration[0]*t_on))
        onRamp = onRamp[::-1]
    else: 
        t = np.arange(0,rampDuration+1./sr,1./sr)
        offRamp = 0.5*(1 + np.cos(np.pi/rampDuration*t))
        onRamp = offRamp[::-1]

    s[0:len(onRamp)]=np.multiply(s[0:len(onRamp)],onRamp)
    s[-len(offRamp):]=np.multiply(s[-len(offRamp):],offRamp)

    return s

def linearRamps(s,rampDuration):
    """linearRamps
    multiply beginning and end of sound with line from 0 to 1

    Parameters
    ----------
    sr: Sampling frequency (Hz)
    rampDuration: rise time from 0 to 1 (s)

    Returns
    -------
    ramped signal 
    """

    sr = context.audio_sr
    if type(rampDuration) is not float and len(rampDuration) == 2:
        t_off = np.arange(0,rampDuration[1]+1./sr,1./sr)
        offRamp = 1-(1./rampDuration[1])*t_off 
        t_on = np.arange(0,rampDuration[0]+1./sr,1./sr)
        onRamp = (1./rampDuration[0])*t_on
    else: 
        t = np.arange(0,rampDuration+1./sr,1./sr)
        offRamp =  1-(1./rampDuration)*t 
        onRamp = (1./rampDuration)*t

    s[0:len(onRamp)]=np.multiply(s[0:len(onRamp)],onRamp)
    s[-len(offRamp):]=np.multiply(s[-len(offRamp):],offRamp)

    return s

def sRamps(s, rampDuration):

    sr = context.audio_sr
    if type(rampDuration) is not float and len(rampDuration) == 2:
        t_off = np.arange(0,rampDuration[1],1./sr)
        offRamp = np.power(np.sin(np.pi/(2.*rampDuration[1])*t_off),2)
        offRamp = offRamp[::-1]
        t_on = np.arange(0,rampDuration[0],1./sr)
        onRamp = np.power(np.sin(np.pi/(2.*rampDuration[0])*t_on),2)
    else: 
        t = np.arange(0,rampDuration,1./sr)
        onRamp = np.power(np.sin(np.pi/(2.*rampDuration)*t),2)
        offRamp = onRamp[::-1]

    s[0:len(onRamp)]=np.multiply(s[0:len(onRamp)],onRamp)
    s[-len(offRamp):]=np.multiply(s[-len(offRamp):],offRamp)

    return s

def repeating(s,nReps,silenceDuration=0,period=0,silenceFirst=True,silenceLast=True):
    """repeating: 
    sound alternating with silence, where either full period duration 
    or duration of the silence is specified 

    Input
    -----
    s: acoustic waveform (vector of floats)
    nReps (int) : number of repetitions of s
    silenceDuration(s) or period (s): specifies spacing of tones
    silenceFirst (bool) :  first section of signal is silence?
    silenceLast (bool) : last section of signal is silence? 
    sr (Hz)

    Returns
    -------
    sequence
    """

    sr = context.audio_sr
    assert (period > 0 or silenceDuration > 0)
    assert nReps > 1
    if period > 0:
        silenceDuration = period - len(s)/(1.*sr)

    quiet = silence(silenceDuration)
    for i in range(nReps):
        if i == 0:
            sequence = np.concatenate((quiet,s,quiet)) if silenceFirst else np.concatenate((s,quiet))
        elif i == nReps - 1:
            sequence = np.concatenate((sequence,s,quiet)) if silenceLast  else np.concatenate((sequence,s))
        else:
            sequence = np.concatenate((sequence, s, quiet))

    return sequence

def alternating(s1,s2,nReps):
    """repeating: 
    s1 alternates with s2, nReps times

    Input
    -----
    s1 and s2: acoustic waveform (vector of floats)
    nReps (int) : number of repetitions of s
    sr (Hz)

    Returns
    -------
    sequence
    """

    sr = context.audio_sr
    unit = np.concatenate((s1,s2))
    sequence = np.tile(unit,nReps)

    return sequence

def t(duration):
    """returns a time vector from time = 0 to duration, at the specified sampling rate
    """
    sr = context.audio_sr
    return np.arange(0,duration,1./sr)

###Frequency conversions
def freq2erb(freq):
    return  9.265*np.log(1+freq/(24.7*9.265))

def erb2freq(erb):
    return 24.7*9.265*(np.exp(erb/9.265)-1)

def semitone2freq(f, nSemitones):
    #return frequency that is a certain number of semitones away from f
    return f * np.power(2.,nSemitones/12.)

def octave2freq(f, nOctaves):
    #return frequency that is a certain number of octaves away from f
    return f * np.power(2.,nOctaves)

def octaveRange(cf, nOctaves):
    #n-octave range around centre frequency (cf)
    return [octave2freq(nOctaves/(-2.)), octave2freq(nOctaves/2.)]

def rms(x):
    """rms: a constant which would lead to the same average power as the signal

    input
    -----
    x, a signal defined in terms of pressure amplitude

    ref: Hartmann (pg 25, eqn 3.5)
    """
    return np.sqrt(np.mean(np.square(x)))