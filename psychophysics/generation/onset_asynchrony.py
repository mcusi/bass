import sys
import os
import numpy as np
import soundfile as sf
from scipy.signal import periodogram

# Need klatt synthesizer: 
klattpath = './klsyn/klsyn/'
sys.path.insert(0, klattpath)
import klsyn.klatt_wrap
import klsyn.klpfile
import klsyn.xlsfile

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic  
from util import context

"""
Darwin, C. J., & Sutherland, N. S. (1984). Grouping frequency components of vowels: When is a harmonic not a harmonic?. The Quarterly Journal of Experimental Psychology, 36(2), 193-208. 
"""

def runKlatt(filenames):
    ds = []
    for pfile in filenames:
        fname, fext = os.path.splitext(pfile)
        synth = klsyn.klatt_wrap.synthesizer()
        if fext == '.xls':
            (params, comments) = klsyn.xlsfile.read(pfile)
        else:
            (params, comments) = klsyn.klpfile.read(pfile)
        synth.set_params(params)
        (d,rate) = synth.synthesize()
        ds.append(d*context.rms_ref)
        sf.write(fname + '.wav', d*context.rms_ref, rate)
        if fext == '.xls':
            sys.stderr.write("XLS output not implemented.\n")
        else:
            klsyn.klpfile.write(fname + '.wav.klp', synth=synth, comments=comments)

    return ds

def makeKlattFile(fn, duration, overallGain, f0, F1, F2, F3, F4, F5, b1):
    with open(fn,'w') as f:
        f.write('# Header comment 1\n')
        f.write('# header comment 2\n')
        f.write('sr {}\n'.format(context.audio_sr))
        f.write('du {}\n'.format(duration))
        f.write('oq 70\n')
        f.write('g0 {}\n'.format(overallGain))
        f.write('agc 0\n')
        f.write('f0 {}\n'.format(f0))
        f.write('F1 {}\n'.format(F1))
        f.write('F2 {}\n'.format(F2))
        f.write('F3 {}\n'.format(F3))
        f.write('F4 {}\n'.format(F4))
        f.write('F5 {}\n'.format(F5))
        f.write('b1 {}\n'.format(b1))
        f.write('\n')
        f.write('_varied_params_')

def expt1_settings(overwrite={}):
    settings = {
        "pad_duration": 0.050,  # s
        "vowelDuration": 56,  # ms
        "overallGain": 56,  # dB, an overall gain control is included to permit the user to adjust the output level without having to modify each source amplitude time function. 
        "F1s": [375, 396, 417, 438, 459, 480, 500],  # Hz
        "f0": 125,
        "F2": 2300,
        "F3": 2900,
        "F4": 3800,
        "F5": 4600,
        "b1": 70,
        "rampDurations": [0.016, 0.016],
        "onsets": [0, 0.032, 0.240],  # s
        "offsets": [0, 0.032, 0.240],  # s
        "onsets_and_offsets": None,
        "toneFrequency": 500
    }
    for k in settings.keys():
        if k in overwrite.keys():
            settings[k] = overwrite[k]
    settings["vowel_duration"] = settings["vowelDuration"]/1000.
    settings["whistle_onsets"] = settings["onsets"]
    return settings

def expt1(audio_folder, overwrite={}, save_wav=True):
    """ Generate onset async stimuli for experiment one

    Used klatt synthesizer, we use a python implementation from berkeley linguistics lab
    https://github.com/rsprouse/klsyn
    http://linguistics.berkeley.edu/plab/guestwiki/index.php?title=Klatt_Synthesizer
    http://linguistics.berkeley.edu/plab/guestwiki/index.php?title=Klatt_Synthesizer_Parameters#Open_Quotient_.28oq.29

    """

    settings = expt1_settings(overwrite)
    sr = context.audio_sr; rms_ref = context.rms_ref
    path = os.path.join(os.environ["sound_dir"], audio_folder, "")

    print('Experiment 1')
    print("Saving klatt files.")
    os.makedirs(os.path.join(path, "klsynOutput", ""), exist_ok=True)
    ## basic continuum
    fns = []
    for F1 in settings["F1s"]: 
        fn = os.path.join(path, 'klsynOutput', f'{F1}_basic.klp')
        makeKlattFile(fn, settings["vowelDuration"], settings["overallGain"], settings["f0"], 
          F1, settings["F2"], settings["F3"], settings["F4"], settings["F5"], settings["b1"])
        fns.append(fn)
    _ds = runKlatt(fns)

    ds = []; ds_dict = {}
    print("Making basic continuum")
    for i in range(len(_ds)):
        _d = _ds[i]
        d = basic.linearRamps(_d, settings["rampDurations"])
        d = basic.addSilence(d, [max(settings["onsets"]), max(settings["offsets"])])
        fn = '{}_basic'.format(settings["F1s"][i])
        d_to_save = basic.addSilence(d, settings["pad_duration"])
        if save_wav:
            sf.write(path + fn + ".wav", d_to_save, sr)
        ds.append(d)
        ds_dict[fn] = d_to_save

    settings["amps_vowel_x2"] = [] #Save what the actual levels of the tones will be
    print("Making other continuua")
    for i in range(len(_ds)):
        _d = _ds[i]
        f,Pxx=periodogram(_d, fs=sr, scaling='spectrum')
        i500 = np.argmin(np.abs(f - settings["toneFrequency"]))
        refLevel = 10*np.log10(Pxx[i500]/rms_ref**2)
        settings["amps_vowel_x2"].append(refLevel+6)  
        for onset in settings["onsets"]:
            for offset in settings["offsets"]:
                _dpad = basic.addSilence(_d, [onset, offset])
                toneDuration = len(_dpad)/(1.*sr)
                levelIncrease = 0
                # Check for the appropriate level increase when adding the tone
                for tonePhase in [2*np.pi*j for j in np.arange(0,1,0.1)]:
                    tone = tones.pure(toneDuration, settings["toneFrequency"], level=refLevel+6, phase=tonePhase)
                    stimulus = _dpad + tone
                    y=stimulus[int(onset*sr):int(onset*sr)+len(_d)]
                    f,Pxx=periodogram(y, fs=sr, scaling='spectrum')
                    pi500 = np.argmin(np.abs(f - settings["toneFrequency"]))
                    newLevel=10*np.log10(Pxx[pi500]/rms_ref**2)
                    levelIncrease = newLevel - refLevel
                    if np.abs(levelIncrease - 9.5) < 0.05:
                        break
                d = np.copy(ds[i])
                tone = basic.linearRamps(tone, settings["rampDurations"])
                startIdx = int(max(settings["onsets"])*sr) - int(onset*sr)
                endIdx = startIdx + len(tone)
                d[startIdx:endIdx] += tone
                d_to_save = basic.addSilence(d, settings["pad_duration"])
                fn = '{}_on{}_off{}'.format(settings["F1s"][i],int(1000.*onset),int(1000.*offset))
                if save_wav:
                    sf.write(path + fn + ".wav", d_to_save, sr)
                ds_dict[fn] = d_to_save

    settings["audio_sr"] = sr
    settings["rms_ref"] = rms_ref
    if save_wav:
        np.save(os.path.join(path, "oa_expt1_settings.npy"), settings)

    return ds_dict, settings

if __name__ == "__main__":
    print('Darwin  & Sutherland (1984)')
    print('Grouping frequency components of vowels: When is a harmonic not a harmonic?')
    print('Comment: this code requires the use of the Klatt synthesizer, as in https://github.com/rsprouse/klsyn')
    print('Git cloning this repository should include klsyn as a submodule. Downloading the zip may not.')
    print('Current path to Klatt should be: ' + klattpath)
    audio_folder = str(sys.argv[1])
    print("Saving in ", audio_folder)
    expt1(audio_folder)
    print('~~~~~~~~~~')


