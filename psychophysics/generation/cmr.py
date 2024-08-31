import os
import sys
import numpy as np
import soundfile as sf
import scipy.signal as sps
import matplotlib.pyplot as plt

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic
import psychophysics.generation.noises as noises
from util import context, manual_seed


"""
Hall, J. W., Haggard, M. P., & Fernandes, M. A. (1984). 
Detection in noise by spectro-temporal pattern analysis. 
The Journal of the Acoustical Society of America, 76(1), 50-56.

Schooneveldt, G. P., and Moore, B. C. J. (1989).
Comodulation masking release(CMR) as a function of masker bandwidth, modulator bandwidth, and signal duration. 
JASA, 85(1), 273-281.

~~Naming format~~
Name format for sounds with tone present:
    `{noise type: mult or rand}_tone_bw{bandwidth in Hz}_toneLevel{level of tone in dB}_{trial number}.wav`
Trial number is just an integer index, referring to different noise samples. 
    Eg. sounds with `trial_number = 0` has the same random noise seed, and for mult, the same amplitude profile applied to the masker
Name format for sounds with tone absent:
    `{noise type: mult or rand}_noTone_bw{bandwidth in Hz}_{trial number}.wav`
Trial number matches with the sounds with the tone present, so no-tone_0 has the same noise as tone_0.
"""


def expt1_settings(overwrite={}):
    settings = {
        "n_trials": 1,
        "cf": 1000,  # Hz
        "bandwidths": [25, 50, 100, 200, 400, 1000],  # Hz
        "noise_duration": 0.400,  # s
        "noise_spectrum_level": 40,  # dB
        "tone_ramp_duration": 0.050,  # s
        "tone_levels": np.arange(20, 90, step=5),
        "silence_duration": 0.300,  # s
        "plot_option": False,
        "lowpass_for_mod": 50,  # Hz
        "silence_padding": 0.020
    }
    for k in settings.keys():
        if k in overwrite.keys():
            settings[k] = overwrite[k]
    settings["noise_ramp_duration"] = settings["tone_ramp_duration"]
    settings["tone_duration"] = 0.300 + 2*settings["tone_ramp_duration"]  # s
    settings["gap"] = basic.silence(settings["silence_duration"])
    return settings


def expt1(audio_folder, overwrite={}, paper='hall', seed=0, save_wav=True):

    print('CMR: Experiment 1. Settings:')
    manual_seed(seed)

    if "hall" in paper:
        settings = expt1_settings(overwrite)
    elif 'schooneveldt' in paper:
        overwrite = {
            "cf": 2000,
            "bandwidths": [100, 200, 400, 800, 1600, 3200],
            "tone_ramp_duration": 0.010,
            "lowpass_for_mod": 12.5
        }
        settings = expt1_settings(overwrite)
    print(settings)
    if save_wav:
        os.makedirs(os.environ["sound_dir"] + audio_folder, exist_ok=True)

    # Multiplied noise
    expt_dict = {}
    for bandwidth in settings["bandwidths"]:
        for i in range(settings["n_trials"]):
            meanMultSpectrumLevel = 0
            meanRandSpectrumLevel = 2
            addLevel = 0
            wentAbove = False
            while (np.abs((meanMultSpectrumLevel - settings["noise_spectrum_level"])) > 1) or (np.abs((meanRandSpectrumLevel - settings["noise_spectrum_level"])) > 1)  or (np.abs(meanMultSpectrumLevel - meanRandSpectrumLevel) > 1):

                # Generate noises than necessary duration, and then cut
                # "Noise spectrum levels were measured integrating over a time of 2 min"
                wgn0 = noises.bpWhite(
                    settings["noise_duration"]*10,
                    settings["noise_spectrum_level"] + 10*np.log10(context.audio_sr/2.)
                    )
                wgn = noises.bpWhite(
                    settings["noise_duration"]*10,
                    settings["noise_spectrum_level"] + 10*np.log10(context.audio_sr/2.) + addLevel
                    )
                lpn = noises.bpWhite(settings["noise_duration"]*10, 0, band=[0., settings["lowpass_for_mod"]])
                lpn /= np.sqrt(np.mean(np.square(lpn)))
                multnoise = np.multiply(wgn, lpn)

                # filter to appropriate bandwidth
                bandLimits = [
                    settings["cf"]-bandwidth/2.,
                    settings["cf"]+bandwidth/2.
                ]
                Wn = [_*(1./(context.audio_sr/2.)) for _ in bandLimits]
                b, a = sps.butter(4, Wn, btype='bandpass')
                filtMultNoise = sps.filtfilt(b, a, multnoise)
                filtRandNoise = sps.filtfilt(b, a, wgn0)

                # Check that manipulation worked
                # 1. relative spectrum level: check that they're roughly equal
                # for the multiplied and random noise 
                f, PxxMult = sps.welch(filtMultNoise, fs=context.audio_sr, nperseg=10000, scaling='density')
                f, PxxRand = sps.welch(filtRandNoise, fs=context.audio_sr, nperseg=10000, scaling='density')
                df = f[2] - f[1]

                lowf = np.argmin((np.abs(f - bandLimits[0]))) 
                highf = np.argmin((np.abs(f - bandLimits[1])))
                meanMultSpectrumLevel = 10*np.log10(np.sum(df*PxxMult[lowf:highf])/context.rms_ref**2)-10*np.log10(bandwidth) if lowf != highf else 10*np.log10(df*PxxMult[lowf]/context.rms_ref**2)-10*np.log10(bandwidth)
                meanRandSpectrumLevel = 10*np.log10(np.sum(df*PxxRand[lowf:highf])/context.rms_ref**2)-10*np.log10(bandwidth) if lowf != highf else 10*np.log10(df*PxxRand[lowf]/context.rms_ref**2)-10*np.log10(bandwidth)

                # Absolute spectrum level: check that they're around desired level
                # otherwise adjust until appropriate
                if (np.abs((meanMultSpectrumLevel - settings["noise_spectrum_level"])) > 1):
                    if meanMultSpectrumLevel > settings["noise_spectrum_level"]:
                        addLevel -= 1
                        wentAbove = True
                    else:
                        addLevel += (10 if not wentAbove else 1)

            if settings["plot_option"]:
                print('~~~~~ bandwidth: ' + str(bandwidth) + '~~~~~~')
                plt.plot(f, 10*np.log10(PxxMult/context.rms_ref**2))
                plt.plot(f, 10*np.log10(PxxRand/context.rms_ref**2))
                plt.show()
                print('Spectrum level for multiplied noise: ' + str(meanMultSpectrumLevel))
                print('Spectrum level for random noise: ' + str(meanRandSpectrumLevel))

            mn = basic.raisedCosineRamps(
                filtMultNoise[0:int(settings["noise_duration"]*context.audio_sr)],
                settings["noise_ramp_duration"]
                )
            fn = f'mult_noTone_bw{bandwidth}_seed{seed}-{i}'
            sound_to_save = basic.addSilence(mn, settings["silence_padding"])
            if save_wav:
                lll = lpn[0:int(settings["noise_duration"]*context.audio_sr)]
                lll = basic.raisedCosineRamps(lll, settings["noise_ramp_duration"])
                lpn_to_save = basic.addSilence(lll, settings["silence_padding"])
                np.save(
                    os.path.join(os.environ["sound_dir"], audio_folder, fn + ".npy"),
                    lpn_to_save
                )
                sf.write(
                    os.path.join(os.environ["sound_dir"], audio_folder, fn + ".wav"),
                    sound_to_save,
                    context.audio_sr
                )
            expt_dict[fn] = sound_to_save

            rn = basic.raisedCosineRamps(
                filtRandNoise[0:int(settings["noise_duration"]*context.audio_sr)],
                settings["noise_ramp_duration"]
            )
            fn = f'rand_noTone_bw{bandwidth}_seed{seed}-{i}'
            sound_to_save = basic.addSilence(rn, settings["silence_padding"])
            if save_wav:
                sf.write(
                    os.path.join(os.environ["sound_dir"], audio_folder, fn + ".wav"),
                    sound_to_save,
                    context.audio_sr
                )
            expt_dict[fn] = sound_to_save

            for toneLevel in settings["tone_levels"]:
                tone = tones.pure(
                    settings["tone_duration"],
                    settings["cf"],
                    level=toneLevel
                )
                tone = basic.raisedCosineRamps(tone, settings["tone_ramp_duration"])

                mt = (tone + mn)
                fn = f'mult_tone_bw{bandwidth}_toneLevel{toneLevel}_seed{seed}-{i}'
                sound_to_save = basic.addSilence(
                    mt,
                    settings["silence_padding"]
                )
                if save_wav:
                    sf.write(
                        os.path.join(os.environ["sound_dir"], audio_folder, fn + ".wav"),
                        sound_to_save,
                        context.audio_sr
                    )
                expt_dict[fn] = sound_to_save

                rt = (tone + rn)
                fn = f'rand_tone_bw{bandwidth}_toneLevel{toneLevel}_seed{seed}-{i}'
                sound_to_save = basic.addSilence(
                    rt, settings["silence_padding"]
                )
                if save_wav:
                    sf.write(
                        os.path.join(os.environ["sound_dir"], audio_folder, fn + ".wav"),
                        sound_to_save,
                        context.audio_sr
                    )
                expt_dict[fn] = sound_to_save

    if save_wav:
        settings["audio_sr"] = context.audio_sr
        settings["rms_ref"] = context.rms_ref
        np.save(
            os.path.join(os.environ["sound_dir"], audio_folder, f"cmr_expt1_settings_seed{seed}.npy"),
            settings
        )

    return expt_dict, settings


if __name__ == "__main__":
    audio_fp = str(sys.argv[1])
    os.makedirs(audio_fp, existok=False)
    print('Hall, Haggard, Fernandes (1984)')
    print('Detection in noise by spectro-temporal pattern analysis.')
    # Settings used in paper
    with context(audio_sr=20000, rms_ref=1e-6):
        overwrite = {
                        "n_trials": 1,
                        "lowpass_for_mod": 10,
                        "tone_levels": np.arange(40, 90, 5),
                        "bandwidths": [1000, 400, 200, 100],
                        "seed": 0
                    }
        expt_dict, settings = expt1(audio_fp, overwrite=overwrite, seed=overwrite['seed'])
    print("~~~~")
