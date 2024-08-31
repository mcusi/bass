
import os
import numpy as np
import soundfile as sf

import psychophysics.generation.tones as tones
import psychophysics.generation.basic as basic  
import psychophysics.generation.noises as noises
from util import context, manual_seed

"""
Warren, R. M., Obusek, C. J., & Ackroff, J. M. (1972).
Auditory induction: Perceptual synthesis of absent sounds. Science, 176(4039), 1149-1151. 
"""


def toneContinuity(path, fig, masked, maskers, settings, seed, save_wav=True):

    sr = context.audio_sr
    pad = settings["pad"]
    durations = masked['duration']

    # continuity, separate stim
    expt_dict = {}
    print(f'Figure {fig}: continuity, separate stimuli')
    overlapSamples = int(durations['ramp']*sr)
    for f in masked['freqs']:
        for L in masked['levels']:
            for j in range(settings["nRepetitions"]+1):
                if j == 0 or j == settings["nRepetitions"]:
                    target = basic.raisedCosineRamps(
                        tones.pure(
                            0.5*durations['tone'] + durations['ramp']*2,
                            f,
                            level=L,
                            phase=np.random.rand()*2*np.pi
                        ), durations['ramp']
                    )
                    silence_for_target = basic.silence(0.5*durations['tone'])
                    if j == settings["nRepetitions"]:
                        target = np.concatenate((target, silence_for_target))
                else:
                    target = basic.raisedCosineRamps(
                        tones.pure(
                            durations['tone'] + durations['ramp']*2,
                            f,
                            level=L,
                            phase=np.random.rand()*2*np.pi
                        ), durations['ramp']
                    )  # also see Warren Bashford Healhy & Brubaker re: phase
                _stimulus = silence_for_target if j == 0 else stimulus
                stimulus = overlapSections(
                    _stimulus,
                    [target, maskers[j]],
                    overlapSamples
                )
            stimulus = np.concatenate((maskers[-1], stimulus))
            stimulus = basic.addSilence(stimulus, [pad, pad])
            fn = f'continuity_f{f:04}_l{L:03}'
            if save_wav:
                sf.write(
                    os.path.join(path, f"{fn}_seed{seed}.wav"),
                    stimulus,
                    sr
                )
            expt_dict[fn] = stimulus

    return expt_dict


def toneMasking(path, fig, masked, masker, settings, seed, save_wav=True):

    sr = context.audio_sr
    pad = settings["pad"]
    durations = masked['duration']

    def addParts(stimulus, masker):
        # make stimulus and masker the same length in order to add them
        stimulus = stimulus[:np.min([len(stimulus), len(masker)])-1]
        if len(stimulus) < len(masker):
            stimulus = np.concatenate((stimulus, np.zeros(len(masker) - len(stimulus))))
        stimulus = stimulus + masker
        return stimulus

    print(f'Figure {fig}: masking, separate stimuli')
    expt_dict = {}
    spacing = basic.silence(durations['tone'])
    for f in masked['freqs']:
        for L in masked['levels']:
            # Instead of random, always start the tone at a half-tone-duration
            # for ease of analysis.
            stimulus = basic.silence(durations['tone']/2.)
            for i in range(settings["nRepetitions"]):
                target = basic.raisedCosineRamps(
                    tones.pure(
                        durations['tone'] + durations['ramp']*2,
                        f,
                        level=L,
                        phase=np.random.rand()*2*np.pi
                    ), durations['ramp']
                )
                stimulus = np.concatenate((stimulus, target, spacing))
            fn = f'masking_f{f:04}_l{L:03}'
            stimulus = basic.addSilence(addParts(stimulus, masker), pad)
            if save_wav:
                sf.write(
                    os.path.join(path, f"{fn}_seed{seed}.wav"),
                    stimulus,
                    sr
                )
            expt_dict[fn] = stimulus

    return expt_dict


def rms(signal):
    return 1.*np.sqrt(np.mean(np.square(signal)))


def expt3_settings(overwrite={}):
    settings = {
        # Overall sound parameters
        "nRepetitions": 3,
        "pad": 0.05,
        # Common parameters to masker and masked
        "duration": 0.300,
        "rampDuration": 0.010,
        "tone_levels": [44, 48, 52, 56, 60, 64, 68, 72, 76, 80],
        "tone_freqs": [250, 500, 1000, 2000, 4000],
        # masker parameters
        "filter_type": "taps",
        "maskerLevel": 80,
    }
    for k in settings.keys():
        if k in overwrite.keys():
            settings[k] = overwrite[k]
    return settings


def taps(audio_sr=20000):
    """ FIR Taps for bandstop
        Between 700 and 1400 with attenuation=-40dB at 1000Hz
        Used http://t-filter.engineerjs.com/
        Settings:
            lf, hf, gain, ripple
            0, 700, 1, 5dB
            900, 1200, 0.01, 1dB
            1400, 10000, 1, 5dB
    """
    if context.audio_sr != audio_sr:
        raise Exception(f"You need new taps, audio sampling rates don't match. {context.audio_sr}!={audio_sr}")
    return np.array([-0.015577274517279938,0.030533050118549492,-0.00825748237950438,-0.01397898723049071,-0.005071524104298594,0.005345632768326657,0.01091274780911486,0.010629681867536472,0.00664507440289955,0.0018868018865685363,-0.0015643559830007718,-0.002710225944403154,-0.001660468683792251,0.0007143673690190915,0.003605180329849864,0.006009429078670933,0.007570064094989503,0.008028088023137302,0.007486523646885843,0.006135465589200283,0.0042614336903648966,0.0021645604317968676,0.00012983562263495785,-0.0016586270229433683,-0.0030285885678243687,-0.0038626394682541045,-0.004133703806009184,-0.003854198078675574,-0.0031824632410976813,-0.0022463721804199506,-0.0012481445418198514,-0.0003667230564791033,0.00022483555521972366,0.0004316647766640746,0.0002481951775651643,-0.0002303097041761057,-0.0008717350132457248,-0.0014851199838875763,-0.0018648040730444195,-0.0018286774066965097,-0.0012487144914096885,-0.00013199463849865894,0.001434807373097817,0.0032535351272135384,0.005066245342910189,0.0065610845244960265,0.0074340227882452444,0.007443152944968865,0.006480462975399647,0.0045726204433676285,0.0019047898482337582,-0.0012187804392649229,-0.004407706171579182,-0.007229924352623322,-0.009297642287286474,-0.010306178610846421,-0.01011183232417355,-0.00874797041667745,-0.006430196710254523,-0.003511725676647493,-0.000431979137890738,0.0023622040197591168,0.0044686576123568284,0.005620922865514489,0.005734002193861438,0.00493060081072157,0.0035170131283467168,0.00190909037364047,0.0005682754286682201,-0.00009034008697832827,0.00022403252232110546,0.0015993575260788008,0.0038787578811510055,0.0066603176480735105,0.009361847701843935,0.011301225993082448,0.011827681869367042,0.010436516272907636,0.0068849162040543165,0.0012703298482227615,-0.005946142769871973,-0.01395932809730549,-0.021706372269001278,-0.028003506768277,-0.031720312854306494,-0.03196320856207601,-0.028236556991532782,-0.020544773895244943,-0.009447440260228432,0.003979610518983279,0.018243646603109363,0.03161401890517911,0.04234579252855491,0.04891642765402422,0.050248655404528254,0.045880454078452156,0.03605047444784346,0.021697383906201784,0.004360626453557126,-0.013997493131864403,-0.031227294126824606,-0.04527482719553493,-0.05444586333409544,0.9423687507347286,-0.05444586333409544,-0.04527482719553493,-0.031227294126824606,-0.013997493131864403,0.004360626453557126,0.021697383906201784,0.03605047444784346,0.045880454078452156,0.050248655404528254,0.04891642765402422,0.04234579252855491,0.03161401890517911,0.018243646603109363,0.003979610518983279,-0.009447440260228432,-0.020544773895244943,-0.028236556991532782,-0.03196320856207601,-0.031720312854306494,-0.028003506768277,-0.021706372269001278,-0.01395932809730549,-0.005946142769871973,0.0012703298482227615,0.0068849162040543165,0.010436516272907636,0.011827681869367042,0.011301225993082448,0.009361847701843935,0.0066603176480735105,0.0038787578811510055,0.0015993575260788008,0.00022403252232110546,-0.00009034008697832827,0.0005682754286682201,0.00190909037364047,0.0035170131283467168,0.00493060081072157,0.005734002193861438,0.005620922865514489,0.0044686576123568284,0.0023622040197591168,-0.000431979137890738,-0.003511725676647493,-0.006430196710254523,-0.00874797041667745,-0.01011183232417355,-0.010306178610846421,-0.009297642287286474,-0.007229924352623322,-0.004407706171579182,-0.0012187804392649229,0.0019047898482337582,0.0045726204433676285,0.006480462975399647,0.007443152944968865,0.0074340227882452444,0.0065610845244960265,0.005066245342910189,0.0032535351272135384,0.001434807373097817,-0.00013199463849865894,-0.0012487144914096885,-0.0018286774066965097,-0.0018648040730444195,-0.0014851199838875763,-0.0008717350132457248,-0.0002303097041761057,0.0002481951775651643,0.0004316647766640746,0.00022483555521972366,-0.0003667230564791033,-0.0012481445418198514,-0.0022463721804199506,-0.0031824632410976813,-0.003854198078675574,-0.004133703806009184,-0.0038626394682541045,-0.0030285885678243687,-0.0016586270229433683,0.00012983562263495785,0.0021645604317968676,0.0042614336903648966,0.006135465589200283,0.007486523646885843,0.008028088023137302,0.007570064094989503,0.006009429078670933,0.003605180329849864,0.0007143673690190915,-0.001660468683792251,-0.002710225944403154,-0.0015643559830007718,0.0018868018865685363,0.00664507440289955,0.010629681867536472,0.01091274780911486,0.005345632768326657,-0.005071524104298594,-0.01397898723049071,-0.00825748237950438,0.030533050118549492,-0.01557727451727993]) 


def expt3(audio_folder, overwrite={}, save_wav=True):
    """ Band-reject (700-1400 Hz) pink noise masker
        with max attenuation (40 dB) at 1000 Hz
    """
    seed = overwrite.get('seed', 0)
    manual_seed(seed)

    settings = expt3_settings(overwrite)
    sr = context.audio_sr
    rms_ref = context.rms_ref
    settings["audio_sr"] = sr
    settings["rms_ref"] = rms_ref
    path = os.path.join(os.environ["sound_dir"], audio_folder, "")
    if save_wav:
        os.makedirs(path, exist_ok=True)
        np.save(path + f"pi_expt3_settings_seed{seed}.npy", settings)

    maskerRMS = rms_ref*np.power(10, settings["maskerLevel"]/20.)
    masked = {
        'duration': {
            'ramp': settings["rampDuration"],
            'tone': settings["duration"]
        }, 
        'levels': settings["tone_levels"],
        'freqs': settings["tone_freqs"]
    }

    n_maskers = settings["nRepetitions"] + 2
    maskers = [
        noises.pink(settings["duration"], settings["maskerLevel"]) 
        for _ in range(n_maskers)
    ]
    maskers = [np.convolve(masker, taps(), mode='valid') for masker in maskers]
    maskers = [
        basic.raisedCosineRamps(masker*maskerRMS/rms(masker), settings["rampDuration"]) 
        for masker in maskers
    ]
    expt_dict = toneContinuity(path, 3, masked, maskers, settings, seed, save_wav=save_wav)

    masker = noises.pink(
        settings["duration"]*2*settings["nRepetitions"],
        settings["maskerLevel"]
    )
    masker = np.convolve(masker, taps(), mode='valid')
    masker = basic.raisedCosineRamps(masker*maskerRMS/rms(masker), settings["rampDuration"])
    mask_expt_dict = toneMasking(path, 3, masked, masker, settings, seed, save_wav=save_wav)
    expt_dict.update(mask_expt_dict)

    return expt_dict, settings


def overlapSections(stimulus, additions, overlapSamples):
    for add in additions:
        stimulus = np.concatenate((
            stimulus, np.zeros(len(add)-overlapSamples)
        ))
        stimulus[-len(add):] += add
    return stimulus


if __name__ == "__main__":
    import sys
    print('Warren, Obusek, & Ackroff (1972)')
    print('Auditory induction: Perceptual synthesis of absent sounds')
    print('Warning: takes a while...')
    audio_folder = str(sys.argv[1])
    print("Saving in ", audio_folder)
    expt3(audio_folder)
