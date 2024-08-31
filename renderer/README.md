# Renderer (a.k.a., synthesizer)

This code implements the differentiable synthesizer that transforms the latent variable description sampled in `model` into a sound (and cochleagram). A differentiable renderer was required so that we could use stochastic variational inference to infer scene descriptions. For an overview of how sounds are synthesized see Section 2.2.3., *Generative model: Likelihood* in the paper. More details are in Section A.5., *Generative model: Likelihood*.

- `excitation`: modules to differentiably generate whistles, harmonic, or noise sounds, i.e., excitations which are then amplitude-modulated or filtered
- `am_and_filters`: modules to differentiably amplitude-modulate or filter the spectrum of a sound
- `cochleagrams`: modules to differentiably create cochleagrams from sounds. Thank you to Jenelle Feather for her [chcochleagram](https://github.com/jenellefeather/chcochleagram) package and Dan Ellis for his [gammatonegram](https://www.ee.columbia.edu/~dpwe/LabROSA/matlab/gammatonegram/) implementation.
- `trimmer`: modules to trim sounds to reduce memory usage during rendering, while (1) maintaining batching for efficient variational inference and (2) keeping track of the timing of events in source sounds in order to recombine them to create the scene sound
- `util`: a variety of useful signal processing functions. Thank you to Josh McDermott who provided useful MATLAB code as part of his [Sound Texture Toolbox](https://mcdermottlab.mit.edu/downloads.html)