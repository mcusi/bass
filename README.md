# bass

Code for Bayesian auditory scene synthesis ([project page](https://mcdermottlab.mit.edu/mcusi/bass/)). 

The corresponding publication is:
```
Maddie Cusimano, Luke Hewitt, Josh H. McDermott. (2024). Listening with generative models. _Cognition_. In press.
```
The READMEs in this repository, as well as the code, contain references to sections, equations and figures, which correspond to the numbering in this publication.

## Directory overview

- `model/`: The generative model. See sections 2.2.1 and 2.2.2, and Appendix A.
- `renderer/`: The renderer/synthesizer which transforms a scene description into a sound. See section 2.2.3 and section A.5.
- `inference/`: enumerative and sequential inference algorithms (with a lot of platform-specific code). See section 2.3 and Appendix B. 
- `psychophysics/`: Code for experiments with classic illusions, see Appendix C.
    - `generation/`: Generation of stimuli
    - `hypotheses/`: Creation of initial hypotheses for enumerative inference
    - `analysis/`:  Analysis of model inferences to get experiment results, and plot creation
    - `comparisons/`: Code to compute human-model dissimilarity, see section 2.6 and Appendix D.
- `everyday_sound_expts/`: Create and analyze everyday sound experiments, see section 2.8 and Appendix F. 