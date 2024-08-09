# Model comparisons

## Model alternatives (Figure 9A-C)

The `config` folder (up a level) contains the various configuration files used for the Fixed, Uniform, Spectral Swap and Stationary model alternatives.

## Neural network comparisons (Figure 9D)

In the paper we compare our model to source separation neural networks. Since these networks output soundwaves, instead of a symbolic scene description like our model, we needed to figure out a way to translate these soundwaves into a perceptual judgment. The code in this folder `comparisons/` shows how to obtain psychophysical judgments from output soundwaves, based on the use of standard sets (see Appendix D).

We used openly available neural networks, which required the following repositories:
- [Asteroid](https://github.com/asteroid-team/asteroid): ConvTasNet trained on Librimix
- [Google Research: Source separation](https://github.com/google-research/sound-separation): TCDN++ trained on FUSS, MixIT trained on YFCC100m
- [demucs](https://github.com/facebookresearch/demucs): temporal/spectral bi-U-Net using a transformer, trained on MUSDB plus 800 additional songs
- [SepFormer](https://huggingface.co/speechbrain/sepformer-whamr16k): convolutional encoder + transformer masking network + convolutional decoder, trained on WHAMR!
- [Open-Unmix for Pytorch](https://github.com/sigsep/open-unmix-pytorch): OpenUnmix networks (preprint only)

We also trained the TCDN++ network with samples from our generative model (limited to contain 1-4 sources), using [models/dcase2020_fuss_baseline/train_model.py](https://github.com/google-research/sound-separation/blob/master/models/dcase2020_fuss_baseline/train_model.py) from the `google/sound-separation` repository.

## Dissimilarity computation and statistics

`analysis.r` contains code for calculating model-human correlations, as well as obtaining bootstrap statistics and confidence intervals.