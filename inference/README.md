# Inference

We use two kinds of inference in our model in order to either emulate:
1. psychophysics experiments where listeners are typically meant to report what they hear out of a few options (Figures 6-8)
2. more everyday, unconstrained listening situations (Figure 10)

We call these enumerative and sequential inference respectively. Appendix B describes the inference procedures in detail.

## Enumerative inference

The main inference function here is `serial.solo_optimization`. 
This function takes in an observed sound and a potential hypothesis for the scene description. 
We initialize a guide distribution $q$ based on the `hypothesis` (using `Scene.hypothesize`), and optimize $q$ using stochastic variational inference (see `optimize.variational_update`, `inference.metrics.importance_weighted_bound`). 
We use importance sampling to calculate the evidence lower bound (ELBO) (see `metrics.elbo`), as an estimate of the marginal probability.

During enumerative inference, we call this function once per hypothesis. We can compare the ELBO across hypotheses.
Code in `psychophysics.analysis` uses these probabilities to generate the model psychophysics results in Figures 6-8.

This method is described in section B.3, _Enumerative inference_. 

## Sequential inference

The overarching inference procedure is coordinated in `distributed.py`, with the main function `iterative`.  `distributed` is highly adapted to the SLURM scheduling system that we use for parallel computing. *Please note that much of the complexity in the code results from guarding against failures in the scheduler.*

`iterative` takes in an observed sound and uses a set of neural network proposals for the events in possible scenes (see Events proposal below).
Inspired by sequential Monte Carlo, `iterative` will proceed through progressively longer durations of the observed sound until it constructs a scene description that can explain the whole sound.

It first constructs a set of scene hypotheses from the event proposals (see Source construction below).
Then it launches parallel jobs to optimize and assess each of these hypotheses, ultimately through `distributed.solo_optimization` (similar to in `serial`). 
It also launches a job to select the best scene hypotheses for the current sound clip (`analyze_complete_round`) after hypothesis optimization is complete.
These best scene hypotheses are passed to the next call of `iterative`, where they are combined with more event proposals.
This process repeats until the entire sound is accounted for.

We describe each step of sequential inference more fully below, and in Appendix B. The results of sequential inference were used for our everyday sound experiments (Figure 10, listen to examples [here](https://mcdermottlab.mit.edu/mcusi/bass/recognizable.html)).

### Events proposal

We built the event proposal network on top of [detectron2](https://github.com/facebookresearch/detectron2/tree/main/detectron2). We trained this network on samples from the generate model (obtained with `dataset.py`).

In contrast to traditional image segmentation, our event proposal network outputs 'candidate events', each of which is a set of latent variables describing an event in our generative model.
We formatted these latent variables as network outputs in `dataloaders.py`, and adapted the network output architecture (see `alt_mask_opts.py` and `soft_masks.py`) .

`inference.py` will output candidate events, using the train and inference configuration that we used for the preprint. 

See section B.2.1, _Event proposals via amortized inference -- segmentation network_.

### Source construction

The function that coordinates source construction is `construction.create_new_scene_hypotheses`. Key steps include:
  - `proposals.add_event_to_scene`: generates new hypotheses using candidate events, combining them with scenes from the previous sequential inference round (if beyond the first round)
  - `sequential_heuristics` cuts down on the number of hypotheses. These heuristics are fairly simple, mainly based on: 
      - prioritizing smaller scene descriptions (e.g., fewer sources) 
      - minimizing overlap between candidate events (i.e., avoiding duplicate events)
      - prioritizing scenes with events which the neural network rated with high confidence  
      
See section B.2.2, _Source construction_, for full details. On the last round, we also tried `cleanup` proposals (see section B.2.4, _Cleanup proposals_).

### Hypothesis optimization

We described this above step in Enumerative inference. 
One added feature for sequential inference is that we first optimized all hypotheses for a small number of iterations (`earlystop`).
This allowed us to choose only the best scenes after `earlystop` for further optimization (`distributed.continue_after_earlystop`), reducing our computational burden further. See section B.2.3, _Hypothesis optimization and scene selection (sequential inference)_.

### Scene selection

After hypothesis optimization, the hypotheses are compared in `distributed.analyze_complete_round`. This function then takes the top hypotheses (`n_to_keep`) and passes them to the next round.

## Source priors

The `metasource` module contains the code used to fit the meta-source parameters of the source priors. See Appendix A, section A.3, _Generative model: Source priors_. These methods operate on `Scenes` objects (see 'Fitting source priors' in `model/README.md`).
