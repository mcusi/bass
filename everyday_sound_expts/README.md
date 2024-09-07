# Everyday sound experiments

For our everyday sound experiments, we used the FUSS dataset, which contains a set of mixtures and their premixture sounds. The dataset can be downloaded [here](https://zenodo.org/record/3694384/files/FUSS_ssdata.tar.gz?download=1). FUSS does not come labeled with categories, so we traced the labels from [FSD50K](https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1) to enable our analysis.

We first sampled a set of fifty sound mixtures (`sample_mixtures.py`), which we performed inference on. These model inferences, along with the selected mixtures and premixture sounds, go into creating Experiment 1 (`make_experiment_one.py`) and Experiment 2 (`make_experiment_two.py`).

Example HTML for the experiment setup can be viewed on our [website](https://mcdermottlab.mit.edu/mcusi/bass/fig9.html).