# ijcnlp17_emo
Code to replicate experiments in the IJCNLP 2017 paper "Modelling Representation Noise in Emotion Analysis using Gaussian Processes" (to appear)

## Requirements

### Data and resources

Download the "Affective Text" dataset and put the gzipped file in the "data" folder:

https://web.eecs.umich.edu/~mihalcea/downloads.html

Download GloVe word embeddings, the version trained on Wikipedia and Gigaword ("glove.6B.zip") and put it in the "folder":

https://nlp.stanford.edu/projects/glove/

### Tools

All code uses Python, we recommend you use virtualenv to define an isolate virtual environment for the experiments.

Install NLTK

> pip install nltk

Install scikit-learn

> pip install scikit-learn

Install GPflow

> pip install gpflow

## Configuration and Preprocessing

All code assume the repository is located in your home folder with its original name, so you can run the scripts from anywhere in your command line. If the repository is somewhere else and/or you cloned it with a different name, change the `MAIN_FOLDER` variable inside the `bin/config.sh` file.

Once you're set, you can run the preprocessing script:

> `bin/preprocess.sh`

This will unpack the dataset and the word embeddings, as well as formatting the dataset in a friendlier format for the experiment scripts. It will also generate the data splits for 10-fold cross-validation, in the `splits` folder.

## Running the experiments

The `bin/run_all.sh` script replicates all models in the paper with 10-fold cross-validation, saving predictions in the `preds` folder and final scores in the `results` folder. This can take a long time to finish though.

A more fine-grained replication is available using the `bin/run.py` script, which should be tried first for testing purposes. A single run of this script will train a `RidgeCV` regression model on the first fold. Options for the remaining models are available, use the '--help' flag for a description.

## Collecting final results

Any instances of `bin/run.py` will save the scores (Pearson's r and NLPD) in the `results` folder. The `bin/collect_results.py` script will gather all the scores in that folder and summarise them in the output. If you ran all models and folds, this should give you the same numbers published in the paper. However, you can also run it if you have partial results as well.