# Sentence embeddings t-SNE cluster visualization

## Installation

### (Optional) Create and activate a new virtual environment

You will need a Python3 environment.

For instance, using conda:

```
conda create --name tsne-cluster-vis python=3.9
conda activate tsne-cluster-vis
```

### Install Python 3 libraries

```
pip3 install -r requirements.txt
```

The libraries are:

matplotlib
mplcursors
nltk
numpy
pandas
scipy
sentence-transformers
scikit-learn

## Usage

### (Optional) Choose input data

Either use one of the provided text files, or use your own text file. Provide the filename in line 156.

```
f = open("data/{your-text-file}.txt", "r")
 ```

### Run script

```
python3 compare_clusters.py
```

### Interact with the visualization

1. Lasso the first cluster by clicking and dragging to draw a circle around it.

1. Press the "A" key to lock in the first cluster.

1. Lasso the second cluster.

1. Press the "B" key to lock in the second cluster.

1. Press the "F" key to finish the selection and populate the augmenting plots.

1. Examine the visualization plots. You can hover over points in all plots to see the sentence text.

1. Press the "X" key to close the visualization.
