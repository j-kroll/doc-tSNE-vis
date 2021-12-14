# Document t-SNE cluster visualization

## Instructions

### (Optional) Create and activate a new virtual environment

You will need a Python3 environment.

For instance, using conda:

```
conda create --name tsne-cluster-vis python=3.9
conda activate tsne-cluster-vis
```

### Install Python 3 libraries:

```
pip3 install -r requirements.txt
```

The libraries are:

matplotlib
mplcursors
nltk
numpy
pandas
re
scipy
seaborn
sentence_transformers
sklearn
textwrap

### Choose input data

Either use one of the provided text files, or use your own text file. Provide the filename in line 166.

```
f = open("{your-text-file}.txt", "r")
 ```

### Run script

```
python3 tsne_lasso.py
```

### Interact with the visualization

1. Lasso the first cluster.

1. Press the "A" key to lock in the first cluster.

1. Lasso the second cluster.

1. Press the "B" key to lock in the second cluster.

1. Press the "F" key to finish the selection.

1. Examine the visualization plots.

1. Press the "X" key to close the visualization.
