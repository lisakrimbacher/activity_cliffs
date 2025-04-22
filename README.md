This is the GitHub repository corresponding to the Bachelor's Thesis "Leveraging Contrastive Learning for Improved Activity-Based Classification of Activity Cliff Molecules" by Lisa Krimbacher (2025). 

# Project Structure
```
activity_cliffs/
├── code/                           # training & evaluation scripts
│   ├── main.py                     # entry point
│   ├── model.py                    # custom MLP class, RF helper function
│   ├── preprocessing.py            # data preprocessing functions
│   ├── cliffs.py                   # activity cliffs helper class, altered from https://github.com/molML/MoleculeACE
│   ├── data_prep.py                # preprocessing helper functions, altered from https://github.com/molML/MoleculeACE
|   ├── cliff_visualizations.ipynb  # visualization scripts
│   └── figures/                    # SVG plots 
├── data/                           # processed datasets 
│   └── CHEMBL*/                    # datasets taken from https://github.com/molML/MoleculeACE, train/val/test splits
├── models/                         # model checkpoints 
├── results/                        # logged metrics
└── LICENSE                         # MIT license 
```

## Reproducing Experiments
All experiments of the thesis can be reproduced by cloning the repository and running `main.py`. 

### Select Dataset
```
dataset_folder = "CHEMBL234_Ki"
```
This is the dataset described in the thesis, alternatively, `"CHEMBL214_Ki"` can be used.

### Choose Model
```
train_eval_rf = True
```
With this configuration, a Random Forest with 100 trees is trained on the selected dataset. The results are printed to the console and automatically saved.

```
train_eval_rf = False
use_contrastive_learning = False
```
Instead of the Random Forest, this flag combination leads to the training and evaluation of a Multi-layer Perceptron (named MLP BCE in the thesis).

```
train_eval_rf = False
use_contrastive_learning = True
use_cosine_sim = True
```
Here, a Multi-layer Perceptron is trained and tested using a Triplet Loss with the Cosine Similarity (named MLP Triplet in the thesis).

```
train_eval_rf = False
use_contrastive_learning = True
use_cosine_sim = False
```
Lastly, this configuration enables the training and evaluation of a Multi-layer Perceptron using a Triplet Loss with the Manhattan Distance (results are not explicitly reported in the thesis).

