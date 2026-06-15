# DSLR — DataScience x Logistic Regression

A from-scratch implementation of logistic regression to sort Hogwarts students into their houses, inspired by the 42 school project. The classifier is trained with one-vs-all logistic regression using gradient descent, and comes with data exploration tools built without high-level statistical libraries.

## Project structure

```
.
├── datasets/
│   ├── dataset_train.csv   # Labeled training data
│   └── dataset_test.csv    # Unlabeled test data
├── describe.py             # Statistical summary (reimplements pandas describe)
├── histogram.py            # Score distribution per house
├── scatter_plot.py         # Most correlated feature pair
├── pair_plot.py            # Full feature pair plot
├── train.py                # Train the logistic regression model
├── predict.py              # Predict house from trained weights
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Describe the dataset

Prints count, mean, std, min, quartiles, max, range, and IQR for every numeric feature — computed without `pandas.describe()`.

```bash
python describe.py datasets/dataset_train.csv
```

### 2. Visualize the data

**Histogram** — shows score distributions per house to identify homogeneous features:
```bash
python histogram.py datasets/dataset_train.csv
```

**Scatter plot** — automatically finds and plots the two most correlated features:
```bash
python scatter_plot.py datasets/dataset_train.csv
```

**Pair plot** — full pairwise feature grid colored by house:
```bash
python pair_plot.py datasets/dataset_train.csv
```

### 3. Train the model

Trains a one-vs-all logistic regression and saves weights to `weight.csv`.

```bash
# Batch gradient descent (default)
python train.py gradientdescent datasets/dataset_train.csv

# Stochastic gradient descent
python train.py SGD datasets/dataset_train.csv

# Mini-batch gradient descent (batch size 64)
python train.py minibatch datasets/dataset_train.csv
```

### 4. Predict

Applies the saved weights to a dataset and writes predictions to `houses.csv`. If the dataset includes the true house labels, also prints accuracy.

```bash
python predict.py datasets/dataset_test.csv weight.csv
```

## How it works

- **Preprocessing**: drops low-information features (`Arithmancy`, `Care of Magical Creatures`), fills missing values with the training mean, then standardizes all features.
- **Training**: one sigmoid classifier is trained per house (Ravenclaw, Slytherin, Gryffindor, Hufflepuff) via gradient descent. Three optimizers are available: full batch, SGD, and mini-batch.
- **Prediction**: each student is assigned the house whose classifier returns the highest score (argmax over the four sigmoid outputs).

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn` (accuracy score in `predict.py` only)
