# Project README

## Table of Contents

- [Project README](#project-readme)
  - [Table of Contents](#table-of-contents)
  - [Summary](#summary)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Parameter Guide](#parameter-guide)

## Summary

This project utilizes a CBOW-based model to embed the Karate Club network, allowing both hyperparameter tuning via grid search and straightforward single-run experiments for quick prototyping.

## Installation

1. Clone the repository.

    ```bash
    git clone https://github.com/Kostas-Xafis/Deepwalk.git
    cd ./Deepwalk
    ```

2. Create a Python virtual environment and activate it.

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

> [!IMPORTANT]
The code has not been tested in a Colab environment and therefore it might not work.

## Usage

1. Single training run (w/ default parameters):

    ```bash
    python deepwalk.py
    ```

2. Grid search:

    ```bash
    python deepwalk.py --exec grid
    ```

3. Adjust parameters, for example:

    ```bash
    python deepwalk.py --window_size 5 --walk_length 3
    ```

## Parameter Guide

```markdown
--window_size:    Size of the context window (default: 3).  
--walk_length:    Length of the random walk (default: window_size).  
--num_walks:      Number of random walks to generate for each node (default: 1000).  
--embedding_dim:  Dimension of the node embeddings (default: 12).  
--batch_size:     Batch size for training (default: 256).
--epochs:         Number of epochs to train the model (default: 500).  
--exec:           Execution mode 'grid'|'train' (default: train).
--verbose:        Output any information while training (default: False).
```
