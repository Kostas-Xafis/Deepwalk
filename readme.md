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

## Usage

1. Single training run:

    ```bash
    python training.py --exec train
    ```

2. Grid search:

    ```bash
    python training.py --exec grid
    ```

3. Adjust parameters, for example:

    ```bash
    python training.py --window_size 5 --walk_length 3
    ```

## Parameter Guide

• window_size (int, default=3): Context window size.  
• walk_length (int, default=window_size): Steps per walk.  
• num_walks (int, default=1000): Walks per node.  
• embedding_dim (int, default=12): Dimension of embeddings.  
• epochs (int, default=500): Training epochs.  
• exec (str, default="grid"): Execution mode ("grid" or "train").