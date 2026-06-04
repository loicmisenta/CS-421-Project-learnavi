# Lernnavi Disengagement Prediction

This repository contains the code and [report](./Report.pdf) for a machine learning project on learner disengagement prediction on **Lernnavi**, a Swiss high-school practice platform. Unlike standard MOOCs, Lernnavi has no fixed course structure, no completion label, and no common end date. For this reason, disengagement cannot be modeled as simple course dropout. Instead, we define disengagement through platform activity patterns and study it with a two-stage prediction pipeline.

## Project Overview

The project separates disengagement into two related tasks:

1. **Stage 1: Early dabbler prediction**  
   Predict whether a learner returns after their first three weeks of activity. This stage separates short-lived users, or *early dabblers*, from learners who continue using the platform.

2. **Stage 2: Sustained interruption prediction**  
   Among continuing learners, predict whether a learner is about to enter a sustained interruption, defined as at least four consecutive inactive weeks. Interruptions overlapping with July or August are handled separately because they often correspond to school holidays rather than meaningful disengagement.

## Methods

We compare several model families across the two stages:

- **Logistic Regression** as an interpretable baseline for tabular prediction.
- **LightGBM** as the main tabular model for heterogeneous behavioral, temporal, performance, demographic, and contextual features.
- **LSTM** models for Stage 2, where weekly learner histories are represented as 12-week behavioral sequences.

The experiments include feature-set ablations, calendar-feature ablations, data augmentation tests, trajectory clustering, and fairness analysis across gender and canton groups.

## Main Findings

The results show that disengagement on Lernnavi is both behavioral and contextual. Activity patterns, regularity, performance, and topic breadth are useful predictive signals, but calendar and contextual variables are especially important. In Stage 2, the best model reaches an AUC-ROC of `0.9125`, while removing calendar features causes a large performance drop.

However, the fairness analysis shows that stronger predictive performance does not automatically imply more reliable predictions for all groups. Canton-level differences are especially large and should be interpreted carefully because some cantons contain few users.

## Intended Use

The models should be viewed as **low-stakes decision-support tools**, not as automatic intervention systems. A responsible deployment would use predictions to support teachers or platform developers by highlighting interpretable risk signals, while keeping human judgment, calendar context, and fairness monitoring in the loop.



## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r environment.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

The LSTM notebooks run on CPU, but CUDA is much quicker if you have an NVIDIA GPU. If you want GPU acceleration, install the CUDA-compatible PyTorch build for your machine from the official PyTorch instructions instead of the default CPU wheel.

## How to Run

Run the notebooks from the project root, in this order:

1. `preprocess_data.ipynb`  
   Cleans raw `data/` files and writes the reusable datasets and feature lists to `outputs/`.

2. `stage_1_analysis.ipynb`  
   Trains and evaluates Stage 1 models for early dabbler vs continuing learner prediction.

3. `stage_1_fairness.ipynb`  
   Audits Stage 1 reliability across demographic and contextual groups.

4. `stage_2_analysis.ipynb`  
   Trains Stage 2 LightGBM and LSTM models for sustained interruption prediction.

5. `stage_2_data_augmentation.ipynb`  
   Tests Stage 2 augmentation methods such as noise, masking, mixup, and SMOTE.

6. `stage_2_clustering.ipynb`  
   Builds behavioral trajectory clusters and exports cluster summaries.

7. `stage_2_fairness.ipynb`  
   Audits Stage 2 fairness, including comparisons with and without demographic features.

## Repository Structure

```text
.
|-- data/                 Raw Lernnavi data and study files
|-- outputs/              Cleaned datasets, model results, plots, and cluster outputs
|-- src/                  Shared preprocessing, modeling, metrics, fairness, clustering, and augmentation helpers
|-- old/                  Experimental first version kept for project history
|-- new/                  Empty/unused workspace folder
|-- preprocess_data.ipynb Main preprocessing notebook
|-- stage_1_*.ipynb       Current Stage 1 analysis and fairness notebooks
|-- stage_2_*.ipynb       Current Stage 2 analysis, augmentation, clustering, and fairness notebooks
|-- Report.pdf            Final project report
|-- environment.txt       Python dependencies
```

