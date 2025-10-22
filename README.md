# An Interpretable Stacking Ensemble with TabNet-Enhanced Features for Breast Cancer Diagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Purpose:** This document provides the necessary code and instructions to reproduce the results presented in the research paper titled *"An Interpretable Stacking Ensemble with TabNet-Enhanced Features for Breast Cancer Diagnosis"* .

>**Abstract**:Addressing the challenge of balancing accuracy and interpretability in breast cancer diagnosis, this work introduces a novel interpretable stacking ensemble. It utilizes TabNet as a deep feature extractor feeding differentiated feature sets to XGBoost and LightGBM base learners in a dual-pathway architecture, integrated by Logistic Regression. Validated via nested cross-validation on the WDBC dataset, the model achieves high accuracy (97.5% ± 1.3%), outperforming baselines. An end-to-end SHAP framework ensures multi-level interpretability aligned with clinical knowledge, demonstrating a path towards trustworthy AI decision support.
---

## Key Features & Contributions

* **Novel Use of TabNet:** Leverages TabNet as a sophisticated deep feature extractor.
* **Differentiated Stacking Architecture:** Employs a dual-pathway stacking approach with heterogeneous base learners (XGBoost, LightGBM) trained on distinct feature sets.
* **End-to-End Interpretability:** Constructs a multi-level interpretability framework using SHAP.

---

## Model Architecture Overview

The model uses a three-layer stacking approach:
1.  **Layer 1:** TabNet extracts various features (embeddings, attention-weighted, etc.).
2.  **Layer 2:** XGBoost and LightGBM act as base learners, trained on different feature sets derived from TabNet.
3.  **Layer 3:** Logistic Regression serves as the meta-learner, combining predictions from the base learners.

<img width="905" height="508" alt="overall" src="https://github.com/user-attachments/assets/9324dc2c-f23f-423b-8a0d-8196f26defdf" />


---

## Dataset

* **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset, sourced from the **UCI Machine Learning Repository**.
    * **Link:** [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
* **Preprocessing:**
    * Diagnosis labels ('M'/'B') are encoded (1/0).
    * Numerical features are standardized using `RobustScaler`.
    * 'ID' column is dropped.

---

## Experimental Setup

* **Software Environment:** All experiments were conducted in a Python 3.8.6 (64-bit) environment.
* **Hardware:** The hardware used consisted of an Intel(R) Core(TM) i5-8250U CPU and an Intel(R) UHD Graphics 620 GPU, running on a 64-bit operating system.
* **Reproducibility:** To ensure reproducibility, all experiments were conducted with a fixed random seed (42).
* **Core Libraries:** The implementation was based on key libraries including Pandas (2.0.3), NumPy (1.2.4.3), Scikit-learn (1.3.2), PyTorch (2.4.1+cpu), PyTorch-TabNet (4.1.0), XGBoost (2.1.4), and LightGBM (4.6.0).


---

## Ablation Study

To systematically evaluate the contribution of each key component in our proposed stacking ensemble model, a comprehensive ablation study was performed. This involved training and evaluating different configurations of the model by selectively combining components.

**Configurations Tested:**

The following model configurations were compared using the same nested cross-validation framework:

1.  `TabNet` (Baseline)
2.  `TabNet + XGBoost` 
3.  `TabNet + LightGBM`
4.  `TabNet+XGBoost+LightGBM`
5.  `TabNet + XGBoost + LightGBM + LR` (Final Model)



**Code for Ablation Study:**
---

