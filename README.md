# Student Performance Explorer AI

**An independent machine-learning project exploring the prediction of final Mathematics grades.**

## Overview

This project investigates whether student-related features can help predict final Mathematics grades using Python and supervised machine learning.

Using the Mathematics subset of the UCI Student Performance dataset, I compared three Linear Regression models with different feature-set sizes against a simple mean-prediction baseline.

## Research Question

How does the number of input features affect the prediction of final Mathematics grades when earlier-period grades are excluded?

## Dataset

- **Source:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Records:** 395 students
- **Target:** `G3` — final Mathematics grade
- **Excluded from inputs:** `G1` and `G2` — earlier-period grades

## Results

| Model | Features | Test MAE |
|---|---:|---:|
| Mean baseline | Constant prediction | 3.65 |
| Linear Regression — Version 1 | 3 | 3.56 |
| Linear Regression — Version 2 | 8 | 3.57 |
| Linear Regression — Version 3 | 30 | **3.40** |

Lower MAE indicates a smaller average absolute prediction error. The 30-feature model performed best in this single train-test split, with a modest improvement over the baseline.

## Tools

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

## Repository Contents

- `PROJECT_001_Student_Performance_Explorer.ipynb` — project notebook
- `project_report.md` — detailed project report

## Limitations

- Evaluation used one 80:20 train-test split.
- The model's improvement over the baseline was modest.
- The dataset comes from two Portuguese schools.
- Results do not establish causation or guarantee performance on other student populations.

## Future Work

- Evaluate the model using cross-validation.
- Analyse prediction errors.
- Compare alternative regression models.
- Test generalization using an independent dataset.

## Acknowledgements

Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/320/student+performance).

## Model Performance Comparison

![Model Performance Comparison](Model%20Performance%20Comparison.png)

Lower MAE is better. In this experiment, the 30-feature Linear Regression model achieved the lowest MAE (3.40), compared with the baseline MAE of 3.65. Results are from a single 80:20 train-test split.
