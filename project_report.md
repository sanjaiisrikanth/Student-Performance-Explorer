# Student Performance Explorer AI
## Predicting Final Mathematics Grades Using Machine Learning

**Project type:** Independent Machine Learning Project  
**Field:** Artificial Intelligence, Machine Learning, and Educational Data Analysis  
**Author:** Sanjaii Srikanth  
**Language:** Python  
**Environment:** Google Colab  
**Libraries:** Pandas, Scikit-learn, Matplotlib  
**Dataset:** UCI Student Performance — Mathematics (`student-mat.csv`)  
**Model:** Linear Regression  
**Status:** Initial implementation and evaluation completed

---

## Abstract

This project investigates whether student-related information can be used to predict final Mathematics grades using supervised machine learning. Using the Mathematics subset of the UCI Student Performance dataset, I developed a workflow for data exploration, feature selection, preprocessing, model training, and evaluation.

I compared three Linear Regression models using 3, 8, and 30 input features against a mean-prediction baseline. The earlier-period grades G1 and G2 were excluded from all models to investigate prediction without directly using those grades.

The 30-feature model achieved a Mean Absolute Error (MAE) of 3.40 on a held-out test set of 79 students, compared with 3.65 for the baseline. This represents a modest improvement on the selected test split. The results are preliminary and require further validation before conclusions can be drawn about performance on other student populations.

## 1. Introduction

Machine learning provides tools for identifying patterns in data and evaluating whether those patterns can help predict outcomes. In education, student records can be used to investigate associations between academic performance and characteristics such as study habits, attendance, and family circumstances.

For this project, I selected the UCI Student Performance dataset and focused on its Mathematics records. I wanted to investigate whether student-related features could help predict the final Mathematics grade without using the earlier-period grades as model inputs.

The project also provided practical experience with the core stages of a machine-learning workflow, from examining a dataset to evaluating and interpreting predictions.

## 2. Research Question and Objectives

### 2.1 Research question

**How does the number of input features affect the performance of a Linear Regression model predicting final Mathematics grades when earlier-period grades are excluded?**

### 2.2 Objectives

1. Examine the distribution of final Mathematics grades.
2. Check the dataset for missing values and duplicate records.
3. Explore relationships between selected features and final grades.
4. Train regression models using different feature-set sizes.
5. Compare the models against a simple baseline using Mean Absolute Error.
6. Identify the limitations of the experiment and propose further improvements.

## 3. Dataset Description

The project uses `student-mat.csv`, the Mathematics subset of the UCI Student Performance dataset.

**Dataset source:** [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)

The dataset contains student grades and information about demographic, social, family, and school-related circumstances. It was collected from two Portuguese schools.

| Property | Description |
|---|---|
| Number of records | 395 students |
| Total columns | 33 |
| Available non-grade input features | 30 |
| First-period grade | G1 |
| Second-period grade | G2 |
| Final Mathematics grade | G3 |
| Grade scale | 0–20 |

### 3.1 Feature groups

The available non-grade features cover several areas:

- **Academic habits:** weekly study time and previous class failures.
- **Attendance:** school absences.
- **Student characteristics:** age, school, and home address type.
- **Family context:** parental education, parental jobs, family relationships, and support.
- **Daily circumstances:** travel time, free time, health, and extracurricular activities.

The dataset includes both numerical and categorical variables.

### 3.2 Prediction target

The target variable was `G3`, representing the final Mathematics grade.

The earlier grades `G1` and `G2` were excluded from the input features. This design made the prediction task more challenging and allowed the experiment to focus on other student-related information.

## 4. Data Exploration and Quality Checks

I used Pandas to load and inspect the dataset before training the models.

### 4.1 Descriptive statistics

The recorded descriptive statistics for G3 were:

| Statistic | Value |
|---|---:|
| Number of records | 395 |
| Mean | 10.42 |
| Standard deviation | 4.58 |
| Minimum | 0 |
| Maximum | 20 |

The grade histogram showed that grades were distributed across the available range, with many observations around the middle and some very low and high grades.

Understanding the distribution helped establish the scale and variability of the prediction target.

### 4.2 Data quality

The initial data-quality checks returned:

| Check | Result |
|---|---:|
| Missing values | 0 |
| Duplicate rows | 0 |

No missing cells or exact duplicate rows were identified by these checks.

### 4.3 Correlation analysis

I examined the Pearson correlations between G3 and three selected numerical features.

| Feature | Correlation with G3 |
|---|---:|
| Study time (`studytime`) | +0.098 |
| Previous class failures (`failures`) | −0.360 |
| School absences (`absences`) | +0.034 |

Among these three features, previous class failures had the strongest absolute linear correlation with the final grade in this dataset. Study time showed a small positive correlation, while absences had a correlation close to zero.

Correlation describes an observed statistical relationship. It does not establish that a feature causes a change in academic performance, and a weak Pearson correlation does not rule out every possible nonlinear relationship.

## 5. Methodology

### 5.1 Problem formulation

I formulated the task as a supervised regression problem.

The input features are represented by \(X\), while the target variable is the final Mathematics grade, \(y = G3\). The model learns a relationship between the input data and the numerical target, then produces grade predictions for unseen records.

Linear Regression was selected as the initial model because it provides a straightforward baseline for exploring relationships between multiple features and a numerical outcome.

### 5.2 Feature-set experiments

I trained three Linear Regression models with different input feature sets.

**Version 1 — Three features**

- `studytime`
- `failures`
- `absences`

**Version 2 — Eight features**

The original three features, plus:

- `age`
- `traveltime`
- `famrel`
- `freetime`
- `health`

**Version 3 — Thirty features**

All available non-grade features, excluding `G1`, `G2`, and `G3`.

This design allowed me to compare a small feature set with progressively broader representations of student circumstances.

### 5.3 Train-test split

I used an 80:20 train-test split with `random_state=42`.

| Subset | Number of records |
|---|---:|
| Training set | 316 |
| Test set | 79 |
| Total | 395 |

The training set was used to fit the models. The test set was held out for evaluating predictions on records that were not used for model fitting.

The same split was used for the three model versions to support a direct comparison.

### 5.4 Data preprocessing

Versions 1 and 2 used numerical input features.

Version 3 included categorical variables, so I used a Scikit-learn preprocessing pipeline with:

- **One-hot encoding** for categorical variables, with unknown categories handled using `handle_unknown="ignore"`.
- **Median imputation** for numerical variables.
- **Linear Regression** as the estimator.

The transformations were incorporated into a pipeline and fitted as part of the training workflow. Although the initial quality checks found no missing cells, median imputation was included as a safeguard in the preprocessing pipeline.

### 5.5 Baseline model

I created a baseline using Scikit-learn's `DummyRegressor` with the mean strategy.

The baseline predicts the average G3 value from the training set for every test record. It provides a simple reference for determining whether the trained models improve on a constant prediction.

### 5.6 Evaluation metric

I used Mean Absolute Error (MAE) to compare model performance.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Here:

- \(n\) is the number of test observations.
- \(y_i\) is an actual grade.
- \(\hat{y}_i\) is the predicted grade.

MAE measures the average absolute difference between actual and predicted values. Because the grade scale is 0–20, the result is expressed in grade points. Lower MAE indicates smaller average absolute prediction errors.

## 6. Results

### 6.1 Model comparison

The recorded test results were:

| Approach | Input features | Test MAE |
|---|---:|---:|
| Mean-prediction baseline | Constant prediction | 3.65 |
| Linear Regression — Version 1 | 3 | 3.56 |
| Linear Regression — Version 2 | 8 | 3.57 |
| Linear Regression — Version 3 | 30 | **3.40** |

The 30-feature model achieved the lowest recorded MAE among the approaches evaluated.

Compared with the baseline, its MAE decreased by 0.25 grade points, corresponding to an approximate 6.8% reduction in MAE on this test split.

The three-feature model also performed slightly better than the baseline. However, the eight-feature model performed marginally worse than the three-feature model.

These results show that increasing the number of input features did not guarantee improved predictions.

### 6.2 Interpretation of the best result

The Version 3 model's MAE of 3.40 means that its predictions differed from the actual test grades by an average absolute amount of approximately 3.4 grade points.

This result indicates a modest improvement over the baseline, not high predictive accuracy. Further evaluation is necessary to determine whether the improvement is stable across different data splits.

### 6.3 Prediction behaviour

I also examined an actual-versus-predicted scatter plot for Version 2.

The predicted values were more concentrated around the middle of the grade range than the actual values. This suggested that the model had difficulty capturing some of the more extreme outcomes.

This pattern is consistent with predictions tending towards the average, but further residual analysis would be needed to characterize the errors systematically.

## 7. Discussion

The experiment produced three main observations.

**First, the selected features contained some predictive information.** The three-feature Linear Regression model performed slightly better than the mean baseline on the selected test split.

**Second, the broader feature set achieved the best result.** The 30-feature model obtained an MAE of 3.40, lower than the scores of the smaller models.

**Third, adding features did not guarantee better performance.** The eight-feature model did not outperform the three-feature model, demonstrating the importance of empirical evaluation.

Excluding G1 and G2 was an important design decision. Earlier-period grades are closely related to the final grade, so including them could make the prediction task easier. Excluding them allowed me to investigate whether other student-related features could provide useful predictive information.

However, this experiment does not establish which factors cause academic outcomes. The dataset's limited scope and the use of a single test split also mean that the findings should not be generalized automatically to other student populations.

## 8. Limitations

Several limitations should be considered when interpreting the results.

1. **Limited dataset scope:** The dataset contains 395 records from two Portuguese schools. The findings may not generalize to different schools, countries, or educational systems.
2. **Single train-test split:** The results depend on one split. Cross-validation would help evaluate performance variability.
3. **Modest improvement:** The best model improved on the baseline, but the reduction in MAE was limited.
4. **Limited model comparison:** The experiment used Linear Regression for all three feature-set versions. Alternative algorithms may perform differently.
5. **Correlation is not causation:** Statistical relationships cannot establish that a particular feature causes an academic outcome.
6. **Limited error analysis:** MAE summarizes average absolute error but does not fully describe how errors vary across the grade range.
7. **No external validation:** The models have not been evaluated on an independent dataset from another school or student population.

The model should be treated as an educational experiment, not as a validated tool for evaluating individual students or determining their academic potential.

## 9. Future Work

The next stage of the project could investigate several improvements.

### 9.1 Cross-validation

Evaluate the models across multiple folds to determine whether the results are stable across different subsets of the data.

### 9.2 Residual analysis

Examine prediction errors to identify patterns, such as whether the model systematically underpredicts or overpredicts certain grade ranges.

### 9.3 Alternative regression algorithms

Compare Linear Regression with models such as Decision Tree Regression or Random Forest Regression using a consistent evaluation procedure.

### 9.4 Feature analysis

Investigate which features contribute useful predictive information and whether the model relies on unstable or potentially misleading relationships.

### 9.5 Generalization testing

Where a suitable independent dataset is available, evaluate whether the model's performance transfers to students from a different population.

### 9.6 Reproducibility

Document the software dependencies, data-loading instructions, notebook execution steps, and evaluation procedures so that other people can reproduce the experiment.

Any further model comparisons should use a fair evaluation procedure and should not select a model solely because it performs best on one test split.

## 10. Skills and Learning Outcomes

Through this project, I practised:

- Loading and inspecting a real-world dataset with Pandas.
- Exploring distributions and feature correlations.
- Formulating a supervised regression problem.
- Selecting input features and a numerical target.
- Splitting data into training and test subsets.
- Encoding categorical variables.
- Constructing preprocessing and regression pipelines.
- Training Linear Regression models using Scikit-learn.
- Evaluating predictions with Mean Absolute Error.
- Comparing trained models with a simple baseline.
- Interpreting experimental results and documenting limitations.

The project helped me understand that machine learning involves more than fitting an algorithm. It also requires careful problem formulation, appropriate evaluation, reproducibility, and honest interpretation of evidence.

## 11. Conclusion

This project developed an introductory supervised machine-learning workflow for predicting final Mathematics grades using the UCI Student Performance dataset.

I compared a mean-prediction baseline with Linear Regression models using three, eight, and thirty input features. The 30-feature model achieved the lowest recorded test MAE of 3.40, compared with 3.65 for the baseline.

The results suggest that the broader feature set provided a modest improvement on the selected test split. However, the experiment does not establish that the model will generalize reliably to other student populations.

The main outcome was practical experience with data exploration, preprocessing, feature selection, regression, model evaluation, and critical interpretation. Future work will focus on cross-validation, residual analysis, and testing alternative approaches.

## References

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository.

Dataset page: https://archive.ics.uci.edu/dataset/320/student+performance

## Project Repository

The accompanying Jupyter notebook contains the code and recorded outputs for data exploration, preprocessing, model training, and evaluation.

