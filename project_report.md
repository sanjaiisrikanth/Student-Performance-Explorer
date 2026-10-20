Student Performance Explorer AI
Predicting Final Mathematics Grades Using Machine Learning
Project Type: Independent Machine Learning Project
Field: Artificial Intelligence, Machine Learning, and Educational Data Analysis
Tools: Python, Google Colab, Pandas, Scikit-learn, Matplotlib
Dataset: UCI Student Performance — Mathematics (student-mat.csv)
Model: Linear Regression
Project Status: Initial implementation and evaluation completed
________________________________________
Abstract
This project explores whether a student's final Mathematics grade can be predicted using information about their study habits, academic history, school attendance, and personal and family circumstances. Using the Mathematics subset of the UCI Student Performance dataset, I developed a supervised machine-learning workflow to examine the data, prepare input features, train regression models, and evaluate their predictions.
To investigate the effect of feature selection, I compared three Linear Regression models using 3, 8, and 30 input features. I also established a mean-prediction baseline. To focus on prediction before the final grading period, the earlier-period grades G1 and G2 were excluded from all three models.
The best-performing model in this experiment used 30 features and achieved a Mean Absolute Error (MAE) of 3.40 on a held-out test set of 79 students, compared with 3.65 for the baseline. These preliminary results suggest that the additional features provided a modest improvement on this particular test split. However, the experiment does not establish generalizability to other student populations, and further validation is necessary.
1. Introduction
Machine learning can be used to investigate patterns in educational data and evaluate whether information available about students is associated with their academic outcomes. Understanding these relationships can be a useful starting point for exploring data-driven educational research.
For this project, I chose the UCI Student Performance dataset and focused on Mathematics. Rather than relying on earlier-period grades, I investigated whether other available student characteristics could help predict the final Mathematics grade.
The project also gave me an opportunity to practise the complete introductory machine-learning workflow: data exploration, feature selection, preprocessing, model training, evaluation, and comparison against a simple baseline.
2. Research Question
How does the number of student-related input features affect the performance of a Linear Regression model predicting final Mathematics grades when earlier-period grades are excluded?
The experiment compares models with different feature sets under the same train-test split.
Objectives
1.	Explore the distribution of final Mathematics grades.
2.	examine data quality and identify relationships between selected features and final grades.
3.	Train regression models using different numbers of features.
4.	Compare model performance using Mean Absolute Error.
5.	Interpret the findings and identify limitations and opportunities for further investigation.
3. Dataset Description
The project uses student-mat.csv, the Mathematics subset of the UCI Student Performance dataset.
Source: UCI Machine Learning Repository — Student Performance
The dataset was collected from two Portuguese schools and includes student grades alongside demographic, social, family, and school-related information.
Dataset property	Value
Student records	395
Total columns	33
Non-grade input features available	30
Earlier-period grades	G1 and G2
Prediction target	G3
Target scale	0–20
The three grade columns are G1 (first-period grade), G2 (second-period grade), and G3 (final grade). I excluded all three from the input feature list, using G3 only as the prediction target.
Selected feature groups
•	Academic and attendance: weekly study time, previous class failures, school absences.
•	Student characteristics: age, school, and home address type.
•	Family and social context: parental education, family relationships, and family support.
•	Daily circumstances: travel time, free time, health, and extracurricular activities.
The 30 available non-grade predictors include both numerical and categorical variables.
4. Data Exploration and Quality Checks
I used Pandas to load and examine the dataset before training the models.
4.1 Final-grade distribution
The descriptive statistics for G3 were:
Statistic	Value
Mean	10.42
Standard deviation	4.58
Minimum	0
Maximum	20
Number of records	395
A histogram showed that final grades were concentrated around the middle of the available range, with some students receiving very low grades and others receiving high grades.
This distribution helped establish the scale and variability of the prediction target.
4.2 Data quality
The initial checks identified:
•	Missing values: 0
•	Duplicate rows: 0
These checks indicated that the dataset contained no missing cells or exact duplicate rows requiring removal at this stage.
4.3 Exploring individual features
I examined selected relationships between input features and G3.
Feature	Observed relationship with G3
Study time	Pearson correlation: approximately +0.098
Previous class failures	Pearson correlation: approximately −0.360
School absences	Pearson correlation: approximately +0.034
In this dataset, previous class failures had the strongest absolute correlation with G3 among these three selected variables. Study time had a small positive linear correlation, while absences had a correlation close to zero.
These are descriptive associations, not evidence that any feature directly causes a change in academic performance. A weak linear correlation also does not rule out every possible relationship.
5. Methodology
5.1 Problem formulation
I formulated the task as a supervised regression problem.
The model receives a set of input features, represented by (X), and learns to estimate the final Mathematics grade, represented by (y = G3).
Regression was appropriate because the target was a numerical grade rather than a category such as pass or fail.
5.2 Feature selection
To compare the effect of different feature-set sizes, I developed three model versions.
Version 1 — Three features
•	studytime
•	failures
•	absences
Version 2 — Eight features
The original three features plus:
•	age
•	traveltime
•	famrel
•	freetime
•	health
Version 3 — Thirty features
All available input features except G1, G2, and G3.
This design allowed me to compare a small feature set with progressively broader representations of student circumstances.
5.3 Train-test split
I divided the dataset into training and test sets using an 80:20 split, with random_state=42.
Subset	Number of students
Training	316
Testing	79
Total	395
The training set was used to fit each model. The test set was reserved for evaluating predictions on records not used for model fitting.
The same split was used across the three model versions to make their results more directly comparable.
5.4 Preprocessing
For Versions 1 and 2, the selected input features were numerical, so they could be passed directly to Linear Regression.
Version 3 included categorical variables. I used a preprocessing pipeline with:
•	One-hot encoding for categorical variables, with unknown categories handled safely.
•	Median imputation for numerical variables.
•	Linear Regression as the estimator.
The preprocessing and regression steps were combined in a Scikit-learn pipeline so that the transformations were fitted using the training data as part of the training workflow.
5.5 Baseline model
I created a baseline using Scikit-learn's DummyRegressor with the mean strategy.
This baseline predicts the average G3 value from the training set for every test student. It provides a simple reference point for evaluating whether the trained models offer an improvement over a constant prediction.
5.6 Evaluation metric
I used Mean Absolute Error (MAE) as the main evaluation metric.
[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
]
Here, (y_i) represents an actual grade, (\hat{y}_i) represents the predicted grade, and (n) is the number of test observations.
MAE measures the average absolute difference between actual and predicted grades. Because grades range from 0 to 20, the reported MAE is expressed in grade points. Lower values indicate smaller average prediction errors.
6. Results
6.1 Model comparison
The recorded results from the experiment were:
Approach	Input features	Test MAE
Mean baseline	Constant prediction	3.65
Linear Regression — Version 1	3	3.56
Linear Regression — Version 2	8	3.57
Linear Regression — Version 3	30	3.40
Version 3 achieved the lowest test MAE among the approaches evaluated.
Compared with the baseline, its MAE was lower by 0.25 grade points, equivalent to an approximate 6.8% reduction in MAE on this particular test split.
The three-feature model also improved slightly on the baseline, while the eight-feature model performed marginally worse than the three-feature model. This illustrates that increasing the number of features does not necessarily improve a model's predictions.
6.2 Interpreting the best result
The 30-feature model's MAE of 3.40 means that its predictions differed from the actual test grades by an average absolute amount of approximately 3.4 points.
This is a preliminary result, not evidence of high predictive accuracy. The improvement over the baseline is modest, and the result comes from a single train-test split.
6.3 Prediction behaviour
I plotted actual test grades against the Version 2 predicted grades to inspect the model's behaviour.
The predictions were concentrated more closely around the middle of the grade range than the actual grades. This suggests that the model had difficulty representing some of the more extreme outcomes.
This observation is consistent with the limitations of a simple linear model, although further analysis would be needed to quantify errors across different grade ranges.
7. Discussion
The experiment produced three main observations.
First, the selected features contained some predictive information. The three-feature model performed slightly better than the mean baseline on the chosen test split.
Second, a broader feature set produced the best result in this comparison. The 30-feature model achieved an MAE of 3.40, lower than the scores of the smaller models.
Third, adding features did not guarantee improvement. The eight-feature model did not outperform the three-feature model. This reinforces the importance of evaluating models empirically rather than assuming that a larger input set is automatically better.
An important methodological choice was excluding G1 and G2. The UCI documentation notes that these earlier grades are strongly correlated with G3. Excluding them makes the prediction task more challenging, but it also makes the investigation more relevant to estimating final grades without relying on those earlier-period grades.
The findings should be interpreted cautiously. The experiment does not establish which factors cause academic outcomes, nor does it demonstrate that the model would work equally well in other schools or educational systems.
8. Limitations
Several limitations remain.
1.	Limited dataset scope: The Mathematics subset contains 395 records from two Portuguese schools. The results cannot automatically be generalized to students in other countries, schools, or educational systems.
2.	Single train-test split: The reported results depend on one random split. Cross-validation would help assess variability across multiple splits.
3.	Modest predictive improvement: The best model improved upon the baseline, but the improvement was limited.
4.	Model choice: Only Linear Regression was used for the three feature-set experiments. Other algorithms may behave differently, but their performance would need to be evaluated fairly.
5.	Feature interpretation: Correlations and predictions do not establish causation. The model should not be used to judge a student's ability or potential.
6.	Limited evaluation: MAE was the principal metric. Additional analysis of residuals, performance across grade ranges, and model stability would provide a more complete evaluation.
7.	No external validation: The model has not been tested on an independent dataset from another school or student population.
9. Future Work
The next stage of the project could investigate the following improvements:
•	Cross-validation: Evaluate model performance across multiple folds rather than relying on one split.
•	Residual analysis: Examine where prediction errors occur and whether they vary systematically with actual grades.
•	Alternative regression models: Compare Linear Regression with carefully configured models such as Decision Tree Regression or Random Forest Regression.
•	Feature analysis: Investigate which features contribute useful predictive information, using methods appropriate to the selected model.
•	Generalization testing: Evaluate the final approach on a suitable independent dataset, if available.
•	Reproducibility: Organize the notebook, document dependencies, and publish a clear README with instructions for reproducing the experiment.
Any further model comparison should use a consistent evaluation procedure and avoid selecting a model solely because it performs best on one test split.
10. Skills and Learning Outcomes
Through this project, I practised:
•	Loading and inspecting a real-world dataset with Pandas.
•	Exploring numerical distributions and feature relationships.
•	Defining a regression problem and selecting a target variable.
•	Designing experiments with different feature sets.
•	Splitting data into training and testing subsets.
•	Encoding categorical variables and constructing preprocessing pipelines.
•	Training Linear Regression models with Scikit-learn.
•	Evaluating predictions using Mean Absolute Error.
•	Comparing a trained model against a simple baseline.
•	Interpreting results cautiously and identifying limitations.
The project helped me understand that machine learning is not only about training a model. It also requires thoughtful problem formulation, appropriate evaluation, and honest interpretation of results.
11. Conclusion
In this project, I developed an introductory supervised machine-learning workflow to predict final Mathematics grades from student-related features using the UCI Student Performance dataset.
I compared a mean baseline with Linear Regression models using three, eight, and thirty input features. The 30-feature model achieved the lowest recorded test MAE of 3.40, compared with 3.65 for the baseline.
The experiment suggests that the broader feature set was more useful for prediction on the selected test split, but the improvement was modest. Further validation is necessary before drawing conclusions about general performance.
The most valuable outcome was the practical experience of taking a machine-learning question from data exploration through model evaluation. The next stage is to strengthen the experiment through cross-validation, error analysis, and reproducible documentation.
References
Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
UCI Machine Learning Repository. Student Performance — Dataset Documentation. https://archive.ics.uci.edu/dataset/320/student+performance
