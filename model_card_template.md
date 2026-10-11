# Model Card

<!-- For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf -->

## Model Details
* **Model Date:** October 2026.
* **Model Type:** **Random Forest Classifier** optimized via `RandomizedSearchCV` over hyperparameters including `n_estimators`, `max_features`, `max_depth`, and `bootstrap`. 
* **Objective:** The model uses a random state seed of `143` and is optimized using the **ROC AUC** scoring metric via 5-fold cross-validation.
* **Best Hyperparameters** `{'n_estimators': 366, 'max_features': 'log2', 'max_depth': 20, 'bootstrap': False}`

## Intended Use
* **Primary Intended Uses:** This model is designed to **predict whether an individual's income exceeds \$50,000 per year** based on demographic and employment attributes. It is intended for socioeconomic research, demographic studies, and high-level economic stratification analysis.

## Training Data
* **Dataset Source:** **UCI Machine Learning Repository Census Income Dataset**
* **Data Composition:** Data consists of 1994 Census database extractions. Containing **48,842 Samples** spanning 14 discrete and continuous features including age, education, marital status, race, sex, and occupation. 

## Evaluation Data
* **Dataset Split:** Evaluation was performed on a held-out test partition at **20%**.

## Metrics
* **Precision** measures the proportion of individuals predicted to earn over $50K who actually do, highlighting the model's accuracy when it makes a positive prediction.
* **Recall** measures the proportion of actual high earners that the model successfully identifies, highlighting its ability to capture the full positive population.
* **F1-Score** calculates the harmonic mean of precision and recall, providing a single balanced metric that penalizes extreme imbalances between the two classes.

**Overall performance** Precision of 0.7592, a Recall of 0.5772, and an F1-Score of 0.6558. 

## Ethical Considerations
* **Data Protected Classes:** The model utilizes features linked to **protected personal attributes** including race, sex, age, and native country. 

## Caveats and Recommendations
* **Data Aging:** The training data reflects a historical 1994 financial landscape. It does not account for modern shifts in wage distributions, inflation, or new occupational frameworks.
