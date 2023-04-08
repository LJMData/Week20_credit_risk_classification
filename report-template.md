# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The code is performing a binary classification task to predict loan risk (0 for healthy loans and 1 for high-risk loans) using logistic regression. The code first splits the data into training and testing sets, fits a logistic regression model on the original training data, and evaluates the model's performance using the testing data. The code then resamples the training data using RandomOverSampler to balance the data and fits a logistic regression model on the resampled training data. Finally, the code evaluates the performance of the resampled model using the testing data. The evaluation metrics used in the code include balanced accuracy score, confusion matrix, and classification report.

* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
The dataset provided includes financial information on individual borrowers who have taken out loans. The data includes the following features:
loan_size: The size of the loan that the borrower has taken out.
interest_rate: The interest rate charged on the loan.
borrower_income: The income of the borrower.
debt_to_income: The borrower's debt-to-income ratio, which is the ratio of their total monthly debt payments to their monthly income.
num_of_accounts: The number of credit accounts the borrower has.
derogatory_marks: The number of derogatory marks on the borrower's credit report.
total_debt: The total amount of debt the borrower has.
The goal of this dataset is to predict the loan_status of the borrower, which is a binary variable indicating whether the borrower has paid off the loan (Fully Paid) or has defaulted on the loan (Charged Off). In other words, given the financial information of a borrower, the task is to predict whether they will pay off their loan or default on it.

* Describe the stages of the machine learning process you went through as part of this analysis.
Data preprocessing: The data was put into the corrected formulas

Model selection: Logistic regression was selected

Model training: The model was trained on the labeled dataset.

Model evaluation: The model was evaluated using a confusion matrix and a classification report 

This was then repeated with resampled training data. 

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
Balanced accuracy score: 0.99

Confusion matrix:
True negative (predicted 0, actual 0): 18663
False positive (predicted 1, actual 0): 102
False negative (predicted 0, actual 1): 56
True positive (predicted 1, actual 1): 563

Classification report:
Precision:
Class 0: 1.00
Class 1: 0.85
Recall:
Class 0: 0.99
Class 1: 0.91
F1-score:
Class 0: 1.00
Class 1: 0.88
Support:
Class 0: 18765
Class 1: 619

Overall accuracy: 0.95


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
Balanced accuracy score: 0.99

Confusion matrix:
True negative (predicted 0, actual 0): 18649
False positive (predicted 1, actual 0): 116
False negative (predicted 0, actual 1): 4
True positive (predicted 1, actual 1): 615

Classification report:
Precision:
Class 0: 1.00
Class 1: 0.84
Recall:
Class 0: 0.99
Class 1: 0.99
F1-score:
Class 0: 1.00
Class 1: 0.91
Support:
Class 0: 18765
Class 1: 619

Overall accuracy: 0.99

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
Thei frist model The logistic regression model seems to perform very well in predicting both the "healthy loan" (0) and "high-risk loan" (1) labels. The balanced accuracy score of 0.95 indicates that the model is able to accurately predict both classes with high precision and recall.

From the confusion matrix, we can see that out of 18765 "healthy loan" samples, the model predicted 18663 correctly and 102 incorrectly. Similarly, out of 619 "high-risk loan" samples, the model predicted 563 correctly and 56 incorrectly. This indicates that the model is able to correctly identify the majority of both "healthy loan" and "high-risk loan" samples, with a relatively low number of false positives and false negatives.

The classification report further confirms the good performance of the model, with high precision, recall, and F1-score for both "healthy loan" and "high-risk loan" classes. The weighted average F1-score of 0.99 indicates that the model is able to accurately predict the majority of samples in the testing data, with only a small number of misclassifications. Overall, the logistic regression model appears to be a good fit for this dataset.

The second model The logistic regression model fit with oversampled data performs very well at predicting both the 0 (healthy loan) and 1 (high-risk loan) labels. The balanced accuracy score is 0.99, which is higher than the score from the previous model. Additionally, the confusion matrix shows that there are only a few misclassified data points, with 116 false positives and 4 false negatives. The classification report also shows high precision and recall scores for both labels, indicating that the model performs well at identifying both healthy and high-risk loans. Overall, the model seems to be a good fit for the data.

I would recommend the second model as it has a higher acciracy rate. 