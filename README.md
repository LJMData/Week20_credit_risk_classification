# Week20_credit_risk_classification

The objective of this code is to create a logistic regression model to predict whether a loan is high-risk or healthy.

The code follows These steps:

Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.
Check the balance of the labels variable (y) by using the value_counts function.
Split the data into training and testing datasets by using train_test_split.
Fit a logistic regression model by using the training data (X_train and y_train).
Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
Evaluate the model’s performance by calculating the accuracy score, generating a confusion matrix, and printing the classification report.
Use the RandomOverSampler module from the imbalanced-learn library to resample the data.
Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.
Evaluate the model’s performance by calculating the accuracy score, generating a confusion matrix, and printing the classification report.

Details of the results of this analysis can be found in the report also contained in the repo. 