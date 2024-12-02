# credit-risk-classification
## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.

We were originally provided with a csv file which included some information related to loan information, borrower information, and lastly a column which classified the related data into a "healthy loan" or "0", or a "high-risk loan" or "1". Our task was to create a Logistic Regression model to see if we could accurately predict based on the below data provided, if the loan is healthy or high-risk.

The original data provided was:
  - loan size
  - interest rate
  - borrower income
  - debt to income
  - number of accounts
  - derogatory marks
  - total debt

I did not drop any of the above columns of data, nor did I scale any of the data when creating the Logistic Regression model. I used all the data as is.


* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

When analyzing the original "loan status" column, which was the column that classified as either a healthy loan or high-risk loan, the value counts for each were:
0/healthy loan = 75036
1/high-risk loan = 2500

Before creating any prediction model, I can clearly see that there are a signficantly higher amount of healthy loan classifications compared to high-risk loans. This would represent a biased dataset.

* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).

I decided to use a Logistic Regression model for this prediction. My first step was to separate my X and y variables. My X variables were all of the columns/data which I had referenced above. My y variable was the Loan Status column. Next, I split my data using train_test_split, and assigned a random_state of 1. Next, I fit my Logistic Regression Model using my training data. Next, my model made predictions based on the test data, and I was able to review the results:



## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.
 
                 precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?

Since I only used a Logistic Regression model, I am not currently able to predict if a different model would produce higher precision scores. Analyzing the results from the model though, the precision scores were fairly high for both predicting a healthy loan and a high-risk loan. The model did predict a healthy loan at a much higher precision as shown above, with a score of 100%. The model predicted a high-risk loan at a less precise score as shown above, at 85%. This is most likely due to the fact that our original data was biased, with most of the original loan scores being classified as healthy. 


  
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

Since I believe the goal when presented with this dataset was ultimately to create a prediction model that not only would predict the healthy loans at a high accuracy, but probably more imporantly, the high-risk loans, I believe this model is very effective, since the model predicted the high-risk loan classification at a precision of 85%, which is still very high.

If you do not recommend any of the models, please justify your reasoning.
