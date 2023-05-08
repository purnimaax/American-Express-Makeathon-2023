# American-Express-Makeathon-2023
ML-based fraud detection --> Problem statement name: Develop and maintain ML-based fraud detection models that are effective at identifying evolving fraud patterns even in the presence of imbalanced data.

Credit Card Fraud Detection

Problem statement
Develop and maintain ML-based fraud detection models that are effective at identifying evolving fraud patterns even in the presence of imbalanced data.

Project Understanding
Say a customer service representative from your bank calls to let you know that your card will expire in a week. You immediately check your card information and see that it expires in eight days. The executive now requests that you confirm a few data, including your credit card number, expiration date, and CVV number, to renew your membership. Will you give the executive this information?
You should exercise caution in these circumstances since the information you could divulge to them could give them full access to your credit card account.
Although India saw a 51% increase in digital transactions from 2018 to 2019, there are still concerns about their security. With over 52,304 occurrences of credit/debit card fraud recorded in FY 2019 alone, fraudulent activities have multiplied. Due to the sharp rise in banking fraud, it is vital that these fraudulent transactions be found out as soon as possible to assist both customers and banks that are daily losing their credit value. To identify fraudulent transactions, machine learning may be quite useful.
Let’s know about the dataset used in this project.

About the Dataset
•	There are a total of 2,84,807 transactions in the data set, 492 of which are false, and it was acquired from the Kaggle website. The data set needs to be handled because it is so severely unbalanced before a model can be built.
•	For clients to avoid being charged for products they did not buy, credit card issuers must be able to identify fraudulent credit card transactions.
•	The dataset includes credit card transactions performed by European cardholders in September 2013. We have 492 frauds out of 284,807 transactions in our dataset of transactions that took place over the course of two days. The dataset is quite skewed, with frauds making up 0.172% of all transactions in the positive class.
•	All the input variables are numbers that have undergone PCA transformation. The major components derived with PCA are features V1, V2, …., V28. The only features that have not been changed with PCA are "Time" and "Amount." The seconds that passed between each transaction and the dataset's initial transaction are listed in the 'Time' feature. The transaction amount is represented by the feature "Amount," which may be utilised for example-dependent, cost-sensitive learning. The response variable, feature "Class," has a value of 1 in cases of fraud and 0 in all other cases.
•	It is advised to measure accuracy using the Area Under the Precision-Recall Curve (AUPRC) given the class imbalance ratio. For categorization that is not balanced, confusion matrix accuracy is meaningless.
After performing EDA as well we got to know that the dataset is highly imbalanced. So, before splitting the data into train and test data and fitting it to different machine learning models, we need to balance out the values in ‘Class’ feature. This means that number of rows with 1 in the ‘Class’ column should be equal to the number of rows with 0.

Dealing with Imbalanced Data
To fix this issue, out of many options available, like Choose Proper Evaluation Metric, resampling (Oversampling and Undersampling), synthetic Minority Oversampling Technique(SMOT),etc., we have used Random Undersampling.
Undersampling for Imbalanced Classification
A collection of methods known as undersampling are used to balance the class distribution in classification datasets where the class distribution is skewed.
A class distribution that is unbalanced will include one or more minority classes with few examples and one or more majority classes with many examples. It makes the most sense in the context of a binary (two-class) classification issue where class 0 is the majority class and class 1 is the minority class.
On a training dataset, undersampling techniques can be applied directly before a machine learning model is fitted. Typically, undersampling techniques are combined with an oversampling method for the minority class, and this approach frequently yields better performance on the training dataset than either oversampling or undersampling alone.
In the simplest undersampling method, instances from the majority class are randomly chosen and removed from the training dataset. In statistics, this is known as random undersampling. 
We have performed random undersampling with the training credit card dataset and created a new data set with the help of legit and fraud data with 473 values each. 
Next step was to split the newly formed dataset with the help of train_test_split library from sklearn.model_selection. Test data is set to be 20% of the new data with stratify as y and random_state as 42.

Model Training
Now it’s time to train the newly formed data on different algorithms such as Logistic Regression, Decision Tree, Random Forest and XGBoosting. Let’s get introduced to all these algorithms.

Logistic Regression
A statistical technique for forecasting binary classes is logistic regression. The result or goal variable has a binary nature. It may be used, for instance, to issues with cancer detection. It determines the likelihood that an event will occur.
When the target variable is categorical, linear regression is used in a specific way. A log of the odds is used as the dependent variable. Using a logit function, logistic regression makes predictions about the likelihood that a binary event will occur.
The sigmoid function, also known as the logistic function, produces a 'S'-shaped curve that may transfer any real-valued integer to a value between 0 and 1. Y anticipated will become 1 if the curve travels to a positive infinity, and 0 if the curve goes to a negative infinity.


Decision Tree Algorithm 
A decision tree is a tree structure that resembles a flowchart where each leaf node symbolises the result and each inside node indicates a characteristic (or attribute). The root node in a decision tree is located at the top and gains the ability to divide data depending on attribute values. It uses visualisation to replicate thinking at the human level and is simple to understand and interpret. The amount of records and number of characteristics in the provided data determine the temporal complexity of decision trees. It is a non-parametric or distribution-free strategy that does not rely on the assumptions of a probability distribution.


Random Forest 
The bagging technique is extended by the random forest algorithm, which uses feature randomness in addition to bagging to produce an uncorrelated forest of decision trees. The random subspace approach, also known as feature bagging, creates a random subset of features that guarantees minimal correlation between decision trees. The main distinction between decision trees and random forests is this. Random forests merely choose a portion of those feature splits, whereas decision trees consider all potential feature splits.

XGBoost
XGBoost is an optimized distributed gradient boosting library designed for efficient and scalable training of machine learning models. It is an ensemble learning method that combines the predictions of multiple weak models to produce a stronger prediction. It has become one of the most popular and widely used machine learning algorithms due to its ability to handle large datasets and its ability to achieve state-of-the-art performance in many machine learning tasks. It has built-in support for parallel processing, making it possible to train models on large datasets in a reasonable amount of time. It is highly customizable and allows for fine-tuning of various model parameters to optimize performance.
We fitted the training data on all the four models mentioned above, with the help of fit() method. Now, to compare the performance of all the models, we used score parameters.

Score Parameters Used
Model selection and evaluation using tools, take a scoring parameter that controls what metric they apply to the estimators evaluated.
Accuracy
The ratio of true positives and true negatives to all positive and negative observations is referred to as the model accuracy, which is a performance statistic for machine learning classification models. In other words, accuracy indicates the proportion of times our machine learning model will predict a result accurately out of all the predictions it has made.
Precision
The percentage of labels that were correctly predicted positively is represented by the model accuracy score. Another name for precision is the positive predictive value. False positives and false negatives are traded off using precision together with recall.

Recall
The model's ability to properly forecast positives out of real positives is measured by the model recall score. This differs from precision, which counts the proportion of accurate positive predictions among all positive predictions given by models.

F1 Score
A popular statistic for assessing the effectiveness of binary classifiers is the F1-score. It offers a gauge of the model's overall accuracy by combining precision and recall into a single score.

	Logistic Regression	Decision Tree	Random Forest	XGBoosting
Accuracy	94.73%	91.57%	94.73%	95.26%
Precision	89.47%	90.52%	90.52%	91.57%
Recall	100%	92.47%	98.85%	98.86%
F1 Score	94.44%	91.48%	94.50%	95.08%

As, we can clearly see in the table above that accuracy score, precision score,  recall score and F1 score, all are highest in the case of XGBoosting.

Conclusion
XGBoost have highest accuracy on testing data, i.e., 95.26%. Based on the performance of this model with the test data, we can conclude that it is effective in accurately predicting the outcome of credit card transactions and detecting fraudulent activity.
