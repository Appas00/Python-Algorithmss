Project Overview
This project aims to predict whether a customer will repurchase a vehicle using various machine learning algorithms. The dataset includes customer information, vehicle details, and repurchase history. The steps involved in this project are data preprocessing, exploratory data analysis (EDA), model training, and model evaluation.

Data Preprocessing
Loading the Dataset: The dataset is loaded using Pandas.
Handling Missing Values: Missing values are handled by filling them with zero.
Data Type Conversion: Categorical features are converted to string types to ensure proper encoding later.
Feature and Target Separation: The features and target variable are separated, with the 'ID' column being dropped.
Train-Test Split: The dataset is split into training and testing sets using an 80-20 ratio.
Exploratory Data Analysis (EDA)
Basic Information: Display basic information about the dataset including data types and missing values.
Summary Statistics: Generate summary statistics for numerical variables.
Target Variable Distribution: Plot the distribution of the target variable to understand the class imbalance.
Outlier Detection: Use boxplots to identify outliers in numerical features.
Correlation Heatmap: Generate a heatmap to visualize correlations among numerical features.
Model Training
Decision Tree Classifier
Description: A tree-based model that splits the data into subsets based on the value of input features.
Training: Train the decision tree classifier on the preprocessed data.
Visualization: Visualize the decision tree structure.
Gradient Boosting Machines (GBM)
Description: An ensemble method that builds multiple weak learners sequentially to minimize prediction error.
Training: Train a GBM classifier on the encoded training data.
Feature Importance: Plot feature importance scores.
K-Nearest Neighbors (KNN)
Description: A simple algorithm that classifies based on the majority class among the k-nearest neighbors.
Training: Train the KNN classifier and visualize the distribution of selected features.
Naive Bayes
Description: A probabilistic classifier based on Bayes' theorem with an independence assumption between features.
Training: Train the Naive Bayes classifier and evaluate it using a ROC curve.
Random Forest Classifier
Description: An ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction.
Training: Train the random forest classifier and visualize the confusion matrix.
Support Vector Machine (SVM) Classifier
Description: A classifier that finds the optimal hyperplane that maximizes the margin between classes.
Training: Train the SVM classifier and scale the features appropriately.
Model Evaluation
Accuracy: Calculate the accuracy of each model.
Classification Report: Generate a classification report including precision, recall, and F1-score.
Confusion Matrix: Generate and visualize the confusion matrix for each model to evaluate performance.
