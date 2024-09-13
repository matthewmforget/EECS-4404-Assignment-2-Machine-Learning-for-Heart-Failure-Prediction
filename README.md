# EECS-4404-Assignment-2-Machine-Learning-for-Heart-Failure-Prediction

This project is aimed at predicting heart failure in patients using various machine learning models. The dataset includes several patient attributes, such as age, weight, and other health-related factors. The goal is to build models that can classify the likelihood of heart failure based on these inputs.

Table of Contents

Project Overview
Features
Installation
Usage
Models Used
Evaluation
Results

**Project Overview**

Heart failure is a severe medical condition that requires early detection to improve patient outcomes. In this project, I used machine learning techniques to predict heart failure based on clinical data, leveraging various classification algorithms.

The dataset was split into an 80/20 train-test ratio, and models were evaluated based on accuracy, precision, recall, and ROC AUC scores.

**Features**

Data cleaning and preprocessing
Model training and evaluation
Multiple machine learning algorithms used for prediction
Data visualization and ROC curve plotting

**Installation**

To run this project, you'll need the following libraries installed:

bash:

pip install numpy pandas scikit-learn matplotlib seaborn
You can clone the repository and run the Jupyter notebook for the full analysis:

bash:

git clone https://github.com/matthewmforget/EECS-4404-Assignment-2-Machine-Learning-for-Heart-Failure-Prediction.git
cd heart-failure-prediction
Usage

Open the Jupyter Notebook FORGET_212798542.ipynb.
Run the cells step-by-step to see the data preprocessing, model training, and evaluation.
Make sure you have the required libraries installed and the dataset in the correct path.

**Models Used**

The following machine learning models were implemented to predict heart failure:

Logistic Regression
K-Nearest Neighbors (KNN)
Random Forest (Ensemble Learning)
and more!

**Evaluation**

The models were evaluated using several metrics:

Accuracy
Precision
Recall
F1 Score
AUC ROC
ROC curves were plotted to assess model performance visually.

**Results**

The best-performing model was [model name] with an accuracy of [accuracy score], an AUC ROC of [ROC score], and the following confusion matrix:

| True Positive | False Positive |
| False Negative | True Negative |
