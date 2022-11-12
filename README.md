# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
•	This dataset contains demographic, financial, and other features related to consumers of banking services details of the customers of a Portuguese bank. We seek to predict if the customer will subscribe to bank term deposit or not (column y).
•	The first being a standard Scikit-learn Logistic Regression model, for which Hyperdrive was used in the Azure ML SDK to run it with different values for the two hyperparameters (C - inverse of regularization strength, max_iter -  the maximum number of iterations to converge). This model had an accuracy of 90.880%. 
•	Then, the same dataset is provided to Azure AutoML to try and find the best model using its functionality. The best performing model was a "Voting Ensemble" chosen by Azure Automated ML with an accuracy of 91.824%


## Scikit-learn Pipeline
•	The Scikit-learn pipeline first the dataset is retrieved from the given URL. 
•	The training script train.py, in which the data is cleaned using clean_data method and some pre-processing steps are performed, such as converting categorical variables to binary encoding, using one hot encoding, etc. The dataset is split into 80:20 for train and test. Sklearn's LogisticRegression is used to define the Logistic Regression model.
•	Parameter sampling: I used random parameter sampling. Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance jobs. In random sampling, hyperparameter values are randomly selected from the defined search space.
•	Early Stopping policy: The early stopping policy that I implemented was a Bandit policy. This is based on slack factor/slack amount and evaluation interval, so it early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run. The main benefit of using early stopping is it saves a lot of computational resources
•	primary_metric: The name of the primary metric needs to exactly match the name of the metric logged by the training script
•	goal: It can be either Maximize or Minimize and determines whether the primary metric will be maximized or minimized when evaluating the jobs.
•	 I defined, with the primary metric being "Accuracy" and the goal being to maximize it.
The best model parameters:
 
Best Run Id	HD_13c75d31-47e4-446e-b8ae-1db4f8a75bbe_2
best run metrics	Regularization Strength: 0.1, 'Max iterations:100
Accuracy	0.9088012139605463


## AutoML
Automated machine learning iterates over many combinations of machine learning algorithms and hyperparameter settings. It then finds the best-fit model based on your chosen accuracy metric.

## Pipeline comparison
•	The data are retrieved from the provided URL
•	The data are cleaned using the same process as described above.
•	The variables and target dataframes are merged prior to the autoML process.
•	The joined dataset is used as input in the autoML configuration and the autoML run is processed locally.
The parameters that I used in setting up the AutoML run were as follows:
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    n_cross_validations=5,
    debug_log = 'automl_errors.log',
    compute_target = cpu_cluster,
    enable_early_stopping = True,
    enable_onnx_compatible_models = True)

The best model using AutoML was a Voting Ensemble model achieving an accuracy of 0.9182, as opposed to a Logistic Classifier with an accuracy of 0.9088 obtained using hyperparameter turning.

In both cases, accuracy was similar, but AutoML was ever so slightly higher. In contrast, Hyperdrive was limited to only two hyperparameters for Scikit-learn Logistic Regression, whereas AutoML used a variety of machine learning models.


## Future work
One thing which i would want in future as further improvement will be to able to give different custom cross validation strategy to the AutoML model.Increase the timeout duration for the AutoMLConfig, to allow the AutoML engine to trial a larger number of algorithms and potentially find an even better training algorithm than VotingEnsemble.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
![deleting cluster](https://user-images.githubusercontent.com/64579075/201487781-a031699b-c3ac-4577-b0db-9e7db1fc5f2f.PNG)
