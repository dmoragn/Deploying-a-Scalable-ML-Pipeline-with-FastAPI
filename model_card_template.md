# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Name: Random Forest Classifier for Income Prediction
Model Version: 1.0
Model Type: Random Forest Classifier
Owner: Devin Moragne
Contact Information: dmoragn@wgu.edu

## Intended Use
This model is used to predict whether or not an individual has an income which exceeds $50,000. This prediction is based on various demographic features and is also used for analysis, educational purposes, and research. This is a model which should not be used for making critical financial life decisions or financial decisions. 

## Training Data
This training data associated with this model includes the attributes of age, workclass, final weight, education, education number, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country. These are the attributes that are being used to make predictions on if someone makes $50,000 or not. The training data consists of 80 percent of the dataset, and the remaining 20 percent of the data was used for evaluation. 

## Evaluation Data
The evaluation data is a subset of the "Census Income" dataset, representing 20% of the entire dataset. It includes the same features as the training data and was used to evaluate the performance of the model.

## Metrics
The performance of the model was evaluated using the following metrics:

Precision: Measures the accuracy of positive predictions.
Model Performance: 0.7376
Recall: Measures the proportion of actual positives correctly identified.
Model Performance: 0.6288
F1 Score: The harmonic mean of precision and recall.
Model Performance: 0.6789
The model achieved the following performance metrics on the test set:

Precision: 0.7376
Recall: 0.6288
F1 Score: 0.6789

## Ethical Considerations
When incorporating this model into decision-making, it is very important to understand the ethical aspect of this equation. The "Census Income" dataset reflects historical data, which may or may not include biases related to race, gender, and socioeconomic status. One should take into consideration these biases as they are applying the model to new data. 

## Caveats and Recommendations
The model includes several limitations such as the model's performance being based on the specific dataset and may not generalize well to other populations or time periods. The model's perdictions also should not be used for making critical decisions without further validation. Lastly, the model may not perform well on data points with missing values. Reccomendations for the model include enhancing the model by using more advanced algorithms such as hyperparameter tuning, extending the evaluation to include more diverse datasets, and implementing techniques to address and mitigate biases within the training data. 
