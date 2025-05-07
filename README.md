# Machine-Learning-Basic-Models
In this project, I perform analysis, visualization and processing of some datasets and build basic supervised machine learning models to predict corresponding labels in the real world.

The project includes 5 notebooks as follows:

1. Predicting home loans using Linear Regression models and regularized versions Ridge, Lasso, and Elastic Net, all of these performances achieved an $R^2$ of 99.84%. In addition, the results of Polynomial Features also achieved similar scores.

2. Detecting patients with heart disease using a logistic regression model. The model table at the end of this notebook shows all the solvers and their corresponding penalties (only saga matches all penalties) and they all return an F1 score greater than 98.83%. This is a very good parameter to ensure the model is high performing and suitable for practical use.

3. Predicting leads for a travel insurance business using Naive Bayes and Decision Tree models, with an initial pre-populated dataset and pre-weighting it to avoid imbalance, including Random UnderSamping, Smoteenn, and Smoteenn datasets. To reach more customers, I focused on the Recall metric. All models perform best on the Smoteenn dataset, with Gaussian Naive Bayes (71%), Bernoulli Naive Bayes (81%), Decision Tree (98%), and Random Forest (98%).

4. Segment customers into segments using K-Means. The dataset has a lot of outliers (>30%), so I first used a logarithmic transformation, then normalized them using Standard Scaler to remove outliers by Z-score. At the end of this notebook, I tried to visualize the data in both 2D and 3D to see the groups.

5. Detecting credit fraud using SVM and MLP models. The dataset was severely imbalanced due to the nature of the dataset (label '0' has 24712 data samples while label '1' has only 422), so I used SMOTEENN to resample the data. The SVM model achieved an accuracy of 96.93% and the MLP score was 97.19%, meaning that most of the accounts are not fraudulent in reality.