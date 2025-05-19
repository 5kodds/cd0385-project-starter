# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### OLASENI OTUSANYA

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
TODO: When preparing the initial predictions for submission to Kaggle, I observed that the model, evaluated using root_mean_squared_error, could produce negative values for the "count" variable. Since the number of bike shares cannot be negative and Kaggle's evaluation requires non-negative predictions, a change was necessary. I inspected the predictions using predictions.describe() and confirmed the presence of negative values with (predictions < 0).sum().sum(). To address this, I implemented a step to set all predicted "count" values less than zero to zero (predictions[predictions < 0] = 0) before generating the submission file.

### What was the top ranked model that performed?
WeightedEnsemble_L3 was the initial training top-ranked model, which combines prediction from all the individual models achieved the best prediction performance

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The exploratory data analysis (EDA) involved several steps. Initially, the datetime column was parsed correctly during data loading. Histograms of all features (train.hist(figsize=(20,15))) were generated to understand their distributions.

Based on these initial observations and further analysis, the following features were created in stages:

# Initial Feature Additions (to train dataframe):

hours: Extracted the hour from the datetime column, as demand patterns often vary by time of day.

season: Converted the numerical season column to category type.

weather: Converted the numerical weather column to category type. (Assuming the code train["weather"] = train.season.astype("category") was intended to be train["weather"] = train.weather.astype("category")).

Advanced Feature Additions (to train_new and test_new dataframes):

# Temperature-based features:
hot: Categorical flag for temperatures above mean + 1 standard deviation.

cold: Categorical flag for temperatures below mean - 1 standard deviation.

mild: Categorical flag for temperatures between the 'hot' and 'cold' thresholds.

Windspeed-based features:

very_windy: Categorical flag for windspeed above the 75th percentile.

mild_wind: Categorical flag for windspeed below the 75th percentile.

Humidity-based features:

very_humid: Categorical flag for humidity above the 75th percentile.

not_humid: Categorical flag for humidity below or equal to the 50th percentile.

Rush Hour feature:

is_rush_hour: A categorical flag indicating if the time falls within predefined morning (7-9 AM), afternoon (11 AM - 2 PM), or evening (5-7 PM) rush hours, excluding holidays. All these newly created flag features were also converted to the category dtype. These features were derived by analyzing descriptive statistics (train_new.describe()) like mean, standard deviation, and percentiles.

### How much better did your model preform after adding additional features and why do you think that is?
The model performed significantly better after adding the more comprehensive set of features (temperature, wind, humidity, and rush hour indicators, in addition to the initial hour extraction and categorical conversions). The Kaggle RMSE score improved from 1.80128 (initial model) to 0.74559 (model with new features).

This improvement is likely because these engineered features provided the model with more explicit and relevant information. For example:

Converting season and weather to categorical types helps the model treat them as distinct classes rather than ordered numerical values.

Extracting the hour and identifying is_rush_hour explicitly captures cyclical daily demand patterns.

The hot, cold, mild, very_windy, mild_wind, very_humid, and not_humid features discretize continuous variables into meaningful bins that might correlate strongly with bike usage (e.g., extremely hot or very windy conditions might deter riders). These flags make it easier for the model to learn these non-linear relationships.

Essentially, the new features helped the model better understand the context of each record and identify more nuanced patterns related to bike demand.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Hyperparameter tuning (HPO) resulted in a further significant improvement in model performance. The Kaggle RMSE score for the HPO model was 0.47655, which is a notable decrease from the feature-engineered model's score of 0.74559.

The HPO process involved:

Focusing on GBM (Gradient Boosting Machine, likely LightGBM as used by AutoGluon) and NN_TORCH (Neural Network) models.

Defining search spaces for key hyperparameters:

For NN_TORCH: num_epochs (set to 15), learning_rate (log-uniform between 1e-4 and 1e-2), activation ('relu', 'softrelu', 'tanh'), and dropout_prob (0.0 to 0.5).

For GBM: num_boost_round (set to 100) and num_leaves (integer between 26 and 66).

Utilizing hyperparameter_tune_kwargs with num_trials=6 and searcher='random'.

The training was performed with presets="best_quality" and a time_limit=900.

This systematic search for better hyperparameter configurations allowed AutoGluon to fine-tune the specified models, leading to a more accurate predictor.



### If you were given more time with this dataset, where do you think you would spend more time?
If more time were available, I would focus on the following areas:

Advanced Feature Engineering: Explore interaction terms between existing features (e.g., weather conditions during rush hours). Create more granular time-based features (e.g., parts of the day like 'mid-morning', 'late_evening', or specific flags for peak weekend hours). Consider lagging features if relevant (e.g., demand in the previous hour, though this dataset might not be structured for that directly without re-sorting).

Data Quality and Outliers: Conduct a more thorough investigation of outliers or anomalies in features like temp, humidity, and windspeed and decide on a robust handling strategy.

Target Variable Transformation: Investigate if a transformation of the target variable count (e.g., log transformation) could help normalize its distribution and improve model performance, especially since RMSE is sensitive to outliers. Remember to inverse-transform predictions.

More Exhaustive Hyperparameter Tuning: Increase num_trials for HPO. Try different search strategies (e.g., Bayesian optimization if 'auto' defaults to it or a different random seed). Fine-tune the single best-performing model identified by AutoGluon after an initial broad search.

Explore More Models: While GBM and NN_TORCH were specified for HPO, allow AutoGluon to explore its full range of models during HPO, or specifically add other strong performers like XGBoost, CatBoost, or RandomForest to the HPO configuration.

Refined Feature Selection: After creating many features, employ feature selection techniques to retain only the most impactful ones, potentially reducing model complexity and training time.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
| model                      | hpo1                                           | hpo2                                                            | hpo3                                                        | score   |
| :------------------------- | :--------------------------------------------- | :-------------------------------------------------------------- | :---------------------------------------------------------- | :------ |
| `initial_training`         | `presets="best_quality"`                       | `time_limit=600`                                                | Default AutoGluon models                                    | 1.80128 |
| `add_features`             | `presets="best_quality"`                       | `time_limit=600`                                                | All engineered features (hours, categorical, temp/wind/humidity/rush_hour flags) | 0.74559 |
| `hyperparameter_optimized` | Tuned: `GBM`, `NN_TORCH`                       | `num_trials=6`, `searcher='random'`                             | `time_limit=900`, `presets="best_quality"`                    | 0.47655 |

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
This project aimed to predict bike sharing demand using the Kaggle dataset and the AutoGluon automated machine learning library. The primary evaluation metric was Root Mean Squared Error (RMSE).

The process began with an initial training run using AutoGluon's default settings (presets="best_quality", time_limit=600) on the raw training data. A crucial initial step was realizing that predictions for "count" could be negative; these were subsequently clipped to zero before submission, achieving an initial Kaggle RMSE of 1.80128.

Next, Exploratory Data Analysis (EDA) and extensive feature engineering were performed. This involved:

Extracting the hour from the datetime feature.

Converting season and weather columns to categorical types.

Creating several new categorical features based on descriptive statistics: hot, cold, mild (for temperature); 
very_windy, mild_wind (for windspeed); very_humid, not_humid (for humidity); and an is_rush_hour flag. Training a new AutoGluon model with these additional features led to a substantial improvement, reducing the Kaggle RMSE to 0.74559. This highlighted the significant impact of providing the model with more informative and well-structured features.

Finally, hyperparameter tuning (HPO) was conducted on Gradient Boosting Machines (GBM) and Neural Networks (NN_TORCH) using AutoGluon's HPO capabilities (num_trials=6, search_strategy='random', time_limit=900, presets="best_quality"). Specific search spaces were defined for parameters like num_epochs, learning_rate, activation, and dropout_prob for neural networks, and num_boost_round and num_leaves for GBMs. This HPO phase further refined the model, achieving the best Kaggle RMSE of 0.47655.

In summary, the project successfully demonstrated an iterative approach to improving a predictive model. AutoGluon provided a strong baseline and an efficient framework for training and tuning. Feature engineering proved to be the most impactful stage for performance gains, followed by systematic hyperparameter optimization. Each step progressively enhanced the model's ability to accurately predict bike sharing demand.


