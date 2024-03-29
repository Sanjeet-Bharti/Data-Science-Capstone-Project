CAR DETAILS

1.Data Preprocessing:

 *The dataset is read using pandas from a CSV file named "car details.csv".
 *Duplicates are removed from the DataFrame.
 *The "name" column is dropped.
 *Data types for "year", "km_driven", and "selling_price" columns are coerced to appropriate types.
 *Numeric and categorical features are separated.

2.Preprocessing Pipelines:

 *Separate preprocessing pipelines are defined for numeric and categorical features.
 *For numeric features, missing values are imputed using median and then scaled using MinMaxScaler.
 *For categorical features, missing values are imputed with a constant value and then one-hot encoded.

3.Data Visualization:

 *Various histograms, scatter plots, and count plots are created to visualize the distributions and relationships between different variables in the dataset.

4.Modeling:

 *Three regression models (Linear Regression, Decision Tree Regression, and Random Forest Regression) are initialized.
 *For each model, the dataset is split into training and testing sets, preprocessed, and then fitted to the model.
 *Mean squared error (MSE) and R-squared (R^2) scores are computed for each model on the test set.
 *Trained models are saved using joblib.

5.Model Evaluation:

 *The Linear Regression model is loaded from the saved file.
 *Subset data (20 data points) is randomly sampled from the original dataset.
 *The subset data is preprocessed and used to make predictions using the loaded Linear Regression model.
 *MSE and R^2 scores are computed for the predictions on the subset data.
