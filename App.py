import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump, load
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("car details.csv")


df.drop_duplicates(inplace=True)
df.drop(columns=["name"], inplace=True)



df["year"] = df["year"].astype(int)
df["km_driven"] = df["km_driven"].astype(int)
df["selling_price"] = df["selling_price"].astype(float)


numeric_features = ["year", "km_driven"]
categorical_features = ["fuel", "seller_type", "transmission", "owner"]


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
])




categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])



preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])



sns.histplot(df['year'], bins=10)
plt.title('Distribution of Year')
plt.show()


sns.histplot(df['selling_price'], bins=50)
plt.title('Distribution of Selling Price')
plt.show()


sns.histplot(df['km_driven'], bins=50)
plt.title('Distribution of Km Driven')
plt.show()


sns.scatterplot(x='year', y='selling_price', data=df)
plt.title('Correlation between Year and Selling Price')
plt.show()



sns.scatterplot(x='km_driven', y='selling_price', data=df)
plt.title('Correlation between Km Driven and Selling Price')
plt.show()


sns.countplot(x='fuel', data=df)
plt.title('Distribution of Fuel')
plt.show()



sns.countplot(x='seller_type', data=df)
plt.title('Distribution of Seller Type')
plt.show()



sns.countplot(x='transmission', data=df)
plt.title('Distribution of Transmission')
plt.show()


sns.countplot(x='owner', data=df)
plt.title('Distribution of Owner')
plt.show()



sns.boxplot(x='fuel', y='selling_price', data=df)
plt.title('Correlation between Fuel and Selling Price')
plt.show()



sns.boxplot(x='seller_type', y='selling_price', data=df)
plt.title('Correlation between Seller Type and Selling Price')
plt.show()


sns.boxplot(x='transmission', y='selling_price', data=df)
plt.title('Correlation between Transmission and Selling Price')
plt.show()


sns.boxplot(x='owner', y='selling_price', data=df)
plt.title('Correlation between Owner and Selling Price')
plt.show()



models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
}



for name, model in models.items():
    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} MSE: {mse}, R^2: {r2}")
    # Save the model
    dump(model, f"{name}_model.joblib")


loaded_model = load("LinearRegression_model.joblib")


subset_df = df.sample(n=20, random_state=42)
X_subset = subset_df.drop("selling_price", axis=1)
y_subset = subset_df["selling_price"]
X_subset_processed = preprocessor.transform(X_subset)
y_pred = loaded_model.predict(X_subset_processed)
mse = mean_squared_error(y_subset, y_pred)
r2 = r2_score(y_subset, y_pred)
print(f"Performance on subset data (20 data points):")
print(f"MSE: {mse}, R^2: {r2}")
