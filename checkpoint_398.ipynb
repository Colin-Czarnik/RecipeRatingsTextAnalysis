import numpy as np
import plotly
import pandas as pd
import plotly.express as px
from sklearn.pipeline import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
recipes = pd.read_csv("/Users/Owner/Downloads/recipes/RAW_recipes.csv")
interactions = pd.read_csv("/Users/Owner/Downloads/recipes/RAW_interactions.csv")
display(recipes.head(3))
display(interactions.head(3))
print(recipes.columns)
print(interactions.columns)
print(recipes.isna().any())
print(interactions.isna().any())


print(recipes['nutrition'][1])

merged_df = pd.merge(recipes, interactions, left_on = 'id', right_on = 'recipe_id', how = 'left')
len(merged_df)
# Replace ratings of 0 to np.nan
merged_df.replace(0, np.nan, inplace = True)
avg_rating_per_recipe = merged_df.groupby('recipe_id')['rating'].mean()
display(avg_rating_per_recipe)
avg_rating_df = pd.DataFrame(avg_rating_per_recipe)
avg_rating_df.rename(columns = {'rating': 'avg_rating'}, inplace = True)
merged_df2 = pd.merge(merged_df, avg_rating_df, on = 'recipe_id')
merged_df2.head()
# Ratings of 0 stars indicate reviews with no star ratings. (gave it a review without giving it a rating)
# Change column nutrition to calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates
import re
# convert a string resembling a list of floats into an actual list of floats (for nutrition column)
def string_to_float_list(s):
    return [float(i) for i in re.findall('\d+\.\d', s)]

# convert a string resembling a list of strings into a list of strings (for tags and steps columns)
def string_to_string_list(s):
    return re.findall('\'(.+?)\'', s)

# apply string to float list to "nutrition", "steps", "tags"
merged = merged_df2.copy()
merged['nutrition'] = merged['nutrition'].apply(string_to_float_list)
merged['steps'] = merged['steps'].apply(string_to_string_list)
merged['tags'] = merged['tags'].apply(string_to_string_list)


merged = merged.assign(**{
    'calories (#)' : merged['nutrition'].str[0],
    'total fat (PDV)' : merged['nutrition'].str[1],
    'sugar (PDV)' : merged['nutrition'].str[2],
    'sodium (PDV)' : merged['nutrition'].str[3],
    'protein (PDV)' : merged['nutrition'].str[4],
    'saturated fat (PDV)' : merged['nutrition'].str[5],
    'carbohydrates (PDV)' : merged['nutrition'].str[6],
    })

def join_reviews(series):
    return ' '.join(series.dropna().astype(str))

merged_grouped = merged.groupby('contributor_id').agg({
    'name': 'first',
    'minutes': 'first',
    'submitted': 'first', # omit contributor_id
    'tags': 'first',
    'nutrition': 'first',
    'n_steps': 'first',
    'steps': 'first',
    'description': 'first',
    'ingredients': 'first',
    'n_ingredients': 'first', # Omit user_id
    'recipe_id': 'first',
    'calories (#)': 'first',
    'total fat (PDV)': 'first',
    'sugar (PDV)': 'first',
    'sodium (PDV)': 'first',
    'protein (PDV)': 'first',
    'saturated fat (PDV)': 'first',
    'carbohydrates (PDV)': 'first',
    'review': join_reviews, # potentially add custom agg function, adding ' ' for each string
    'avg_rating': 'first'
}).reset_index()
merged_grouped.head()
# Only keep numerical columns. TRYING TO DO MISSING VALUE IMPUTATION
numerical_df = merged_grouped.select_dtypes(include='number')

# Drop rows where 'minutes' is NaN to avoid correlation errors
numerical_df = numerical_df.dropna(subset=['minutes'])

# Calculate correlations with 'minutes'
correlations = numerical_df.corr()['minutes'].drop('minutes')  # Drop self-correlation

# Find the column with the highest absolute correlation
most_correlated_col = correlations.abs().idxmax()
highest_corr_value = correlations[most_correlated_col]

print(f"Column with highest correlation to 'minutes': {most_correlated_col}")
print(f"Correlation coefficient: {highest_corr_value}") # sodium has the highest correlation! of 0.06..., no reason to do probabilistic imputation
# Could do random imputation, but because 0.2% is missing, just drop them is the best move to prevent unnecessary noise being introduced to model.
print(merged_grouped[['calories (#)', 'minutes', 'avg_rating', 'tags', 'review', 'total fat (PDV)', 'saturated fat (PDV)', 'n_steps']].isna().any())
# minutes has na avg_rating has na. SAME AS LAST TIME WHICH IS GOOD
print(merged_grouped.shape[0])
print(merged_grouped[['minutes']].isna().sum()) # 56/27926, basically nothing. Safe to just get rid of.
print(merged_grouped[['avg_rating']].isna().sum()) # 706/27926 missing values. Much more but no reason to fill in na
# Cleaned preprocessing and train-test split
merged_copy = merged_grouped.dropna(subset=['minutes', 'avg_rating']).copy()  # Drop NaNs + make a copy
merged_copy['tags'] = merged_copy['tags'].apply(lambda x: ' '.join(x))  # Convert tag list to string

# Perform proper train-test split
X = merged_copy.drop('avg_rating', axis=1)
y = merged_copy['avg_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=214)

# make baseline model
# 2 quantitative, 0 ordinal, 0 nominal
def create_baseline_model(X_train, y_train): # Concatenating average
    # transform numerical columns
    log_transformer = FunctionTransformer(lambda x: np.log1p(x))
    
    preprocessing = make_column_transformer(
        (log_transformer, ['calories (#)', 'minutes']),  # Apply log transformation to 'calories' and 'minutes'
        remainder='drop'  # Drop other columns not being used
    )
        
    # The pipeline
    model = make_pipeline(preprocessing, LinearRegression()) # apply l1 regularization
    
    # Define feature columns
    feature_cols = ['calories (#)', 'minutes']
    return model.fit(X_train[feature_cols], y_train)

baseline_model = create_baseline_model(X_train, y_train)

baseline_model.predict(pd.DataFrame([{
    'calories (#)': 1, # slight positive coefficient, more caloric recipes more favorable
    'minutes': 8000000 # slight negative coefficient, longer recipes less favorable
}]))[0]
# make experimental model (Vector of vector spaces, each index is a date, manipulate so its reflective of time series)
# 3 quantitative +  a million more quantitative, 0 nominal, 0 ordinal.
def create_experimental_model(X_train, y_train): # For average rating, we concatenate all tags into a large string, and then use that string to predict avg rating. Avg calories and minutes
    # transform numerical columns
    log_transformer = FunctionTransformer(lambda x: np.log1p(x))
    
    # Fat ratio function transformer
    fat_ratio_transformer = FunctionTransformer(
        lambda X: ((X['saturated fat (PDV)'] / X['total fat (PDV)']).where(X['total fat (PDV)'] != 0, 0)).to_frame(name='fat_ratio')
    )

    # Pipeline to process the ratio: transformer + scaler
    fat_ratio_pipeline = make_pipeline(
        fat_ratio_transformer,
        StandardScaler()
    )

    preprocessing = make_column_transformer(
        (log_transformer, ['calories (#)', 'minutes']),  # Apply log transformation to 'calories' and 'minutes'
        (TfidfVectorizer(), 'tags'),  # Apply TF-IDF vectorization to 'tags'
        (TfidfVectorizer(max_features = 1000), 'review'),
        (fat_ratio_pipeline, ['saturated fat (PDV)', 'total fat (PDV)']),
        remainder='drop'  # Drop other columns not being used
    )
        
    # The pipeline
    # model = make_pipeline(preprocessing, Lasso()) # apply l1 regularization
    param_grid = {'lasso__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]} # testing on values 10^-5 to 10^0.

    grid_search = GridSearchCV(
        make_pipeline(preprocessing, Lasso()),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1  # Need this bc the model with only 1 cpu at a time is super slow to run grid search
    )
    
    # Define feature columns
    feature_cols = ['calories (#)', 'minutes', 'tags', 'review', 'saturated fat (PDV)', 'total fat (PDV)']
    return grid_search.fit(X_train[feature_cols], y_train)

experimental_model = create_experimental_model(X_train, y_train)

experimental_model.predict(pd.DataFrame([{
    'calories (#)': 10,
    'minutes': 70000,
    'tags': '60-minutes-or-less chicken-stew', # FOR NOW MUST BE A STRING NOT A LIST
    'review': 'So good it made me bust',
    'saturated fat (PDV)': 100, 
    'total fat (PDV)': 100
}]))[0]
# ON ALL TEST DATA UNCONDITIONAL OF GROUPING BY MSE OF AVG RATINGS
baseline_predictions = baseline_model.predict(X_test)
experimental_predictions = experimental_model.predict(X_test)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
experimental_mse = mean_squared_error(y_test, experimental_predictions)

# printing mse on test
print("Baseline MSE:", baseline_mse)
print("Experimental MSE:", experimental_mse)
# On test data with two mean squared error estimates (mse of avg_rating > 4 and avg_rating < 4)
high_ratings = y_test > 4
low_ratings = y_test <= 4
baseline_hr_predictions = baseline_model.predict(X_test[high_ratings])
experimental_hr_predictions = experimental_model.predict(X_test[high_ratings])
baseline_lr_predictions = baseline_model.predict(X_test[low_ratings])
experimental_lr_predictions = experimental_model.predict(X_test[low_ratings])

baseline_hr_mse = mean_squared_error(y_test[high_ratings], baseline_hr_predictions)
experimental_hr_mse = mean_squared_error(y_test[high_ratings], experimental_hr_predictions)
baseline_lr_mse = mean_squared_error(y_test[low_ratings], baseline_lr_predictions)
experimental_lr_mse = mean_squared_error(y_test[low_ratings], experimental_lr_predictions)

print("Baseline MSE (avg_rating > 4):", baseline_hr_mse)
print("Baseline MSE (avg_rating ≤ 4):", baseline_lr_mse)
print("Experimental MSE (avg_rating > 4):", experimental_hr_mse)
print("Experimental MSE (avg_rating ≤ 4):", experimental_lr_mse)
# Labels and values
labels = [
    "Baseline (All)",
    "Experimental (All)",
    "Baseline (>4 stars)",
    "Experimental (>4 stars)",
    "Baseline (≤4 stars)",
    "Experimental (≤4 stars)"
]

mse_values = [
    baseline_mse,
    experimental_mse,
    baseline_hr_mse,
    experimental_hr_mse,
    baseline_lr_mse,
    experimental_lr_mse
]
# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, mse_values, color=['#1e87b4', '#ff7f0d', '#1e87b4', '#ff7f0d', '#1e87b4', '#ff7f0d'])

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

plt.ylabel("Mean Squared Error")
plt.title("Model MSE Comparison on Test Data")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
print("Best alpha:", experimental_model.best_params_['lasso__alpha']) # best alpha is 0.0001
# print("Best word limit for reviews tf idf vectorizer:", experimental_model.best_params_['preprocessing__transformers__2__max_features']) # best word limit is 

results = experimental_model.cv_results_
# print(results.keys())
# Extracting the alphas, and the validation MSE
alphas = results['param_lasso__alpha']
val_mse = results['mean_test_score']

# Set to negative of mean to reflect actual MSE
train_mse = -np.mean([results[f'split{i}_test_score'] for i in range(5)], axis=0)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(alphas, -val_mse, label='Validation MSE', marker='o')
plt.plot(alphas, train_mse, label='Training MSE', marker='o')
plt.xscale('log')  # Using log scale for alpha values
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training vs Validation MSE for Different Alphas')
plt.legend()
plt.grid(True)
plt.show()

merged['id'].isna().any()
