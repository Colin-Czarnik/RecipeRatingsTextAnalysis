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
recipes = pd.read_csv("/Users/Owner/Downloads/RAW_recipes_copy.csv")
interactions = pd.read_csv("/Users/Owner/Downloads/RAW_interactions_copy.csv")
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

merged_grouped = merged.groupby('id').agg({
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
print(merged_grouped[['minutes']].isna().sum()) # 1/83781, basically nothing. Safe to just get rid of.
print(merged_grouped[['avg_rating']].isna().sum()) # 2608/83781 missing values. Much more but no reason to fill in na
# Cleaned preprocessing and train-test split
merged_copy = merged_grouped.dropna(subset=['minutes', 'avg_rating']).copy()  # Drop NaNs + make a copy
merged_copy['tags'] = merged_copy['tags'].apply(lambda x: ' '.join(x))  # Convert tag list to string

# Perform proper train-test split
X = merged_copy.drop('avg_rating', axis=1)
y = merged_copy['avg_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=214)

# make baseline model
# 2 quantitative, 0 ordinal, 0 nominal
def create_baseline_model(): # Concatenating average
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
    return model, feature_cols

baseline_model, feature_cols = create_baseline_model()
display(baseline_model)

baseline_model.fit(X_train[feature_cols], y_train)

baseline_model.predict(pd.DataFrame([{
    'calories (#)': 1, # slight positive coefficient, more caloric recipes more favorable
    'minutes': 8000000 # slight negative coefficient, longer recipes less favorable
}]))[0]
# Predict on training and test data
y_train_pred = baseline_model.predict(X_train[feature_cols])
y_test_pred = baseline_model.predict(X_test[feature_cols])

# Compute MSE
baseline_train_mse = mean_squared_error(y_train, y_train_pred)
baseline_test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Baseline Model - Train MSE: {baseline_train_mse:.4f}")
print(f"Baseline Model - Test MSE: {baseline_test_mse:.4f}")

# On test data with two mean squared error estimates (mse of avg_rating > 4 and avg_rating < 4)
high_ratings_test = y_test > 4
low_ratings_test = y_test <= 4
high_ratings_train = y_train > 4
low_ratings_train = y_train <= 4
baseline_hr_train_predictions = baseline_model.predict(X_train[high_ratings_train])
baseline_lr_train_predictions = baseline_model.predict(X_train[low_ratings_train])
baseline_hr_test_predictions = baseline_model.predict(X_test[high_ratings_test])
baseline_lr_test_predictions = baseline_model.predict(X_test[low_ratings_test])

baseline_hr_train_mse = mean_squared_error(y_train[high_ratings_train], baseline_hr_train_predictions)
baseline_lr_train_mse = mean_squared_error(y_train[low_ratings_train], baseline_lr_train_predictions)
baseline_hr_test_mse = mean_squared_error(y_test[high_ratings_test], baseline_hr_test_predictions)
baseline_lr_test_mse = mean_squared_error(y_test[low_ratings_test], baseline_lr_test_predictions)

print("Baseline Model - Train MSE (avg_rating > 4):", baseline_hr_train_mse)
print("Baseline Model - Train MSE (avg_rating ≤ 4):", baseline_lr_train_mse)
print("Baseline Model - Test MSE (avg_rating > 4):", baseline_hr_test_mse)
print("Baseline Model - Test MSE (avg_rating ≤ 4):", baseline_lr_test_mse)
# Labels and values for the bar plot
labels = [
    "Baseline (Train)", 
    "Baseline (Test)", 
    "Baseline (>4 stars) Train", 
    "Baseline (>4 stars) Test", 
    "Baseline (≤4 stars) Train", 
    "Baseline (≤4 stars) Test"
]

mse_values = [
    baseline_train_mse,
    baseline_test_mse,
    baseline_hr_train_mse,
    baseline_hr_test_mse,
    baseline_lr_train_mse,
    baseline_lr_test_mse
]

# Create a bar plot using Plotly
fig = go.Figure()

# Add bars to the plot
fig.add_trace(go.Bar(
    x=labels,
    y=mse_values,
    marker=dict(color=['blue', 'green', 'blue', 'green', 'blue', 'green']),
    text=[f'{val:.4f}' for val in mse_values],  # Annotate bars with MSE values
    textposition='outside',
))

# Update layout for better presentation
fig.update_layout(
    title="Baseline Model - MSE Comparison",
    xaxis_title="Model",
    yaxis_title="Mean Squared Error (MSE)",
    template="plotly_dark",  # Choose a dark theme for the plot
    showlegend=False,
    xaxis_tickangle=-45,
    margin=dict(l=40, r=40, b=60, t=60),
    height=500
)

# Show the plot
fig.show()
# Save to html file
fig.write_html('baseline_model_mse_performance.html', include_plotlyjs = 'cdn')
# make experimental model (Vector of vector spaces, each index is a date, manipulate so its reflective of time series)
# 3 quantitative +  a million more quantitative, 0 nominal, 0 ordinal.
def create_experimental_model(): # For average rating, we concatenate all tags into a large string, and then use that string to predict avg rating. Avg calories and minutes
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
        (TfidfVectorizer(), 'review'),
        (fat_ratio_pipeline, ['saturated fat (PDV)', 'total fat (PDV)']),
        remainder='drop'  # Drop other columns not being used
    )
        
    # Parameter Grid For Grid Search
    param_grid = {'lasso__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], # testing on values 10^-5 to 10^0.
                  'columntransformer__tfidfvectorizer-2__max_features': [1000, 1500, 2000, 2500, 3000]} # testing on values 1000 to 3000.
    
    # The model pipeline
    pipe = make_pipeline(preprocessing, Lasso()) # l1 regularization
    # print(sorted(pipe.get_params().keys()))

    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1  # Need this bc the model with only 1 cpu at a time is super slow to run grid search
    )
    
    # Define feature columns
    feature_cols = ['calories (#)', 'minutes', 'tags', 'review', 'saturated fat (PDV)', 'total fat (PDV)']
    return grid_search, feature_cols

experimental_model, feature_cols = create_experimental_model()
display(experimental_model)

experimental_model.fit(X_train[feature_cols], y_train)

experimental_model.predict(pd.DataFrame([{
    'calories (#)': 10,
    'minutes': 70000,
    'tags': '60-minutes-or-less chicken-stew', # FOR NOW MUST BE A STRING NOT A LIST
    'review': 'So delicious!',
    'saturated fat (PDV)': 100, 
    'total fat (PDV)': 100
}]))[0]
# 16 minutes
# ON ALL TEST DATA UNCONDITIONAL OF GROUPING BY MSE OF AVG RATINGS
final_train_predictions = experimental_model.predict(X_train)
final_train_mse = mean_squared_error(y_train, final_train_predictions)
final_test_predictions = experimental_model.predict(X_test)
final_test_mse = mean_squared_error(y_test, final_test_predictions)

# printing mse on test
print("Baseline Model - Train MSE:", baseline_train_mse)
print("Final Model - Train MSE:", final_train_mse)
print("Baseline Model - Test MSE:", baseline_test_mse)
print("Final Model - Test MSE:", final_test_mse)
# On test data with two mean squared error estimates (mse of avg_rating > 4 and avg_rating < 4)
high_train_ratings = y_train > 4
low_train_ratings = y_train <= 4
high_test_ratings = y_test > 4
low_test_ratings = y_test <= 4

final_hr_train_predictions = experimental_model.predict(X_train[high_train_ratings])
final_lr_train_predictions = experimental_model.predict(X_train[low_train_ratings])
final_hr_train_mse = mean_squared_error(y_train[high_train_ratings], final_hr_train_predictions)
final_lr_train_mse = mean_squared_error(y_train[low_train_ratings], final_lr_train_predictions)

final_hr_test_predictions = experimental_model.predict(X_test[high_test_ratings])
final_lr_test_predictions = experimental_model.predict(X_test[low_test_ratings])
final_hr_test_mse = mean_squared_error(y_test[high_test_ratings], final_hr_test_predictions)
final_lr_test_mse = mean_squared_error(y_test[low_test_ratings], final_lr_test_predictions)

print("Final Model - Train MSE (avg_rating > 4):", final_hr_train_mse)
print("Final Model - Train MSE (avg_rating ≤ 4):", final_lr_train_mse)
print("Final Model - Test MSE (avg_rating > 4):", final_hr_test_mse)
print("Final Model - Test MSE (avg_rating ≤ 4):", final_lr_test_mse)
# MSE values for all recipes (Train vs Test)
labels_all = ["Baseline (Train)", "Baseline (Test)", "Final (Train)", "Final (Test)"]
mse_values_all = [baseline_train_mse, baseline_test_mse, final_train_mse, final_test_mse]
colors_all = ['blue', 'blue', 'orange', 'orange']

fig1 = go.Figure()

fig1.add_trace(go.Bar(
    x=labels_all,
    y=mse_values_all,
    marker_color=colors_all,
    text=[f'{val:.4f}' for val in mse_values_all],
    textposition='outside'
))

fig1.update_layout(
    title="MSE Comparison: All Recipes (Train vs Test)",
    xaxis_title="Model",
    yaxis_title="Mean Squared Error",
    showlegend=False,
    xaxis_tickangle=-15,
    template="plotly_white",
    margin=dict(l=40, r=40, t=60, b=60)
)

fig1.show()
# Save to html file
fig1.write_html('baseline_vs_final_all_mse_performance.html', include_plotlyjs = 'cdn')
import plotly.graph_objs as go
import plotly.express as px

# Labels and values
groups = ['> 4 Stars', '≤ 4 Stars']
splits = ['Train', 'Test']
models = ['Baseline', 'Final']

# Example MSE values — replace these with your actual computed MSEs
mse_values = {
    '> 4 Stars': {
        'Baseline': {'Train': baseline_hr_train_mse, 'Test': baseline_hr_test_mse},
        'Final': {'Train': final_hr_train_mse, 'Test': final_hr_test_mse}
    },
    '≤ 4 Stars': {
        'Baseline': {'Train': baseline_lr_train_mse, 'Test': baseline_lr_test_mse},
        'Final': {'Train':  final_lr_train_mse, 'Test': final_lr_test_mse}
    }
}

# Flatten data for plotting
x = []
y = []
colors = []
texts = []

for group in groups:
    for model in models:
        for split in splits:
            x.append(f"{group}<br>{model} ({split})")
            y.append(mse_values[group][model][split])
            colors.append('blue' if model == 'Baseline' else 'green')
            texts.append(f"{mse_values[group][model][split]:.3f}")

# Create Plotly bar chart
fig = go.Figure(data=[
    go.Bar(
        x=x,
        y=y,
        marker_color=colors,
        text=texts,
        textposition='auto'
    )
])

fig.update_layout(
    title='Train and Test MSE by Rating Group and Model',
    yaxis_title='Mean Squared Error',
    xaxis_tickangle=45,
    plot_bgcolor='white',
    margin=dict(l=40, r=40, t=60, b=100),
    height=500
)

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.show()
fig.write_html('baseline_vs_final_grouped_mse_performance.html', include_plotlyjs = 'cdn')
print("Best alpha:", experimental_model.best_params_['lasso__alpha']) # best alpha is 0.0001
print("Best word limit for reviews tf idf vectorizer:", experimental_model.best_params_['columntransformer__tfidfvectorizer-2__max_features']) # best word limit is 

import plotly.graph_objs as go
import numpy as np

results = experimental_model.cv_results_

# Extract data
alphas = np.array(results['param_lasso__alpha'].data, dtype=float)
max_features = np.array(results['param_columntransformer__tfidfvectorizer-2__max_features'].data, dtype=int)
val_mse = -np.array(results['mean_test_score'])  # Convert to positive MSE

log_alphas = np.log10(alphas)

# Prepare a grid of (log_alpha, max_features)
unique_log_alphas = np.unique(log_alphas)
unique_max_features = np.unique(max_features)

# Create a grid for the MSE values
mse_grid = np.zeros((len(unique_max_features), len(unique_log_alphas)))

# Fill the grid with mean MSE for each (alpha, max_features) combination
for i, alpha in enumerate(unique_log_alphas):
    for j, feature in enumerate(unique_max_features):
        mask = (log_alphas == alpha) & (max_features == feature)
        mse_grid[j, i] = np.mean(val_mse[mask])

# Create the surface plot
fig = go.Figure(data=[
    go.Surface(
        z=mse_grid,
        x=unique_log_alphas,  # log10(alpha)
        y=unique_max_features,  # max_features
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Validation MSE')
    )
])

fig.update_layout(
    title='Validation MSE Surface',
    scene=dict(
        xaxis=dict(title='log10(Alpha)'),
        yaxis=dict(title='Max Features'),
        zaxis=dict(title='Validation MSE')
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()

# Save to html file
fig.write_html('final_model_val_mse_surface_plot.html', include_plotlyjs = 'cdn')
