from sklearn.pipeline import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, Ridge

# make baseline model
def create_baseline_model(merged): # Concatenating average
    # Delete later
    merged_copy = merged.dropna(subset=['calories', 'minutes', 'avg_rating'])  # Drop rows with NaN
    # transform numerical columns
    log_transformer = FunctionTransformer(lambda x: np.log1p(x))
    
    preprocessing = make_column_transformer(
        (log_transformer, ['calories', 'minutes']),  # Apply log transformation to 'calories' and 'minutes'
        remainder='drop'  # Drop other columns not being used
    )
        
    # The pipeline
    model = make_pipeline(preprocessing, LinearRegression()) # apply l1 regularization
    
    # Define feature columns
    feature_cols = ['calories', 'minutes']
    return model.fit(merged_copy[feature_cols], merged_copy['avg_rating'])

baseline_model = create_baseline_model(merged)
baseline_model.predict(pd.DataFrame([{
    'calories': 1, # slight positive coefficient, more caloric recipes more favorable
    'minutes': 8000000 # slight negative coefficient, longer recipes less favorable
}]))[0]


# make experimental model
def create_experimental_model(merged): # For average rating, we concatenate all tags into a large string, and then use that string to predict avg rating. Avg calories and minutes
    # Delete later
    merged_copy = merged.dropna(subset=['calories', 'minutes', 'avg_rating', 'tags'])  # Drop rows with NaN
    # Turn tag list into tag string
    merged_copy.loc[:, 'tags'] = merged_copy['tags'].apply(lambda x: ' '.join(x))
    # transform numerical columns
    log_transformer = FunctionTransformer(lambda x: np.log1p(x))
    
    preprocessing = make_column_transformer(
        (log_transformer, ['calories', 'minutes']),  # Apply log transformation to 'calories' and 'minutes'
        (TfidfVectorizer(), 'tags'),  # Apply TF-IDF vectorization to 'tags'
        remainder='drop'  # Drop other columns not being used
    )
        
    # The pipeline
    model = make_pipeline(preprocessing, Lasso(alpha=1)) # apply l1 regularization
    
    # Define feature columns
    feature_cols = ['calories', 'minutes', 'tags']
    return model.fit(merged_copy[feature_cols], merged_copy['avg_rating'])

baseline_model = create_experimental_model(merged)
baseline_model.predict(pd.DataFrame([{
    'calories': 10,
    'minutes': 70000,
    'tags': '60-minutes-or-less chicken-stew' # FOR NOW MUST BE A STRING NOT A LIST
}]))[0]
