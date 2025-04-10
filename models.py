# preprocessing to do before model evaluation

merged = merged.assign(**{
    'calories (#)' : merged['nutrition'].str[0],
    'total fat (PDV)' : merged['nutrition'].str[1],
    'sugar (PDV)' : merged['nutrition'].str[2],
    'sodium (PDV)' : merged['nutrition'].str[3],
    'protein (PDV)' : merged['nutrition'].str[4],
    'saturated fat (PDV)' : merged['nutrition'].str[5],
    'carbohydrates (PDV)' : merged['nutrition'].str[6],
    })


from sklearn.pipeline import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

# make baseline model
def create_baseline_model(merged): # Concatenating average
    # Delete later
    merged_copy = merged.dropna(subset=['calories (#)', 'minutes', 'avg_rating'])  # Drop rows with NaN
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
    return model.fit(merged_copy[feature_cols], merged_copy['avg_rating'])

baseline_model = create_baseline_model(merged)
baseline_model.predict(pd.DataFrame([{
    'calories (#)': 1, # slight positive coefficient, more caloric recipes more favorable
    'minutes': 8000000 # slight negative coefficient, longer recipes less favorable
}]))[0]


# make experimental model
def create_experimental_model(merged): # For average rating, we concatenate all tags into a large string, and then use that string to predict avg rating. Avg calories and minutes
    # Delete later
    merged_copy = merged.dropna(subset=['calories (#)', 'minutes', 'avg_rating', 'tags'])  # Drop rows with NaN
    # Turn tag list into tag string
    merged_copy.loc[:, 'tags'] = merged_copy['tags'].apply(lambda x: ' '.join(x))
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
        (fat_ratio_pipeline, ['saturated fat (PDV)', 'total fat (PDV)']),
        remainder='drop'  # Drop other columns not being used
    )
        
    # The pipeline
    # model = make_pipeline(preprocessing, Lasso()) # apply l1 regularization
    param_grid = {'lasso__alpha': [2**i for i in range(-3, 4)]}  # Range from 2^-3 to 2^3
    grid_search = GridSearchCV(
        make_pipeline(preprocessing, Lasso()),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error'
    )
    
    # Define feature columns
    feature_cols = ['calories (#)', 'minutes', 'tags', 'saturated fat (PDV)', 'total fat (PDV)']
    return grid_search.fit(merged_copy[feature_cols], merged_copy['avg_rating'])

baseline_model = create_experimental_model(merged)
baseline_model.predict(pd.DataFrame([{
    'calories (#)': 10,
    'minutes': 70000,
    'tags': '60-minutes-or-less chicken-stew', # FOR NOW MUST BE A STRING NOT A LIST
    'saturated fat (PDV)': 100, 
    'total fat (PDV)': 100
}]))[0]
