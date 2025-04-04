from sklearn.pipeline import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression]

def create_baseline_model(merged):
    # Delete later
    merged = merged.dropna(subset=['calories', 'minutes', 'avg_rating'])  # Drop rows with NaN
    # Define a function to log-transform features
    log_transformer = FunctionTransformer(lambda x: np.log1p(x), feature_names_out="one-to-one")
    # preprocessing the columns
    preprocessing = make_column_transformer(
        (log_transformer, ['calories', 'minutes']),
        remainder = 'drop'
    )
        
    # The pipeline
    model = make_pipeline(preprocessing, LinearRegression())
    
    # Define feature columns
    feature_cols = ['calories', 'minutes']
    return model.fit(merged[feature_cols], merged['avg_rating'])

baseline_model = create_baseline_model(merged)
baseline_model.predict(pd.DataFrame([{
    'calories': 560,
    'minutes': 20,
}]))[0]
