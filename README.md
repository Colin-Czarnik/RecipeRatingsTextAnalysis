# Data Analysis of Recipes and Ratings Dataset

## Introduction

Whether you are a michelin-star chef who works for *Hell's Kitchen*, or a humble home cook who loves to try out new recipes, we often find ourselves searching for new recipes and wondering what really makes one stand out from the rest. Is it the ingredients involved, the reviews attributed to that recipe, or maybe the macronutrient distributions? We explore this question further as we analyze two datasets of recipes and user interactions from [Food.com](https://www.food.com). 

The data includes both details on the recipe itself, such as the ingredients, nutritional information, and the steps in the recipe; as well as the details on a user's interaction with the recipe including their rating of the recipe, the date in which they submitted their review, and the review itself. With this information, we aim to predict a recipe's average rating based on these features to better understand what makes recipes successful, hoping to provide insight on how recipes can be improved or better tailored ot individual preferences. The end goal of this analysis is to use our predictive models to support smarter food recommendation systems and to offer a data-driven perspective on what makes a recipe truly stand out.

In the table below, we describe the features in our dataset that will be important for our data analysis and model building. This dataset describes two merged datasets that describe the recipe metadata and the user interactions, containing 1,132,367 rows along with the following relevant columns with their corrresponding descriptions.

| Feature Name | Description |
| ----------- | ----------- |
| tags | Tags assigned by Food.com to describe the recipe (e.g. "60-minutes-to-make") |
| minutes | Minutes to prepare recipe |
| calories (#) | Total number of calories in recipe |
| total fat (PDV) | Percent daily value (%DV) of total fat per serving |
| sugar (PDV) | Percent daily value (%DV) of sugar per serving |
| sodium (PDV) | Percent daily value (%DV) of sodium per serving |
| protein (PDV) | Percent daily value (%DV) of protein per serving |
| saturated fat (PDV) | Percent daily value (%DV) of saturated fat per serving |
| carbohydrates (PDV) | Percent daily value (%DV) of carbohydrates per serving  |
| review | Free-text review of the recipe left by a user |
| rating | A user-submitted score (0–5 stars) reflecting their evaluation of the recipe's quality |

## Data Cleaning and Exploratory Data Analysis

Our first task with this dataset is to clean the the data, this includes converting strings into numerical features and properly handling null values. One important thing is the value of rating. With some digging, we found that reviews of 0 are actually people who did not leave a rating on the recipe, leaving only a review. A good indicator of this is user 452045 reviewing recipe id 433334: "I'm not rating this because I have not made it but I am suggesting that ... references to adding 12 cups of flour!" As can be seen at the start of the text of the review, this user indicated that they were not leaving a rating, yet the rating was listed as a 0. This justifies changing all ratings of 0 to NA, as they shouldn't be included as a bad review when they're really neutral.

Another feature we had to clean is nutrition. We want to use the nutritional information as separate features in our model, but they are initially stored in a string that takes the form of a list. The first step is to use Regular Expressions to convert this string into a list of floats. Each value in this list corresponds to a nutrition information value (either # of calories, or %DV of others). The next step is to make each element of this list into its own column, allowing us to use each of them as individual features. We are going to do text analysis on the tags, column, so we convert the string that looks like a list into a string of the tags separated by a space. This will be easier to work with later when performing text analysis.

For missing values, there was luckily very little to worry about in our dataset. The only NA values in our dataset in the columns we are working with are in the Average Rating column that we just set. Since we are focusing on predicting the ratings based on the features of the recipes for the most part, we felt it was fair to remove these NA values from our dataset, as these people specifically did not leave a rating for a reason. We feel that any form of imputation here would be adding unnecessary noise to our data. The recipes with NA for Average Rating only make up about 3% of our dataset. There was also exactly 1 missing value in the Minutes column, and with it being such a minor difference, we felt safe in removing that recipe too.

For our initial Exploritory Data Analysis, we looked at the distribution of the Average Rating column, as it is the column we will be predicting here. We found that the ratings were heavily skewed towards higher values, with over 60% of the recipes having average ratings higher than 4.75, and over 90% of recipes having average ratings of 4 or higher. This impacts how we will handle our model evaluation, as we want our model to perform well on all ratings, not just the higher ones. The histogram of the Average Rating column can be seen below.
<iframe
src="assets/rate_hist.html"
width="800"
height="600"
frameborder="0"
></iframe>

We also performed bivariate analysis on Average ratings based on two factors that we will be working with in our baseline model, minutes and calories. Shown below are two plots that show the comparison of calories and minutes to average ratings, with side-by-side box plots based on the quantiles of the values. A point showing the mean value is also displayed.
<iframe
src="assets/fat_box.html"
width="800"
height="600"
frameborder="0"
></iframe>
Shown below is a pivot table showing the average ratings based on the quantiles of the data.

| Calories Bins  / Minutes Bins |   (0.999, 16.0] |   (16.0, 30.0] |   (30.0, 45.0] |   (45.0, 75.0] |   (75.0, 1051200.0] |
|:-----------------|----------------:|---------------:|---------------:|---------------:|--------------------:|
| (-0.001, 146.5]  |         4.69033 |        4.6081  |        4.56837 |        4.63054 |             4.62424 |
| (146.5, 248.9]   |         4.67717 |        4.62631 |        4.60803 |        4.61518 |             4.61388 |
| (248.9, 370.6]   |         4.65487 |        4.62798 |        4.59572 |        4.60872 |             4.58442 |
| (370.6, 563.3]   |         4.66003 |        4.63333 |        4.61248 |        4.60986 |             4.60022 |
| (563.3, 45609.0] |         4.63045 |        4.62148 |        4.63718 |        4.61324 |             4.62323 |



## Framing a Prediction Problem

Building off our exploratory analysis, our goal is to predict the **average rating** (avg_rating) of a recipe based on various features such as tags attributed to the recipe, preparation time, nutritional information, and user reviews. This is a regression problem, as the target variable, avg_rating, is continuous. We chose avg_rating as the response variable because it reflects how well a recipe is received by users. In contrast, individual **rating** values can vary greatly from user to user. Aggregating these ratings into an average allows us to capture a more holistic understanding of how well a recipe stands out. Understanding the factors that influence this rating can offer valuable insights into what makes a recipe successful, helping future recommendations and recipe development.

To evaluate our model, we use Mean Squared Error (MSE). Since our target is continuous, MSE is an appropriate metric as it penalizes larger errors more heavily, which helps us prioritize models that make fewer large mistakes in prediction. It also provides a clear, interpretable measure of overall prediction accuracy in the context of numerical outcomes.

In our modeling approach, we aggregate all available reviews for each recipe into a single, combined text input. This means that our model has access to the collective feedback from users across the entire lifespan of a recipe — not just reviews submitted shortly after its publication. As a result, the features we use at prediction time, including the combined review text, reflect the long-term perception and reception of the recipe. While this would not be appropriate for a real-time prediction system (e.g., predicting success before any reviews exist), it aligns well with our project’s objective: to understand the overall factors that contribute to a recipe’s long-term success. Thus, the features that we will have access to by the time of prediction are the following: tags, minutes, calories (#), total fat (PDV), saturated fat (PDV), review. Note that these are all of the features we use for the final model.

By aggregating reviews and using the final average rating as the response variable, we aim to build a model that captures the full story of each recipe — a comprehensive reflection of how it is received by users over time.

## Baseline Model

Our baseline model is a simple **linear regression** pipeline that uses the **calories (#)** and **minutes** of a recipe to predict its **average rating (avg_rating)**, with log tranformations on both of the features to account for magnitudes of calories and minutes for certain recipes (e.g. vanilla extract which takes about half a year).

In this model, there are **2** quantitative features, **0** ordinal features, and **0** nominal features. In addition, since the two features used in the baseline model are already numerical, no additional encoding is necessary.

Below are the results detailing the Mean Squared Error (MSE) of the baseline model for the training and test sets:
<iframe
src="assets/baseline_model_mse_performance.html"
width="800"
height="600"
frameborder="0"
></iframe>

Based on our MSE plot, we can see that the *overall* performance of the baseline model is decent with a test MSE of 0.4118 considering the **avg_rating** feature ranges from 1 to 5. However, when we split its MSE to average ratings above 4 stars and below or equal to 4 stars, we reveal some interesting behavior.
Specifically, our baseline model is quite poor at predicting recipes that have an average rating of 4 stars or less. This is important to note since in our Exploratory Data Analysis, we found through our univariate plot that the distribution of ratings in the dataset was left skewed, meaning that the majority of ratings were 4 stars or higher, causing a sort of imbalance when predicting recipes with high ratings versus lower ratings.
When we build our final model, our primary goal is to minimize the test MSE of recipes with average ratings of 4 stars or less as much as possible, while still maintaining good predictions for recipes with higher average ratings.
 
## Final Model

For our final model, we added 3 new features in addition to the **calories (#)** and **minutes** included in the baseline model:
1. Ratio of Saturated Fat (PDV) to Total Fat (PDV) - This feature provides a standardized measure of how much of the fat content in a recipe comes from saturated fats, often perceived as being unhealthy. Given that user ratings may be influenced by health perceptions, particularly in recipes tagged as "healthy" or "low-calorie", this ratio serves a more interpretable and consistent health-related signal than either of these values individually. Including this feature can help the model capture a user's potential preference for healthier or unhealthy recipes, which is not accounted for in the raw calorie count highlighted in the baseline model.
2. TF-IDF Vectorizer of **Tags** - Tags are high-level descriptors like "60-minutes-or-less" or "for-large-groups" that summarize the nature or context of the recipe. By applying a TF-IDF vectorizer to these tags, we allow the model to consider each of these individual tags that can encode potentially useful information about recipe themes or dietary types that users might favor, making them highly relevant for predicting average ratings.
3. TF-IDF Vectorizer of **Reviews** - Reviews naturally provide a rich source of user sentiment and opinion. By transforming the reviews using TF-IDF, we can extract meaningful words (tokens) that are often associated with positive or negative expereinces. For example words that falls under negative connotation like "bland", "disgusting", or "burnt" carry sentiment and nuance that strongly influence a recipe's rating. Including this additional vectorized text can give the model access to feedback directly tied to the average rating of a recipe, helping it to learn various patterns tied to user preferences and satisfaction.

The modeling algorithm we chose for our final model was **Lasso Regression** since from our two TF-IDF Vectorizer Transformations of the "Tags" and "Reviews", we would gain a lot of new features (columns) for each individual unique tag as well as for each individual unique token in the entire corpus of reviews in the dataset. Performing this kind of regression algorithm helps the model to select which features are the most important in making predictions for a recipe's average rating and to prevent the model from getting too complex from the large quantity of columns added to our feature matrix. Specifically, the addition of L1 regularization to the final model encourages sparsity in the learned coefficients, helping the model automatically select the most relevant features by shrinking the coefficients of less important features to zero, effectively acting as a built-in feature selector. This can help to reduce the chance of overfitting to the training data and to improve the interpretability of our model.

To tune our model and ensure optimal performance, we used GridSearchCV, a form of exhaustive cross-validation that evalutes model performance over many combinations of hyperparameters.

The hyperparameters we chose to tune in the cross validation Grid Search performed on this final model were:
1. Lasso Alpha - This controls the strength of regularization. Smaller values allow more features to have nonzero coefficients, while larger values enforce more aggressive feature selection.
2. Max Features for TF-IDF Vectorizer of Reviews - This limits the number of unique words (tokens) considered from the reviews, ensuring that only the most informative words (by frequency and uniqueness across documents) are included in the feature matrix. This parameters controls the trade-off between the valuable information of the textual features and the complexity of the model.

The plot below shows the overall train and test performance of the final model in terms of Mean Squared Error compared to the baseline model:
<iframe
src="assets/baseline_vs_final_all_mse_performance.html"
width="800"
height="600"
frameborder="0"
></iframe>

We can see that overall the final model performed much better on both the training and test data with neither of the model's showing obvious signs of overfitting to the training data. However, this plot does not give the whole story, since in our baseline model we saw that it only performed well on recipes that already had high average ratings (greater than 4 stars) which represents the majority of the recipes in the dataset. So we also want to evaluate how the final model did on recipes with less flattering average ratings to see how the additional features helped in distinguishing good from bad recipes.

The plot below shows how the final model performed in predicting recipes with high and low average ratings in comparison to the baseline model:
<iframe
src="assets/baseline_vs_final_grouped_mse_performance.html"
width="800"
height="600"
frameborder="0"
></iframe>

While the final model's MSE in both the train and test data for recipes with lower average ratings were still noticably higher than the MSE for recipes with higher average ratings, the final model's MSE for recipes with lower average ratings is much lower than the baseline model's MSE for recipes with lower average ratings. In fact we see over a **50% reduction** in both the train and test MSE of the final model in recipes with lower average ratings. This is great to see because it validates these features as being helpful to predict average ratings of a recipe, as we are able to help resolve the issue introduced from our baseline model. Specifically, the final model was able to predict the average ratings of recipes that have a worse sentiment from users much more effectively than the baseline model.

Lastly, the figure below details a surface plot showing the Average Mean Squared Error on the Validation Set during Cross Validation across different settings of the hyperparameters tuned:
<iframe
src="assets/final_model_val_mse_surface_plot.html"
width="800"
height="600"
frameborder="0"
></iframe>

We can see that after a certain threshold, the Validation MSE plummets for the final model when we decrease our alpha for our Lasso Regression pipeline, with the number of max features of the TF-IDF Vectorizer for "Reviews" not making as much of a significant contribution in the performance of the final baseline. This signals that regularization strength has a much greater impact on model performance than the size of the TF-IDF vocabulary. In other words, once enough informative review terms are included in the model, further increasing the number of features yields diminishing returns. 
Notably, the best-performing model used the smallest alpha value (10^-5^), implying that very minimal regularization yielded the lowest average validation error. This suggests that our final model actually benefits from retaining more and more of the original feature weights, and that the data (particuarly the textual reviews) contains enough valuable information to justify a more flexible and complex model.
In summary, while increasing the number of review features helps up to a certain extent, allowing the model more freedom to leverage these features with a smaller regaularization penalty leads to the greatest performance gains.
