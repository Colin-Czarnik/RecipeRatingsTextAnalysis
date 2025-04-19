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

Another feature we had to clean is nutrition. We want to use the nutritional information as separate features in our model, but they are initially stored in a string that takes the form of a list. The first step is to use Regular Expressions to convert this string into a list of floats. Each value in this list corresponds to a nutrition information value (either # of calories, or %DV of others). The next step is to make each element of this list into its own column, allowing us to use each of them as individual features. 
(will add more on tratment of tags and how we handled imputation)



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
1. Ratio of Saturated Fat (PDV) to Total Fat (PDV) -
2. TF IDF Vectorizer of **Tags** -
3. TF IDF Vectorizer of **Reviews** -

The modeling algorithm we chose for our final model was **Lasso Regression** since from our two TF IDF Vectorizer Transformations of the "Tags" and "Reviews", we would gain a lot of new features for each individual unique tag as well as for each individual unique token in the entire corpus of reviews in the dataset. Performing this kind of regression algorithm helps the model to select which features are the most important in making predictions for a recipe's average rating and to prevent the model from getting too complex from the large quantity of columns added to our feature matrix.

The hyperparameters we chose to tune in the cross validation Grid Search performed on this final model were:
1. Lasso Alpha - 
2. Max Features for TFIDF Vectorizer of Reviews -

The plot below shows the overall train and test performance of the final model in terms of Mean Squared Error compared to the baseline model:

<iframe
src="assets/baseline_vs_final_all_mse_performance.html"
width="800"
height="600"
frameborder="0"
></iframe>

The plot below shows how the final model performed in predicting recipes with high and low average ratings in comparison to the baseline model:

<iframe
src="assets/baseline_vs_final_grouped_mse_performance.html"
width="800"
height="600"
frameborder="0"
></iframe>

Lastly, the figure below details a surface plot showing the Average Mean Squared Error on the Validation Set during Cross Validation across different settings of the hyperparameters tuned:

<iframe
src="assets/final_model_val_mse_surface_plot.html"
width="800"
height="600"
frameborder="0"
></iframe>
