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

## Framing a Prediction Problem

Building off our exploratory analysis, our goal is to predict the **average rating** (avg_rating) of a recipe based on various features such as tags attributed to the recipe, preparation time, nutritional information, and user reviews. This is a regression problem, as the target variable, avg_rating, is continuous. We chose avg_rating as the response variable because it reflects how well a recipe is received by users. In contrast, individual **rating** values can vary greatly from user to user. Aggregating these ratings into an average allows us to capture a more holistic understanding of how well a recipe stands out. Understanding the factors that influence this rating can offer valuable insights into what makes a recipe successful, helping future recommendations and recipe development.

To evaluate our model, we use Mean Squared Error (MSE). Since our target is continuous, MSE is an appropriate metric as it penalizes larger errors more heavily, which helps us prioritize models that make fewer large mistakes in prediction. It also provides a clear, interpretable measure of overall prediction accuracy in the context of numerical outcomes.

In our modeling approach, we aggregate all available reviews for each recipe into a single, combined text input. This means that our model has access to the collective feedback from users across the entire lifespan of a recipe — not just reviews submitted shortly after its publication. As a result, the features we use at prediction time, including the combined review text, reflect the long-term perception and reception of the recipe. While this would not be appropriate for a real-time prediction system (e.g., predicting success before any reviews exist), it aligns well with our project’s objective: to understand the overall factors that contribute to a recipe’s long-term success. Thus, the features that we will have access to by the time of prediction are the following: tags, minutes, calories (#), total fat (PDV), saturated fat (PDV), review. Note that these are all of the features we use for the final model.

By aggregating reviews and using the final average rating as the response variable, we aim to build a model that captures the full story of each recipe — a comprehensive reflection of how it is received by users over time.

## Baseline Model

## Final Model
