Authors: Madhav Pudukottai Krishnaraj, Yaqub Mir
# Overview
We noticed that it was very rare to come by a rating that was not a 5, and wanted to dive deeper into what really prompts a user to give a rating less than a 5, how new a user is to the app, the recipe was bad, or maybe it was just the user's preference. 
# Introduction
Recipe ratings on Food.com are overwhelmingly positive (over 70 percent of ratings are 5s), with most users giving 5-star reviews. This makes it difficult to understand what actually leads people to rate a recipe lower. In this project, we analyze thousands of recipes and interactions to uncover patterns behind ratings below 5, study how user behavior affects scoring, and build predictive models to classify when a rating will be “good” or “bad.”

The two datasets we use in this project are derived from [food.com](https://www.food.com/) and contain information on recipes and interactions with recipes from 2008 - 2018. The original purpose of the datasets is for the recommender system research paper, [Generating Personalized Recipes from Historical User Preferences](https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19c.pdf) by Majumder, Li, Ni, and McAuley.

We used the `recipe` and `interactions` datasets to complete this project. Here's a look at both of them.

`recipe`

| Column           | Description                                                                                                                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`           | Recipe name                                                                                                                                                                                       |
| `id`             | Recipe ID                                                                                                                                                                                         |
| `minutes`        | Minutes required to prepare the recipe                                                                                                                                                            |
| `contributor_id` | User ID of the person who submitted the recipe                                                                                                                                                    |
| `submitted`      | Date the recipe was submitted                                                                                                                                                                     |
| `tags`           | List of Food.com tags associated with the recipe                                                                                                                                                  |
| `nutrition`      | Nutrition info in the form `[calories (#), fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV = percentage of daily value                      |
| `n_steps`        | Number of steps in the recipe                                                                                                                                                                     |
| `steps`          | Ordered text instructions for making the recipe                                                                                                                                                   |
| `description`    | Optional user-provided recipe description                                                                                                                                                         |
| `ingredients`    | List of all ingredients in the recipe                                                                                                                                                             |
| `n_ingredients`  | Total number of ingredients in the recipe                                                                                                                                                         |
                                                 

(83782 rows and 10 columns)

`interactions`

| Column      | Description                                              |
|-------------|----------------------------------------------------------|
| `user_id`   | ID of the user who submitted the rating or review        |
| `recipe_id` | ID of the recipe the user interacted with                |
| `date`      | Date the rating or review was submitted                  |
| `rating`    | Rating given by the user (0–5, with 0 indicating missing)|
| `review`    | Optional text review provided by the user                |

(731927 rows, 5 columns)

# Data Cleaning and Exploratory Analysis

When we read in our data, it was not ideal to work with for our exploratory analysis, so we conducted the following steps to properly clean our data. 

### Data Cleaning Steps

1. **Left merge the recipes and interactions datasets** on `id` and `recipe_id`.  
   This ensured that every review in the interactions dataset matched a recipe that we had metadata for.

2. **Replace all ratings of 0 with `np.nan`**, since a rating of 0 indicates missingness in the Food.com dataset rather than an actual rating.

3. **Convert the `nutrition` column from a string to a list**, since the Food.com dataset stores nutritional information as a stringified list.  
   - After converting to a true list, each element was cast to `float` so nutritional values could be used numerically.

4. **Convert the `submitted` column to datetime** to allow more flexible and accurate time-based operations.

5. **Convert `user_id` and `recipe_id` back to integers.**  
   These columns became floats after merging, which was both visually messy and incorrect for ID fields.

6. **Create the `user_review_count` feature** by grouping by `user_id` and counting how many total reviews each user has written.  
   This feature was merged back into the combined dataset, and the row count remained the same (234,429), confirming the merge did not duplicate or drop rows.

7. **Examine correlations between numeric columns and rating.**  
   The largest coefficient was associated with `user_id`, which (despite being categorical) showed a weak negative relationship with rating. This may suggest that newer users (with higher user IDs) tend to rate slightly lower on average.



(Our final dataframe after all this had 234429 and 26 columns, but here is a shortened dataset of the columns we actually used)

### Combined Dataset Preview

|     | id     | year  | rating | user_id    | user_review_count |
|-----|--------|-------|--------|------------|-------------------|
| 0   | 333281 | 2008  | 4.0    | 386585     | 959.0             |
| 1   | 453467 | 2012  | 5.0    | 424680     | 4934.0            |
| 2   | 306168 | 2008  | 5.0    | 29782      | 73.0              |
| 3   | 306168 | 2009  | 5.0    | 1196280    | 1.0               |
| 4   | 306168 | 2013  | 5.0    | 768828     | 70.0              |
| ... | ...    | ...   | ...    | ...        | ...               |
| 234424 | 308080 | 2009 | 5.0  | 844554     | 492.0             |
| 234425 | 298512 | 2008 | 1.0  | 804234     | 1.0               |
| 234426 | 298509 | 2008 | 1.0  | 866651     | 1.0               |
| 234427 | 298509 | 2010 | 5.0  | 1546277    | 1.0               |
| 234428 | 298509 | 2014 | NaN  | 1803287907 | 1.0               |


# Univariate Analysis 

For our univariate analysis, we looked at the distribution of reviewer experience or our reviewer counts columns distribution. We found that it had a long left tail, which means that more people have less reviews and few people have a lot of reviews. We found that the distrution had a majority between 0 and 400 reviews.

<iframe
  src="univar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

# Bivariate Analysis

For our bivariate analysis, we examined how average recipe ratings differ across three reviewer experience groups. We created these groups using pd.qcut on the user_review_count column, which evenly splits users into low, medium, and high activity categories based on how many reviews they have previously written.

When comparing the mean ratings across these groups higher-experience users had the highest average ratings, while lower-experience users had the lowest. Meaning more active reviewers tend to rate recipes more higher on average, whereas users with little reviewing history are more likely to give lower scores.

<iframe
  src="bivar.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

# Interesting Aggregates 

To better understand why certain recipes fail to receive a perfect 5-star rating, we created groups that represent different possible explanations for non-perfect ratings. Each recipe was assigned to one of four categories: Perfect, User-Preference/Flaw, Recipe Flaw Likely, and Disagreement-Driven. These categories were constructed by **aggregating** both the mean and standard deviation of each recipe’s ratings and applying thresholds (mean < 4) (sd > 1) to separate them meaningfully.

The logic behind these groups is as follows:

 - Perfect recipes have an average rating of exactly 5 and therefore require no further explanation.

 - Recipes with a high mean rating but low standard deviation indicate that users generally agree the recipe is good. If such a recipe does not achieve a perfect score, it is likely due to individual user preference or user-specific issues, rather than the recipe itself.

 - Recipes with a high mean rating but high standard deviation show strong disagreement among reviewers. Although the average may appear strong, the wide variation suggests that some users rated the recipe much lower. These cases fall under Disagreement-Driven, meaning the outcome cannot be attributed solely to recipe quality or user behavior.

 - Finally, recipes with a mean rating below 4 are categorized as Recipe Flaw Likely, since ratings this low are rare in the dataset and strongly suggest genuine quality issues, regardless of variation.

This categorization provides a structured way to interpret rating behavior beyond simple averages. It also highlights that non-perfect ratings can stem from diverse sources — not just recipe flaws, but also user variability and disagreement across reviewers.

### Distribution of Issue Types (mean < 4) (sd > 1)

| Issue Type              | Proportion |
|-------------------------|-----------:|
| Perfect                 | 0.588669   |
| User-Preference/Flaw    | 0.312739   |
| Recipe Flaw Likely      | 0.065909   |
| Disagreement-Driven     | 0.032683   |


<iframe
  src="interestingagg.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> 

**This graph shows us the distribution of different groups, the majority of points are clustered at 5**

# Missingness Analysis

# NMAR Analysis 
We believe the text column in the reviews dataset is Not Missing At Random. New or less experienced reviewers may feel less confident sharing their opinions, especially knowing that other, more experienced users can see and respond to their reviews. Because of this, newer reviewers might avoid writing a text review out of fear of being judged or saying something that goes against the general opinion. This means the likelihood of writing a review depends on who the user is and how comfortable they feel, not on random chance.

# Dependency Testing 
For this portion of the project, we tested whether the rating column's missingness is dependent on other columns or if it was missing completely at random. The two columns we tested on were **user_review_counts** and **minutes** 

## User Review Count vs Rating Missing Analysis

**Null Hypothesis:** The missingness of the ratings does not depend on the user review count of the user rating the recipe


**Alternate Hypothesis:** The missingness of the ratings does depend on the user review count of the user rating the recipe


**Test statistic:** = The difference of means in the user review count of the distribution of the population without missing ratings and with missing ratings

**Significance Level:** = 0.05

<iframe
  src="userreviewcountdist.png"
  width="800"
  height="600"
  frameborder="0"
></iframe> 

We ran a permutation test, sampling 10000 random permutations of ratings to simulate 10000 different test statistics so we could see if the observed test statistic was statistically significant, and if there was a missing dependency with User review counts

<iframe
  src="userreviewhyp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> 

**Since our p-value was 0, which is below the 0.05 significance level, we can reject the null hypothesis. This means there is significant statistical evidence that the missingness in the rating column is not random and depends on the user review count, meaning rating missingness is MAR.**

## Minutes vs Rating Missing Analysis

**Null Hypothesis:** The missingness of the ratings does not depend on the amount of minutes it takes to complete a recipe


**Alternate Hypothesis:** The missingness of the ratings does depend on the amount of minutes it takes to complete a recipe

**Test statistic:** = The difference of means in the minutes it takes for a recipe of the distribution of the population without missing ratings and with missing ratings

**Significance Level:** = 0.05

<iframe
  src="minutesdist.png"
  width="800"
  height="600"
  frameborder="0"
></iframe> 

We ran a permutation test, sampling 1000 random permutations of ratings to simulate 1000 different test statistics, so we could see if the observed test statistic was statistically significant, and there is a Missing dependency with minutes

<iframe
  src="minutesmissinghyp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> 

**Because our p-value was 0.1171, which is greater than the 0.05 significance level, we cannot reject the null hypothesis. We do not have enough statistical evidence to say that rating missingness depends on the minutes column.**

# Hypothesis Testing

The goal of this project was to see why ratings under 5 stars happen and throughout or bivariate and univariate analysis there were many fingers pointing towards how many reviews a user has given before or a users amount of experience on the app. To determine whether reviewer experience affects rating behavior, we performed a permutation test comparing the average ratings of Low Activity users to those of all other users. The reviewer groups were created earlier using pd.qcut on the user_review_count column, allowing us to classify users into Low, Medium, and High activity levels. Our goal being to test whether low-activity users tend to give lower ratings than more experienced reviewers. 

**Null Hypothesis:** Low activity users rate on the same scale as higher activity users (split is classified from the pdqcut in the exploratory analysis portion)


**Alternative Hypothesis:** Low activity users tend to rate on a lower scale compared to higher activity users 


**Test Statistic:** The difference in mean rating between low activity users and higher activity users

**Significance Level:** = 0.05

We performed the permutation test by simulating 1000 different permutations and computing 1000 simulated test statistics to find if the observed test statistic was statistically significant, and low activity users really do rate lower than higher activity users. We used a permutation test because we saw it fit well with our question, since we wanted to see if two different groups of ratings came from the same population.  

<iframe
  src="hypothesistest.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> 

Since our p-value was 0, which is below the 0.05 significance level, we can reject the null hypothesis. This gives us strong statistical evidence that low-activity users rate recipes lower than other users, showing that reviewer experience plays a noticeable role in rating behavior. This could be true because there is higher variation in the average rating for a newer user as they have not rated as many recipes, so they tend to just give lower ratings at the beginning, and as they get better at cooking or as they find better recipes they tend to rate things higher. 

# Framing a Prediction Problem
For our prediction task, we chose a binary classification problem. We aim to build a model that predicts whether a recipe rating will be good or bad. We define good ratings as scores of 4 or 5, while ratings of 1, 2, or 3 are treated as bad. This setup allows us to focus on distinguishing strong user approval from more critical reviews, and it gives our model a clear yes-or-no decision to learn from.  By structuring the task this way, we can investigate which factors are most useful for identifying lower ratings and evaluate how well our model can detect them. We used the F1 score to evaluate our models because it gives us a single metric that balances precision and recall, helping us evaluate the model’s overall performance when dealing with imbalanced classes like good and bad ratings, however the biggest change between the models that we wanted to see was in the precision, as because of class imbalance the baseline almost always predicted True. 
### Why not 5-star vs less than 5-star 
The reason we moved away from predicting a 5 vs everything else was that we tried doing this, and we realized that by doing this, we capped how good our model can be, as the difference between a 4 and a 5 was very minimal, and it was just very hard to predict with a high F1 score. 
# Baseline Model
For our baseline model, we trained a decision tree classifier. To train this model, we used the **user_review_counts** column and we created a new column called **recipe_review_count**, which just has the amount of reviews a recipe has as features. 

The baseline model achieved an **F1 score of around 0.93**, which may seem strong at first, but this performance is largely due to heavy class imbalance in our dataset. Since most ratings are good (4 or 5), the model tends to predict “good” by default most of the time, leading to inflated overall performance but relatively weak precision when identifying the less common bad ratings. This gives us a reasonable starting point, but also shows that we need a more expressive model to handle the imbalance and improve the quality of predictions on the minority class.

<iframe
  src="baselinemodelconfusionmatrix.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> 

This confusion matrix shows that it rarely gets "negative" reviews right and has a lot of False positives. 

# Final Model

For our final model, we used three main features, `review_text`, `user_avg_rating_past`, and `recipe_review_count`


### `review_text` (TF-IDF)

This column contains the text from each user’s review. We included it because review language often reflects sentiment and satisfaction. TF-IDF vectorizer allows us to convert the text into numerical features that show us which words matter most in order to classify a review as a 1. Since the written text provides direct insight into a user’s experience, this feature became one of the strongest predictors in our model.


### `user_avg_rating_past`

This feature measures a user’s average rating outside of the current review. We learned that users differ widely in their rating behavior — low-activity users tend to rate lower, and some users consistently give higher ratings than others. Including this feature helps the model learn each user’s general rating style. We engineered it carefully to avoid leakage by subtracting the current rating when computing the cumulative average. (So we are not using the value to predict itself) This was basically just a more meaningful way of using user_review_counts as we used technically used this feature (-1) to compute the average raitng by a user. (Denominator of the equation used to compute this feature)


### `recipe_review_count`

This feature tracks how many reviews a recipe had received before the current one. We thought this would be a good feature because if a recipe has a lot of reviews, it probably means a lot of people liked it. Including this feature helps the model account for whether a recipe is already relatively new or well established, the latter probably helps it predict 1s accurately based off our reasoning.


## Modeling Approach

We used a **ColumnTransformer** to apply TF-IDF to the review text and a mean imputer to the other features. The classifier for the final model was **Logistic Regression**, and we used GridSearchCV to tune the hyperparameters `C` (regularization strength) and `class_weight`. The best combination was `C = 1` and no class weighting.

Logistic Regression works well with high-dimensional TF-IDF data and avoids the overfitting issues that can occur with more complex models when dealing with sparse text features.


## Results

The final model achieved an **F1 score of around 0.95**, which is a slight improvement over the baseline. The biggest improvement came from the model’s ability to correctly identify **bad** ratings. In the baseline model, almost every prediction was labeled as good, resulting in very few true negatives. In contrast, the final model increased true negatives from around 50 to over **2,400**, while still maintaining strong accuracy on good ratings.

<iframe
  src="finalmodelconfusionmatrix.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> 


This shows that including text and user behavior features helped the model become much more balanced and reliable when predicting lower ratings. Overall, the final model performed noticeably better than the baseline and provides meaningful improvements in precision  for the minority class. 

I also want to add that this was not that complex of a predictive task as there was major class imbalance as over 80 percent of the data was actually a 1 so the hardest part of this task was to make it predict a bad rating accurately. 














