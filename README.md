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

1. Left Merge our recipes and interactions datasets on id and recipe_id (this helped us make sure that every review in the interactions dataset is a review to a recipe that we have data for)

2. We filled all ratings that were 0 with np.nan as they were missing

3. Since we knew from the dsc80 website that our nutritional information was in a list in the form of a string, we needed to make it a list type.
   - Convert the values in the list to a float so we can actually use the nutritional information

5. Convert the `submitted` column to dateime so it is easier and more versatile to use

6. convert `user_id` and `recipe_id` back to int types because when we merged it turned to floats and looked really messy

7. Make the `user_review_counts` feature by grouping by user_id and taking the size of the dataframe, then merging it back into the combined dataframe on user_id (this was a clean step because our dataframe had 234429 rows before and after the merge)

8. Look at the correlation between all relevant columns and rating, the largest coefficient is user_id, although weak, but it is negative. Even though user id is a categorical value, depending on if newer users have a larger user id in terms of numerical value, it could mean that newer users rate lower on average

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

Img

# Bivariate Analysis

For our bivariate analysis, we examined how average recipe ratings differ across three reviewer experience groups. We created these groups using pd.qcut on the user_review_count column, which evenly splits users into low, medium, and high activity categories based on how many reviews they have previously written.

When comparing the mean ratings across these groups higher-experience users had the highest average ratings, while lower-experience users had the lowest. Meaning more active reviewers tend to rate recipes more higher on average, whereas users with little reviewing history are more likely to give lower scores.

IMG

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


IMG 

**This graph shows us the distribution of different groups, the majority of points are clustered at 5**

# Missingness Analysis

# NMAR Analysis 












