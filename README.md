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













