Authors: Madhav Pudukottai Krishnaraj, Yaqub Mir
# Overview
We noticed that it was very rare to come by a rating that was not a 5, and wanted to dive deeper into what really prompts a user to give a rating less than a 5, how new a user is to the app, the recipe was bad, or maybe it was just the user's preference. 
# Introduction
Recipe ratings on Food.com are overwhelmingly positive (over 70 percent of ratings are 5s), with most users giving 5-star reviews. This makes it difficult to understand what actually leads people to rate a recipe lower. In this project, we analyze thousands of recipes and interactions to uncover patterns behind ratings below 5, study how user behavior affects scoring, and build predictive models to classify when a rating will be “good” or “bad.”

The two datasets we use in this project are derived from [food.com](https://www.food.com/) and contain information on recipes and interactions with recipes from 2008 - 2018. The original purpose of the datasets is for the recommender system research paper, [Generating Personalized Recipes from Historical User Preferences](https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19c.pdf) by Majumder, Li, Ni, and McAuley.

We used the `recipe` and `interactions` datasets to complete this project. Here's a look at both of them.

`recipe`
### Recipes Dataset Columns

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
| Column        | Description                                                                                  |
| :------------ | :------------------------------------------------------------------------------------------- |
| `user_id`     | ID of the user who submitted the rating or review                                           |
| `recipe_id`   | ID of the recipe being reviewed                                                             |
| `date`        | Date the interaction (rating/review) was submitted                                          |
| `rating`      | User rating for the recipe (0–5, where 0 indicates a missing rating)                        |
| `review`      | Optional user-written text review                                                           |

(731927 rows, 5 columns)



