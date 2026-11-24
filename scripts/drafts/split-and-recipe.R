library(tidymodels)
library(tidyverse)
library(kknn)

#1 Loading processed Spotify sample and converting outcome to a factor
spotify <- read_csv("data/processed/spotify_sample.csv") |>
  mutate(playlist_genre = as.factor(playlist_genre))

#2 Performing an 80/20 stratified train/test split
set.seed(169)
spotify_split <- initial_split(spotify, prop = 0.80, strata = playlist_genre)
spotify_train <- training(spotify_split)
spotify_test  <- testing(spotify_split)

#Verifying class balance across training (360/genre) and testing (90/genre) sets (verified)
spotify_train |> count(playlist_genre)
spotify_test  |> count(playlist_genre)

#3 Building preprocessing recipe:
#    - Assigning "ID" roles to metadata so KNN ignores them in distance calculations (We might want to reference these later)
#    - Standardizing all 9 numeric predictors (mean = 0, sd = 1) using training stats
knn_rec <- recipe(playlist_genre ~ ., data = spotify_train) |>
  update_role(track_id, track_name, track_artist, new_role = "ID") |>
  step_normalize(all_numeric_predictors())

knn_rec

#4 Creating 5-fold stratified cross-validation resamples on the training data
set.seed(169)
spotify_folds <- vfold_cv(spotify_train, v = 5, strata = playlist_genre)

spotify_folds

# ==============================================================================
# NOTES (Source this script at the top of your modeling file):
#   source("scripts/drafts/split-and-recipe.R")
#
#   - Use "knn_rec" in your workflow so all models use 
#     the same normalized predictors and ID exclusions.
#   - Use `spotify_folds` with `fit_resamples()` to evaluate k = 3, 5, 10
#     (for both "inv" and "rectangular" weight_func) on the same CV folds.
#   - Keep "spotify_test" untouched until the final evaluation step
# ==============================================================================