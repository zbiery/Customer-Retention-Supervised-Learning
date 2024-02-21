library(tidyverse)    #Data cleaning, manipulation, visualization
library(tidymodels)   #Creating machine learning models
library(vip)          #Constructing variable importance plots
library(ranger)       #Creating Random Forest models

#Grab file path to data file
path <- here::here("data", "customer_retention.csv")

#Import data file
retention <- readr::read_csv(path)

#Recode response variable as a factor
retention <- mutate(retention, Status = as.factor(Status))

#Remove null values
retention <- na.omit(retention)

#For reproducibility
set.seed(123)

#70-30 train-test split
split <- initial_split(retention, prop = 0.7, strata = "Status")
train <- training(split)
test <- testing(split)

recipe <- recipe(Status ~ ., data = train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

rf_kfold <- vfold_cv(train, v = 5)

# create random forest object
rf_mod <- rand_forest(mode = "classification") %>%
  set_engine("ranger")

# train model
rf_results <- fit_resamples(rf_mod, recipe, rf_kfold)

# model results
collect_metrics(rf_results)

# create random forest model object with tuning option
rf_hyper <- rand_forest(
  mode = "classification",
  trees = tune(),
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity")

# create the hyperparameter grid
rf_hyper_grid <- grid_regular(
  trees(range = c(50, 800)),
  mtry(range = c(2, 30)),
  min_n(range = c(1, 20)),
  levels = 5
)

# train our model across the hyper parameter grid
set.seed(123)
rf_hyper_results <- tune_grid(rf_hyper, recipe, resamples = rf_kfold, grid = rf_hyper_grid)

# model results
show_best(rf_hyper_results, metric = "roc_auc")

# get optimal hyperparameters
rf_best_hyperparameters <- select_best(rf_hyper_results, metric = "roc_auc")

# create final workflow object
final_rf_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_hyper) %>%
  finalize_workflow(rf_best_hyperparameters)

# fit final workflow object
rf_final_fit <- final_rf_wf %>%
  fit(data = train)

# confusion matrix
rf_final_fit %>%
  predict(test) %>%
  bind_cols(test %>% select(Status)) %>%
  conf_mat(truth = Status, estimate = .pred_class)

rf_final_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 10)