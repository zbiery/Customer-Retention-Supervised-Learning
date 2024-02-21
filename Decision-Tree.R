library(tidyverse)    #Data cleaning, manipulation, visualization
library(tidymodels)   #Creating machine learning models
library(vip)          #Constructing variable importance plots

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

dt <- decision_tree(mode = "classification") %>%
  set_engine("rpart")

# Step 3: fit model workflow
dt_fit <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(dt) %>%
  fit(data = train)

# create resampling procedure
set.seed(123)
dt_kfold <- vfold_cv(train, v = 5)

# train model
dt_results <- fit_resamples(dt, recipe, dt_kfold)

# model results
collect_metrics(dt_results)

#Fit curve
rpart.plot::rpart.plot(dt_fit$fit$fit$fit)

#hyperparamater tuning
dt_hyper <- decision_tree(
  mode = "classification",
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart")

# create the hyperparameter grid
dt_hyper_grid <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

# train our model across the hyper parameter grid
set.seed(123)
dt_hyper_results <- tune_grid(dt_hyper, 
                              recipe, 
                              resamples = dt_kfold, 
                              grid = dt_hyper_grid)

# get best results
show_best(dt_hyper_results, metric = "roc_auc", n = 5)

# get best hyperparameter values
dt_best_model <- select_best(dt_hyper_results, metric = 'roc_auc')

# put together final workflow
dt_final_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(dt_hyper) %>%
  finalize_workflow(dt_best_model)

# fit final workflow across entire training data
dt_final_fit <- dt_final_wf %>%
  fit(data = train)

rpart.plot::rpart.plot(dt_final_fit$fit$fit$fit)

# confusion matrix
dt_final_fit %>%
  predict(test) %>%
  bind_cols(test %>% select(Status)) %>%
  conf_mat(truth = Status, estimate = .pred_class)

# plot feature importance
dt_final_fit %>%
  extract_fit_parsnip() %>%
  vip(10)