library(tidyverse)    #Data cleaning, manipulation, visualization
library(tidymodels)   #Creating machine learning models
library(vip)          #Constructing variable importance plots
library(baguette)     #Creating bagged tree models

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

# create resampling procedure
bag_kfold <- vfold_cv(train, v = 5)

# create bagged CART model object with 5 bagged trees
bag_mod <- bag_tree() %>%
  set_engine("rpart", times = 5) %>%
  set_mode("classification")

# train model
bag_results <- fit_resamples(bag_mod, recipe, bag_kfold)

# model results
collect_metrics(bag_results)

bag_hyper <- bag_tree() %>%
  set_engine("rpart", times = tune()) %>%
  set_mode("classification")

# create the hyperparameter grid
bag_hyper_grid <- expand.grid(times = c(5, 25, 50, 100, 200, 300))

# train our model across the hyper parameter grid
set.seed(123)
bag_hyper_results <- tune_grid(bag_hyper, recipe, resamples = bag_kfold, grid = bag_hyper_grid)

# model results
show_best(bag_hyper_results, metric = "roc_auc", n = 5)

# identify best model
bag_best_hyperparameters <- bag_hyper_results %>%
  select_best("roc_auc")

# finalize workflow object
bag_final_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(bag_hyper) %>%
  finalize_workflow(bag_best_hyperparameters)

# final fit on training data
bag_final_fit <- bag_final_wf %>%
  fit(data = train)

# confusion matrix
bag_final_fit %>%
  predict(test) %>%
  bind_cols(test %>% select(Status)) %>%
  conf_mat(truth = Status, estimate = .pred_class)

bag_vip <- bag_final_fit %>%
  extract_fit_parsnip() %>%
  .[['fit']] %>%
  var_imp() %>%
  slice(1:10)

ggplot(bag_vip, aes(value, reorder(term, value))) +
  geom_col() +
  ylab(NULL) +
  xlab("Feature importance")