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

#Logistic Regression

lr_kfold <- vfold_cv(train, v = 5, strata = Status)

lr_results <- logistic_reg() %>%
  fit_resamples(Status ~ ., lr_kfold)

#mean AUC
collect_metrics(lr_results) %>% filter(.metric == "roc_auc")

#AUC across all folds
collect_metrics(lr_results, summarize = FALSE) %>% 
  filter(.metric == "roc_auc")

#LR model fitting
lr_final <- logistic_reg() %>%
  fit(Status ~ ., data = train)

lr_final %>%
  predict(test) %>%
  bind_cols(test %>% select(Status)) %>%
  conf_mat(truth = Status, estimate = .pred_class)

#LR variable importance plotting
vip(lr_final$fit, num_features = 10)

#LR tuning
lr_hyper <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

lr_grid <- grid_regular(mixture(), penalty(), levels = 10)

lr_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(lr_hyper)

lr_tuning_results <- lr_wf %>%
  tune_grid(resamples = lr_kfold, grid = lr_grid)

lr_tuning_results %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(mean)) %>%
  print(n = 5)

#autoplot(lr_tuning_results)

#LR model w/ tuning
lr_best_hyper <- select_best(lr_tuning_results, metric = "roc_auc")

lr_final_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(lr_hyper) %>%
  finalize_workflow(lr_best_hyper)

lr_hyper_final <- lr_final_wf %>%
  fit(data = train)

lr_hyper_final %>%
  predict(test) %>%
  bind_cols(test %>% select(Status)) %>%
  conf_mat(truth = Status, estimate = .pred_class)
