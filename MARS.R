library(tidyverse)    #Data cleaning, manipulation, visualization
library(tidymodels)   #Creating machine learning models
library(vip)          #Constructing variable importance plots
library(earth)        #Creating MARS models

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


#MARS model
# create MARS model object
mars <- mars(mode = "classification", num_terms = tune(), prod_degree = tune())

#create resampling procedure
set.seed(123)
mars_kfold <- vfold_cv(train, v = 5)

# create a hyper parameter tuning grid
mars_grid <- grid_regular(
  num_terms(range = c(1, 100)), 
  prod_degree(),
  levels = 25
)

# train our model across the hyper parameter grid
mars_results <- tune_grid(mars, recipe, resamples = mars_kfold, grid = mars_grid)

# get best results
show_best(mars_results, metric = "roc_auc")

mars_best_hyperparameters <- select_best(mars_results, metric = "roc_auc")

mars_final_wf <- workflow() %>%
  add_model(mars) %>%
  add_recipe(recipe) %>%
  finalize_workflow(mars_best_hyperparameters)

mars_final_fit <- mars_final_wf %>%
  fit(data = train)

mars_final_fit %>%
  predict(test) %>%
  bind_cols(test %>% select(Status)) %>%
  conf_mat(truth = Status, estimate = .pred_class)

mars_final_wf %>%
   fit(data = train) %>% 
   extract_fit_parsnip() %>%
   vip(10, type = "rss")

mars_final_fit %>% 
  predict(test, type = "prob") %>%
  mutate(truth = test$Status) %>%
  roc_auc(truth, .pred_Current)
