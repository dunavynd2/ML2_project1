### Week 2 Exercises ----

### Set up ----

# load libraries
library(tidyverse) # For all things data manipulation
library(GGally) # for data exploration
library(caret) # For confusionMatrix(), training ML models, and more
library(class) # For knn()
library(pROC)  # For ROC curve and estimating the area under the ROC curve

# set a random seed for reproducibility
set.seed(703)

### Load Data ----
#The dataset has 12 columns, 11 predictors and the last column as target: "quality"
redwine <- read_csv("data-raw/redwine.csv")

# check data structure
str(redwine)

#quality of wine is available in range of 3-8
summary(redwine)

### Step 1: Create a train/test split ----
test_idx <- createDataPartition(
  y = redwine$quality,
  p = 0.3,
  list = FALSE
)

test_data <- redwine[test_idx, ]

train_data <- redwine[-test_idx, ]

### Step 2: Data Exploration ----

train_data |>
  ggpairs(aes(color = factor(quality), alpha = 0.5))

### Step 3: Data pre-processing ----

# visualize the quality variable
train_data |>
  ggplot(aes(x = quality)) + 
  geom_bar()

# As seen, the target variable: The dataset is imbalanced on wine quality
# This will get our model trained mainly on majority (5 and 6) quality observations. 

# Convert quality variable on your data to two categories: 
# "low" if wine quality is <= 5 otherwise, "high"
train_data <- 
  train_data |>
  mutate(
    quality2 = ifelse(quality<= 5, "low", "high")
  )

test_data <- 
  test_data |>
  mutate(
    quality2 = ifelse(quality<= 5, "low", "high")
  )


### Step 4: Feature Engineering ----

# standardize the variables in the training set
standardizer <- preProcess(
  train_data |>
    select(
      where(function(x) ! is.integer(x) & ! is.factor(x)) # only true numeric, not our dummies
    ),
  method = c("center", "scale")
)

train_data <- predict(standardizer, train_data)

# apply the standardization function to the test set as well
test_data <- predict(standardizer, test_data)


### Step 5: Feature & Model Selection ----

# three SVM models with the linear, radial, and polynomial kernels
# Have caret search over parameters for each model. Where applicable:
#   - C = c(0.01,0.1,1,5,10) # all
#   - sigma = c(0.5,1,2,3) # radial
#   - degree = c(2, 3) # polynomial
#   - scale = c(0.01,0.5,1) # polynomial

tr_control <- trainControl( # store since we will reuse
  method = "cv", number = 10, # 10-fold cross validation
  classProbs = TRUE,  # Enable probability predictions
  summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
)

svm_linear <- train(
  quality2 ~.,
  data = train_data |> select(-quality),
  method = "svmLinear",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_linear)

svm_radial <- train(
  quality2 ~.,
  data = train_data |> select(-quality),
  method = "svmRadial",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10), sigma = c(0.5,1,2,3)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_radial)

svm_radial$bestTune

svm_poly <- train(
  quality2 ~.,
  data = train_data |> select(-quality),
  method = "svmPoly",
  tuneGrid = expand.grid(
    C = c(0.01,0.1,1,5,10), 
    scale = c(0.5,1,2,3),
    degree = c(2,3)
  ),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_poly)

svm_poly$bestTune



### Step 6: Model Validation ----

# Compare the best tune on all three models
# Choose the one with the best CV result of AUC
validation_table <- 
  bind_rows(
    list(
      svm_linear$resample |>
        mutate(
          type = "linear",
          mean_auc = mean(ROC)
        ) |>
        cbind(svm_radial$bestTune),
      svm_radial$resample |>
        mutate(
          type = "radial",
          mean_auc = mean(ROC)
        ) |>
        cbind(svm_radial$bestTune),
      svm_poly$resample |>
        mutate(
          type = "polynomial",
          mean_auc = mean(ROC)
        ) |>
        cbind(svm_poly$bestTune)
    )
  )

validation_table |>
  ggplot(aes(x = ROC)) +
  geom_density(aes(fill = type), alpha = 0.5) + 
  geom_vline(aes(xintercept = mean_auc))+
  facet_wrap(~type)


### Step 7: Predictions and Conclusions ----

# get probabilistic predictions on your test set on your chosen model
preds <- predict(svm_radial, test_data, type = "prob")

# plot ROC and calculate AUC
roc_radial <- roc(
  test_data$quality2,
  preds$high
)

plot(roc_radial)

roc_radial$auc

# choose a threshold to create category classifications
# calculate precision and recall.
# can you choose a threshold that gives you better precision/recall than p=0.5?

# pick threshold with highest average of sensitivity and specificity
thresh <- roc_radial$thresholds[which.max(roc_radial$sensitivities + roc_radial$specificities)]

confusion <- confusionMatrix(
  test_data$quality2 |> factor(),
  ifelse(preds$high >= thresh, "high", "low") |> factor()
)

confusion$byClass[c("Precision", "Recall")]
