### Week 1 Exercises ----

### Set up ----

# load libraries
library(tidyverse) # For all things data manipulation
library(GGally) # for data exploration
library(caret) # For confusionMatrix(), training ML models, and more
library(class) # For knn()
library(dplyr) # For some data manipulation and ggplot
library(pROC)  # For ROC curve and estimating the area under the ROC curve

# set a random seed for reproducibility
set.seed(703)

### Load Data ----
#The dataset has 12 columns, 11 predictors and the last column as target: "quality"
redwine <- read_csv("RedWine.csv")

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

# Convert quality variable on your training set to two categories: 
# "low" if wine quality is <= 5 otherwise, "high"
train_data <- 
  train_data |>
  mutate(
    quality2 = ifelse(quality<= 5, "low", "high")
  )

### Step 4: Feature Engineering ----

# standardize the variables in the training set
standardizer <- # create a procedure to standardize data (store means and sds)
  train_data |>
  select(
    where(is.numeric),
    -quality
  ) |>
  preProcess(
    method = c("center", "scale")
  )

train_std <- predict(standardizer, train_data)

# apply the standardization function to the test set as well

test_std <- predict(standardizer, test_data)

### Step 5: Feature & Model Selection ----

# build a KNN model using caret::train()
# use 10 fold cross validation
# search for the optimal "k" from 3 to 10
f_knn <- train(
  quality2 ~ .,
  data = train_std |> 
    select(-quality),
  method = "knn",
  tuneGrid = expand.grid(k = seq(3, 10)),
  trControl = trainControl(
    method = "cv", number = 10, # 10-fold cross validation
    classProbs = TRUE,  # Enable probability predictions
    summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
  ),
  metric = "ROC" # "ROC" gives us AUC(classification) & silences warning about Accuracy
  
)

plot(f_knn)

### Step 6: Model Validation ----

# no need as we did this during training

### Step 7: Predictions and Conclusions ----

# get probabilistic predictions on your test set
preds <- predict(f_knn, test_std, type = "prob")

# plot ROC and calculate AUC
test_std <- 
  test_std |>
  mutate(
    quality2 = ifelse(quality <= 5, "low", "high")
  )

roc_knn <- 
  roc(
    test_std$quality2,
    preds[,"high"]
  )

plot(roc_knn)

roc_knn$auc


# choose a threshold to create category classifications
# calculate precision and recall.
# can you choose a threshold that gives you better precision/recall than p=0.5?

confusion_baseline <- 
  confusionMatrix(
    data = test_std$quality2 |> factor(),
    ifelse(preds$high >= 0.5, "high", "low") |> factor()
  )

confusion_baseline$byClass[c("Precision", "Recall")]

# pick a different threshold
plot(roc_knn$thresholds, roc_knn$sensitivities * roc_knn$specificities, type = "o")

confusion_adjusted <- 
  confusionMatrix(
    data = test_std$quality2 |> factor(),
    ifelse(preds$high >= 0.6, "high", "low") |> factor()
  )

confusion_adjusted$byClass[c("Precision", "Recall")]
