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
#The dataset has 12 columns, 11 predictors and the last column as target: "Attrition_Num"
employee <- read_csv("EmployeeData.csv")

# check data structure
str(employee)
summary(employee)
employee_df <- employee %>%
  mutate(Attrition_Num = ifelse(Attrition == "Yes", 1, 0))

### Step 1: Create a train/test split ----
test_idx <- createDataPartition(
  y = employee_df$Attrition,
  p = 0.3,
  list = FALSE
)

test_data <- employee_df[test_idx, ]

train_data <- employee_df[-test_idx, ]

### Step 2: Data Exploration ----

train_data |>
  ggpairs(aes(color = factor(Attrition), alpha = 0.5))

### Step 3: Data pre-processing ----

# visualize the Attrition_Num variable
train_data |>
  ggplot(aes(x = Attrition)) + 
  geom_bar()

# As seen, the target variable: The dataset is imbalanced on wine Attrition_Num
# This will get our model trained mainly on majority (5 and 6) Attrition_Num observations.

### Step 4: Feature Engineering ----

# standardize the variables in the training set
standardizer <- # create a procedure to standardize data (store means and sds)
  train_data |>
  select(
    where(is.numeric),
    -StandardHours
  ) |>
  preProcess(
    method = c("center", "scale")
  )

train_std <- predict(standardizer, train_data)

# apply the standardization function to the test set as well
test_std <- predict(standardizer, test_data)

# Check for NAs in the training data
sum(is.na(train_std))
# Check for NAs in the test data
sum(is.na(test_std))
train_clean <- na.omit(train_std)
test_clean <- na.omit(test_std)
sum(is.na(train_clean))
sum(is.na(test_clean))

### Step 5: Feature & Model Selection ----
# build a KNN model using caret::train()
# use 10 fold cross validation
# search for the optimal "k" from 3 to 10
# Train the k-NN model
f_knn <- train(
  Attrition ~ .,  # Predict Attrition using all other variables
  data = train_clean,   # Use the standardized training data
  method = "knn",     # Use k-Nearest Neighbors
  tuneGrid = expand.grid(k = seq(3, 10)),  # Tune k from 3 to 10
  trControl = trainControl(
    method = "cv", number = 10,  # 10-fold cross-validation
    classProbs = TRUE,           # Enable class probabilities
    summaryFunction = twoClassSummary  # Use twoClassSummary for AUC
  ),
  metric = "ROC"  # Optimize for AUC (ROC)
)

### Step 6: Model Validation ----

# no need as we did this during training

### Step 7: Predictions and Conclusions ----

# get probabilistic predictions on your test set
# model is trained
preds <- predict(f_knn, test_clean, type = "prob")

roc_knn <- 
  roc(
    test_clean$Attrition,
    preds[,"Yes"] 
  )
# control corresponds to the reference level (No)
# case corresponds to the positive class("Yes")

plot(roc_knn, col = "blue")

roc_knn$auc


# choose a threshold to create category classifications
# calculate precision and recall.
# can you choose a threshold that gives you better precision/recall than p=0.5?

confusion_baseline <- 
  confusionMatrix(
    data = test_clean$Attrition |> factor(),
    ifelse(preds$Yes >= 0.5, "Yes", "No") |> factor()
  )

confusion_baseline$byClass[c("Precision", "Recall")]

# pick a different threshold
plot(roc_knn$thresholds, roc_knn$sensitivities * roc_knn$specificities, type = "o")

confusion_adjusted <- 
  confusionMatrix(
    data = test_clean$Attrition |> factor(),
    ifelse(preds$Yes <= 0.1, "Yes", "No") |> factor()
  )

confusion_adjusted$byClass[c("Precision", "Recall")]
