### Week 2 Exercises ----

### Set up ----

# load libraries
library(tidyverse) # For all things data manipulation
library(GGally) # for data exploration
library(caret) # For confusionMatrix(), training ML models, and more
library(class) # For knn()
library(dplyr) 
library(pROC)  # For ROC curve and estimating the area under the ROC curve

# set a random seed for reproducibility
set.seed(703)

### Load Data ----
#The dataset has 12 columns, 11 predictors and the last column as target: "quality"
employee <- read_csv("EmployeeData.csv")

# check data structure
str(employee)

#quality of wine is available in range of 3-8
summary(employee)

### Step 1: Create a train/test split ----
test_idx <- createDataPartition(
  y = employee$Attrition,
  p = 0.3,
  list = FALSE
)

test_data <- employee[test_idx, ]
train_data <- employee[-test_idx, ]
### Step 2: Data Exploration ----

train_data |>
  ggpairs(aes(color = factor(Attrition), alpha = 0.5))

### Step 3: Data pre-processing ----

# visualize the quality variable
train_data |>
  ggplot(aes(x = Attrition)) + 
  geom_bar()

# As seen, the target variable: The dataset is imbalanced on wine quality
# This will get our model trained mainly on majority (5 and 6) quality observations.

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


# Identify near-zero variance predictors
nzv <- nearZeroVar(train_clean, saveMetrics = TRUE)
print(nzv)
# Remove constant variables
constant_vars <- rownames(nzv[nzv$zeroVar == TRUE, ])
print(constant_vars)
# Remove constant variables from the training set
train_clean <- train_clean[, !(colnames(train_clean) %in% constant_vars)]
# Remove constant variables from the test set
test_clean <- test_clean[, !(colnames(test_clean) %in% constant_vars)]

# Check for missing values in the training set
colSums(is.na(train_clean))
# Check for missing values in the test set
colSums(is.na(test_clean))
# Option 1: Remove rows with missing values
train_clean <- na.omit(train_clean)
test_clean <- na.omit(test_clean)

head(train_clean)
head(test_clean)
dim(train_clean)
dim(test_clean)
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
  Attrition ~.,
  data = train_clean,
  method = "svmLinear",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_linear)

svm_radial <- train(
  Attrition ~.,
  data = train_clean,
  method = "svmRadial",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10), sigma = c(0.5,1,2,3)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_radial)

svm_radial$bestTune

svm_poly <- train(
  Attrition ~.,
  data = train_clean,
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
preds <- predict(svm_radial, test_clean, type = "prob")

# plot ROC and calculate AUC
roc_radial <- roc(
  test_clean$Attrition,
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
  test_clean$Attrition |> factor(),
  ifelse(preds$high >= thresh, "high", "low") |> factor()
)

confusion$byClass[c("Precision", "Recall")]
