### Week 1 K-Nearest Neighbor Classifier ----

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
titanic <-  read_csv("data-raw/titanic.csv")

# check data structure
str(titanic)

summary(titanic)

# any obvious pre-formatting, do here (e.g., numeric to categorical etc.)

#Let's remove variables that we will not use for distance calculation
#due to their data type being categorical (such as sex) or not being relevant (such as Id)
final_data <- 
  titanic |> 
  select(
    -PassengerId, 
    -Name, 
    -Sex, 
    -Ticket, 
    -Cabin, 
    -Embarked
  ) |>
  mutate(
    Survived = factor(Survived) |>
      fct_recode("No" = "0", "Yes" = "1")
  )


### Step 1: Create a train/test split ----

test_idx <- createDataPartition(
  final_data$Survived,
  p = 0.3,
  list = FALSE
)

train_data <- final_data[-test_idx, ]

test_data <- final_data[test_idx, ]

#Check the ratio of 0's and 1's in the target variable is the same between sets
# Don't need to balance classes with KNN
prop.table(table(train_data$Survived))
prop.table(table(test_data$Survived))

### Step 2: Data Exploration ----

train_data |>
  ggpairs(aes(color = Survived, alpha = 0.5))

### Step 3: Data pre-processing ----
#Check for columns/variables with missing values
train_data |>
  is.na() |>
  colSums()

test_data |>
  is.na() |>
  colSums()


#what fraction is missing?
sum(is.na(train_data$Age)) / nrow(train_data)

sum(is.na(test_data$Age)) / nrow(test_data)


#As for the missing 'age' data, we have three options:
# 1. use a good estimate or create a model to predict missing 'age'
# 2. delete the observations that contain a 'NA' entry (small number of observations)
# 3. remove age as a predictor (age is highly correlated with survival so it's not recommended)

#There is 129 missing value in age column, around 20% of our data. 
#Instead of removing the data, let's try to replace missing ones with the mean of age.

# NOTE: We always split the data before any imputation or scaling.
# Then, since we're pretending 'test' (i.e., validation) to the be an unseen dataset, 
# we will use information from train data to impute the test dataset!

train_data$Age[is.na(train_data$Age)] <- mean(train_data$Age, na.rm = TRUE)

test_data$Age[is.na(test_data$Age)] <- mean(train_data$Age, na.rm = TRUE)


### Step 4: Feature Engineering ----
#Next we should scale predictors because distance functions are sensitive to scale
#The first column (target variable Survived) doesn't need to be scaled
#Remember that this variable is categorical by nature. Survived: Yes or No!

#Again, we use information from train set to scale test dataset!

# Note caret is smart. It only scales numeric variables

standardizer <- preProcess(
  train_data,
  method = c("center", "scale")
)

train_data <- predict(standardizer, train_data)

test_data <- predict(standardizer, test_data)

### Step 5: Feature & Model Selection ----

### K-Nearest Neighbors ###
# To start, let's use a rule on thumb to set a value for k
k <- train_data |>
  nrow() |>
  sqrt() |>
  round()

k

# so we'll use 25 for the k. 
#We use method knn() from the package 'class' to run the model
help(knn)

# For KNN, you need not do the prediction: prediction is automatically indicated in modeling
# However, scaling is not. That's why we scaled the data in lines 75 to 89 

# We need a random seed before we apply knn() because if several observations are 
# tied as nearest neighbors, then R will randomly break the tie. 
# Therefore, a seed must be set in order to ensure reproducibility of results.
knn_classifier <- train(
  Survived ~ .,
  data = train_data,
  method = "knn",
  tuneGrid = expand.grid(k = seq(2, 40)),
  trControl = trainControl(
    method = "cv", number = 10, # 10-fold cross validation
    classProbs = TRUE,  # Enable probability predictions
    summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
  ),
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(knn_classifier)

knn_classifier$bestTune

knn_classifier$results

knn_classifier$resample

### Step 6: Model Validation ----
# in this case, validation was completed during training

### Step 7: Predictions and Conclusions ----

# ROC and AUC
roc_knn <- 
  roc(
    test_data$Survived,
    predict(knn_classifier, test_data, type = "prob")[["Yes"]]
  )

plot(roc_knn)

roc_knn$auc

# Confusion matrix statistics
confusion_stats <- 
  confusionMatrix(
    data = test_data$Survived, 
    reference = predict(knn_classifier, test_data, type = "raw"), 
    positive = "Yes"
  )

confusion_stats$table

confusion_stats$byClass[c("Precision", "Recall")]
