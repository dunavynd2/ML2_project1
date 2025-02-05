### Week 3 Example: neuralnet package ----

### Set up ----

# load libraries
#(Install and) load the following libraries
library(tidyverse)
library(GGally) # for data exploration
library(caret) #For confusionMatrix(), training ML models, and more
library(neuralnet) #For neuralnet() function
library(dplyr) #For some data manipulation and ggplot
library(pROC)  #For ROC curve and estimating the area under the ROC curve
library(fastDummies) #To create dummy variable (one hot encoding)

# set a random seed for reproducibility
set.seed(7503)

### Load Data ----

titanic <-  read_csv("data-raw/titanic.csv")

# check data structure
str(titanic)

summary(titanic)

# any obvious pre-formatting, do here (e.g., numeric to categorical etc.)

titanic <- 
  titanic |>
  mutate(
    Survived = factor(Survived) |>
      fct_recode("No" = "0", "Yes" = "1"),
    Sex = factor(Sex)
  )

str(titanic)

#Create dummy variables for each level of the categorical variables of interest,
# and remove the columns used to generate the dummy columns, and first dummy of each variable
final_data <- 
  titanic |>
  dummy_cols(
    select_columns = c('Sex','Pclass'),
    remove_selected_columns = T,
    remove_first_dummy = T
  ) |>
  select( # remove variables we know we won't use
    -PassengerId,
    -Name,
    -Ticket,
    -Cabin,
    -Embarked
  )

str(final_data)

### Step 1: Create a train/test split ----
test_idx <- createDataPartition(
  final_data$Survived,
  p = 0.3,
  list = FALSE
)

train_data <- final_data[-test_idx, ]

test_data <- final_data[test_idx, ]


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

#As for the missing 'age' data, we have three options:
# 1. use a good estimate or create a model to predict missing 'age'
# 2. delete the observations that contain a 'NA' entry (small number of observations)
# 3. remove age as a predictor (age is highly correlated with survival so it's not recommended)

# There is 129 missing value in age column, around 20% of our data. 
# Instead of removing the data, let's try to replace missing ones with the mean of age.

# NOTE: We always split the data before any imputation or scaling.
# Then, since we're pretending 'test' (i.e., validation) to the be an unseen dataset, 
# we will use information from train data to impute the test dataset!

train_data$Age[is.na(train_data$Age)] <- mean(train_data$Age, na.rm = TRUE)

test_data$Age[is.na(test_data$Age)] <- mean(train_data$Age, na.rm = TRUE)

### Step 4: Feature Engineering ----
#Neural networks work best when the predictors are scaled.
#The most common scaling for neuralnets are min-max normalization (values will be between 0 and 1).
#Since the neuralnet() function doesn't have an option to automatically do that, 
# we will scale the predictors ourselves. 
#The first column (target variable: Survived) doesn't need to be scaled
#Remember that this variable is categorical by nature. Survived: Yes or No!

#We are using preProcess function from "caret" package, using "range" (min-max normalization) method
#Again, we are using train information to scale test data!
#NOTE: Predictors that are not numeric are ignored in the calculations of preProcess function

normalizer <- preProcess(
  train_data |>
    select(
      where(function(x) ! is.integer(x) & ! is.factor(x)) # only true numeric, not our dummies
    ),
  method = "range"
)

train_data <- predict(normalizer, train_data)

test_data <- predict(normalizer, test_data)

### Step 5: Feature & Model Selection ----

# example with neuralnet package: One layer, two hidden units
# If you want to do cross validation to to tune parameters, you'll have to
# set it up manually, like we did at the beginning of ML 1

nn1 <- neuralnet(
  Survived ~.,
  data = train_data,
  linear.output = FALSE, # classification vs regression
  hidden=2
)

plot(nn1) #Plot of the network architecture and its weights

nn1$net.result[[1]] #Overall results: estimated probabilities of belonging to classes 0 and 1

nn1$act.fct #the activation function used in training

nn1$result.matrix #a matrix containing the error, weights, and much more

# check auc etc. (all in-sample, this is to check mechanics)
preds_train1 <- predict(nn1, train_data)

roc_train1 <- roc(
  train_data$Survived,
  preds_train1[, 2] # second column corresponds to "Yes" the second level
)

plot(roc_train1)

roc_train1$auc

# two hidden layers
nn2 <- neuralnet(
  Survived ~.,
  data = train_data,
  linear.output = FALSE, # classification vs regression
  hidden = c(3,2) #two layers, layer 1 with three and layer 2 with two neurons
)


plot(nn2)

preds_train2 <- predict(nn2, train_data)

roc_train2 <- roc(
  train_data$Survived,
  preds_train2[, 2] # second column corresponds to "Yes" the second level
)

plot(roc_train2)

roc_train2$auc

# example with caret 
# neuralnet is available in caret *only* for regression
nn3 <- train(
  Survived ~.,
  data = train_data,
  method = "nnet",
  trControl = trainControl( # store since we will reuse
    method = "cv", number = 10, # 10-fold cross validation
    classProbs = TRUE,  # Enable probability predictions
    summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
  ),
  tuneGrid = expand.grid(
    size = c(2, 3, 5, 10), 
    decay = c(0.0001,0.001,0.01,0.1)
    ),
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(nn3)

preds_train3 <- predict(nn3, train_data, type = "prob")

roc_train3 <- roc(
  train_data$Survived,
  preds_train3$Yes
)

plot(roc_train3)

roc_train3$auc

### Step 6: Model Validation ----
# done with caret, skipping with neuralnet for time

  
### Step 7: Predictions and Conclusions ----
p1 <- predict(nn1, test_data)

p2 <- predict(nn2, test_data)

p3 <- predict(nn3, test_data, type = "prob") # caret has different options

roc1 <- roc(
  test_data$Survived,
  p1[, 2]
)

plot(roc1)

roc2 <- roc(
  test_data$Survived,
  p2[, 2]
)

plot(roc2)


roc3 <- roc(
  test_data$Survived,
  p3[, 2]
)

plot(roc3)

auc_compare <- tibble(
  auc = c(roc1$auc, roc2$auc, roc3$auc),
  model = c("neuralnet 1", "neuralnet 2", "caret nnet")
)

auc_compare |>
  ggplot(aes(x = auc, y = model, fill = model)) + 
  geom_bar(stat = "identity") + 
  theme(legend.position = "none") + 
  labs(title = "Comparison of Test Results")

auc_compare

