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

# Convert quality variable on your training set to two categories: 
# "low" if wine quality is <= 5 otherwise, "high"


### Step 4: Feature Engineering ----

# standardize the variables in the training set

# apply the standardization function to the test set as well

### Step 5: Feature & Model Selection ----

# build a KNN model using caret::train()
# use 10 fold cross validation
# search for the optimal "k" from 3 to 10

### Step 6: Model Validation ----

# no need as we did this during training

### Step 7: Predictions and Conclusions ----

# get probabilistic predictions on your test set

# plot ROC and calculate AUC

# choose a threshold to create category classifications
# calculate precision and recall.
# can you choose a threshold that gives you better precision/recall than p=0.5?


