### Week 3 Exercises ----

### Set up ----

# load libraries
library(tidyverse)
library(GGally) # for data exploration
library(caret) #For confusionMatrix(), training ML models, and more
library(neuralnet) #For neuralnet() function
library(dplyr) #For some data manipulation and ggplot
library(fastDummies) #To create dummy variable (one hot encoding)
library(sigmoid) #For the relu activation function

# set a random seed for reproducibility

### Load Data ----
toyota <- read_csv("data-raw/Toyota.csv")

# check data structure
str(toyota)

# any obvious pre-formatting, do here (e.g., numeric to categorical etc.)
#Create dummy variables for each level of the categorical variables of interest
#Removing categorical variables and one dummy level of each categorical variables
final_data <-
  toyota |>
  dummy_cols(
  select_columns = c('Fuel_Type','Color'),
  remove_selected_columns = T,
  remove_first_dummy = T
)

### Step 1: Create a train/test split ----
test_idx <- createDataPartition(
  final_data$Price,
  p = 0.3,
  list = FALSE
)

train_data <- final_data[-test_idx, ]

test_data <- final_data[test_idx, ]

### Step 2: Data Exploration ----
train_data |>
  ggpairs()

### Step 3: Data pre-processing ----

#Check for columns/variables with missing values
train_data |>
  is.na() |>
  colSums()

test_data |>
  is.na() |>
  colSums()

### Step 4: Feature Engineering ----

# normalize the data. Do not normalize price.

### Step 5: Feature & Model Selection ----

# Train a neural net with one hidden layer and 5 units using the neuralnet package
# Use a ReLu activation function: set argument   act.fct = relu
# if you don't converge change the stepmax parameter to a larger value or change the learning rate

# Train a second neural net using caret with the 'nnet' method.
# use 10-fold cross validation and a grid search:
# size from 1 to 10 and decay between 0.01 and 0.2
# What's the best size and decay?

### Step 6: Model Validation ----
# skipping for brevity

### Step 7: Predictions and Conclusions ----

# get predictions from both models

# if you normalized price, convert predictions back to dollars

# calculate r-squared using either mvrsquared::calc_rsquared or caret::R2

# calculate RMSE and MAE (check caret for these functions)

# plot the original price versus the predicted price for both models

# which model do you think works better?
