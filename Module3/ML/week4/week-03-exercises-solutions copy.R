# load libraries
install.packages("tidyverse")
install.packages("GGally")
install.packages("caret")
install.packages("neuralnet")
install.packages("dplyr")
install.packages("fastDummies")
install.packages("sigmoid")

library(tidyverse)
library(GGally) # for data exploration
library(caret) #For confusionMatrix(), training ML models, and more
library(neuralnet) #For neuralnet() function
library(dplyr) #For some data manipulation and ggplot
library(fastDummies) #To create dummy variable (one hot encoding)
library(sigmoid) #For the relu activation function

# set a random seed for reproducibility

### Load Data ----
airbnb <- read_csv("AirbnbListings.csv")

# check data structure
str(airbnb)

# any obvious pre-formatting, do here (e.g., numeric to categorical etc.)
#Create dummy variables for each level of the categorical variables of interest
#Removing categorical variables and one dummy level of each categorical variables
final_data <-
  airbnb |>
  dummy_cols(
    select_columns = c('room_type','neighborhood','bathrooms','superhost'),
    remove_selected_columns = T,
    remove_first_dummy = T
  )
final_data$host_since <- as.Date(final_data$host_since, format = "%m/%d/%Y")
final_data$year <- as.numeric(format(final_data$host_since, '%Y'))
final_data$month <- as.numeric(format(final_data$host_since, '%m'))
final_data$day <- as.numeric(format(final_data$host_since, '%d'))
final_data$day_of_week <- as.numeric(format(final_data$host_since, '%u'))
final_data$day_of_year <- as.numeric(format(final_data$host_since, '%j'))

final_data <- final_data |> select(-host_since)
str(final_data)

final_data <- na.omit(final_data)

### Step 1: Create a train/test split ----
test_idx <- createDataPartition(
  final_data$price,
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
normalizer <- preProcess(
  train_data |>
    select(
      where(function(x) ! is.integer(x) & ! is.factor(x)),
      -price
    ),
  method = "range"
)

train_data <- predict(normalizer, train_data)

test_data <- predict(normalizer, test_data)
str(test_data)
### Step 5: Feature & Model Selection ----

# Train a neural net with one hidden layer and 5 units using the neuralnet package
# Use a ReLu activation function
relu <- function(x) ifelse(x>0,x,0)
# if you don't converge change the stepmax parameter to a larger value or change the learning rate
predictors <- colnames(train_data)[colnames(train_data) != "price"]
# Replace spaces with underscores in your variable names
predictors <- gsub(" ", "_", predictors)
formula <- as.formula(paste("price ~", paste(predictors, collapse = " +")))
colnames(train_data) <- make.names(colnames(train_data))
str(train_data)
nn1 <- neuralnet(
  price ~ .,
  data = train_data,
  hidden = 1,
  threshold = 0.01,
  stepmax = 1e5,
  rep = 1,
  startweights = NULL,
  learningrate.limit = NULL,
  learningrate.factor = list(minus = 0.5, plus = 1.2),
  learningrate = NULL,
  lifesign = "minimal",
  lifesign.step = 1000,
  algorithm = "rprop+",
  err.fct = "sse",
  act.fct = relu, # Use ReLU activation function
  linear.output = TRUE,
  exclude = NULL,
  constant.weights = NULL,
  likelihood = FALSE
)

print(colnames(train_data))
str(train_data)
colSums(train_data)
plot(nn1)

# Train a second neural net using caret with the 'nnet' method.
# use 10-fold cross validation and a grid search:
# size from 1 to 10 and decay between 0.01 and 0.2
# What's the best size and decay?

nn2 <- train(
  Price ~.,
  data = train_data,
  method = "nnet",
  linout = TRUE,
  trControl = trainControl( # store since we will reuse
    method = "cv", number = 10
  ),
  tuneGrid = expand.grid(
    size = 1:10, 
    decay = c(0.01,0.05,0.1,0.15, 0.2)
  ),
  metric = "RMSE"
)

plot(nn2)

nn2$bestTune

### Step 6: Model Validation ----
# skipping for brevity

### Step 7: Predictions and Conclusions ----

# get predictions from both models
p1 <- predict(nn1, test_data)

p2 <- predict(nn2, test_data)

# if you normalized price, convert predictions back to dollars

# calculate r-squared using either mvrsquared::calc_rsquared or caret::R2

# calculate RMSE and MAE (check caret for these functions)

results <- tibble(
  model = c("neuralnet", "caret-nnet"),
  r2 = c(R2(test_data$Price, p1[,1]), R2(test_data$Price, p2)),
  rmse = c(RMSE(test_data$Price, p1[,1]), RMSE(test_data$Price, p2)),
  mae = c(MAE(test_data$Price, p1[,1]), MAE(test_data$Price, p2)),
)

results

# plot the original price versus the predicted price for both models
predictions <- 
  tibble(
    neuralnet = p1[,1],
    caret = p2,
    actual = test_data$Price
  ) |>
  pivot_longer(
    all_of(c("neuralnet", "caret")),
    names_to = "model",
    values_to = "prediction"
  )

predictions |>
  ggplot(aes(x = prediction, y = actual, color = model)) + 
  geom_point(alpha = 0.5) +
  facet_wrap(~model, nrow = 2) +
  theme(legend.position = "none")

# which model do you think works better?

# neuralnet by a mile. 
