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
airbnb <- read_csv("AirbnbListings.csv")

# check data structure
str(airbnb)

# any obvious pre-formatting, do here (e.g., numeric to categorical etc.)
#Create dummy variables for each level of the categorical variables of interest
#Removing categorical variables and one dummy level of each categorical variables
final_data <-
  airbnb |>
  dummy_cols(
    select_columns = c('neighborhood', 'room_type','bathrooms','superhost'),
    remove_selected_columns = T,
    remove_first_dummy = T
  )
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

str(train_data)

test_data$host_since<- mdy(test_data$host_since)
test_data$host_since <- as.numeric(test_data$host_since)
train_data$host_since <- mdy(train_data$host_since)
train_data$host_since <- as.numeric(train_data$host_since)
str(test_data)
str(train_data)
na.omit(train_data)
na.omit(test_data)

### Step 4: Feature Engineering ----

# normalize the data. Do not normalize price.
normalizer <- preProcess(
  train_data |>
    select(
      where(function(x) ! is.integer(x) & ! is.factor(x))
    ),
  method = "range"
)

train_data <- predict(normalizer, train_data)

test_data <- predict(normalizer, test_data)

### Step 5: Feature & Model Selection ----
colnames(train_data) <- gsub(" ", "_", colnames(train_data))
colnames(test_data) <- gsub(" ", "_", colnames(test_data))

# Median imputation for multiple columns
train_data <- train_data %>%
  mutate(across(where(~ any(is.na(.))), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
test_data <- test_data %>%
  mutate(across(where(~ any(is.na(.))), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))


train_data |>
  is.na() |>
  colSums()

test_data |>
  is.na() |>
  colSums()
# Train a neural net with one hidden layer and 5 units using the neuralnet package
# Use a ReLu activation function
# if you don't converge change the stepmax parameter to a larger value or change the learning rate
nn1 <- 
  neuralnet(
    price ~.,
    data = train_data,
    linear.output = TRUE, # classification vs regression
    act.fct = relu,
    hidden = 1
  )

plot(nn1)

# Train a second neural net using caret with the 'nnet' method.
# use 10-fold cross validation and a grid search:
# size from 1 to 10 and decay between 0.01 and 0.2
# What's the best size and decay?

nn2 <- train(
  price ~.,
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
min_price <- normalizer$ranges[1,"price"]
normalizer$ranges
max_price <- normalizer$ranges[2, "price"]
p1_original <- p1[, 1] * (max_price - min_price) + min_price
p2_original <- p2 * (max_price - min_price) + min_price
actual_prices <- test_data$price * (max_price - min_price) + min_price

# if you normalized price, convert predictions back to dollars

# calculate r-squared using either mvrsquared::calc_rsquared or caret::R2

# calculate RMSE and MAE (check caret for these functions)

results <- tibble(
  model = c("neuralnet", "caret-nnet"),
  r2 = c(R2(actual_prices, p1_original), R2(actual_prices, p2_original)),
  rmse = c(RMSE(actual_prices, p1_original), RMSE(actual_prices, p2_original)),
  mae = c(MAE(actual_prices, p1_original), MAE(actual_prices, p2_original))
)

print(results)

predictions <- tibble(
  neuralnet = p1_original,
  caret = p2_original,
  actual = actual_prices
) |>
  pivot_longer(
    all_of(c("neuralnet", "caret")),
    names_to = "model",
    values_to = "prediction"
  )

plot <- predictions |>
  ggplot(aes(x = prediction, y = actual, color = model)) + 
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") + 
  facet_wrap(~model, nrow = 2) +
  labs(
    x = "Predicted Price",
    y = "Actual Price",
    title = "Predicted vs Actual Prices"
  ) +
  theme(legend.position = "none")
print(plot)

# which model do you think works better?

# neuralnet by a mile. 
