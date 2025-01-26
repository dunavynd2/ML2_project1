### Week 2 Support Vector Machines ----

### Set up ----

# load libraries
library(tidyverse) # For all things data manipulation
library(GGally) # for data exploration
library(caret) # For confusionMatrix(), training ML models, and more
library(class) # For knn()
library(pROC)  # For ROC curve and estimating the area under the ROC curve
library(fastDummies) # To create dummy variable (one hot encoding)

# set a random seed for reproducibility
set.seed(202)

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
# Next we should scale predictors. The kernel trick requires distance 
# calculations and distance functions are sensitive to scale.
# The first column (target variable Survived) doesn't need to be scaled
# Remember that this variable is categorical by nature. Survived: Yes or No!

#Again, we use information from train set to scale test dataset!

# Note caret is smart. It only scales numeric variables

standardizer <- preProcess(
  train_data |>
    select(
      where(function(x) ! is.integer(x) & ! is.factor(x)) # only true numeric, not our dummies
    ),
  method = c("center", "scale")
)

train_data <- predict(standardizer, train_data)

test_data <- predict(standardizer, test_data)


### Step 5: Feature & Model Selection ----
tr_control <- trainControl( # store since we will reuse
  method = "cv", number = 10, # 10-fold cross validation
  classProbs = TRUE,  # Enable probability predictions
  summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
)

svm_linear <- train(
  Survived ~.,
  data = train_data,
  method = "svmLinear",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_linear)

svm_radial <- train(
  Survived ~.,
  data = train_data,
  method = "svmRadial",
  tuneGrid = expand.grid(C = c(0.01,0.1,1,5,10), sigma = c(0.5,1,2,3)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_radial)

svm_radial$bestTune

svm_poly <- train(
  Survived ~.,
  data = train_data,
  method = "svmPoly",
  tuneLength = 4, # will automatically try different parameters with CV
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

plot(svm_poly)

svm_poly$bestTune

### Step 6: Model Validation ----
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
# let's choose the best model, polynomial

preds <- predict(svm_poly, test_data, type = "prob")

roc_poly <- roc(
  test_data$Survived,
  preds$Yes
)

plot(roc_poly)

roc_poly$auc

confusion <- confusionMatrix(
  test_data$Survived,
  preds |>
    mutate(
      class = ifelse(Yes >= 0.5, "Yes", "No") |>
        factor()
    ) |>
    select(class) |>
    unlist()
    
)

confusion$byClass[c("Precision", "Recall")]


