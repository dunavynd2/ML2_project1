### Week 4 Exercises ----

#The dataset on American college and university ranking contains information on 471 
#American and college universities offering an undergraduate program. For each university, 
#there are 17 measurements that include continuous measurements (such as tuition and graduation rate)
#and 3 categorical ones (such as location). 


### Set up ----

# load libraries

#(Install and) load the following libraries
library(tidyverse)
library(factoextra) # For PCA helper functions
library(corrplot) # For some correlation and PCA plots

# set a random seed for reproducibility
set.seed(1234)

data <- read_csv("data-raw/Universities.csv")

str(data)

summary(data)

final_data <- data[,-c(1:3)] #Remove categorical variables (the first three columns)


############# Run PCA #############

#Run PCA without target variable

#The first 6 PCs explain 80.89% of the variation

#Visualize explained variances per component


############# Variables Analysis #############

#graph of variables mapped over the first two components

#Extract PCA results for variables

#Quality of representation (analyze row-wise):
#Visualize squared cosine of variables


#Contribution of variables to PCs (analyze column-wise):
#Visualize contribution of variables

#Bar plots of variables contributions, sorted from highest to lowest
# Contributions of variables to PC1

# Contributions of variables to PC2


############# Rows/observations Analysis #############

#Extract PCA results for individuals (i.e., observations / colleges)


#Attach college names and the principal components values of each university

#Plot of PC1 vs PC2 for all the colleges, color coded by Public(1) vs. Private(2)
