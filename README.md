Project 2 - Karen Lopez
================
October 16, 2020

## Introduction

This project uses the bike sharing data set, day.csv, that’s located
*[here](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)*
and contains 731 observations with 15 attributes. This repo contains 
the R markdown file used to analyze the data and output analysis for
each day of the week included in the data set.

The analysis includes reading in data and partitioning it into a 
training set and testing set. The training set is used to fit two 
different tree models that are each predicted on using the training
set. The performance of the two models are compared in order to select
the best model for this instance. The Monday analysis was performed
first, so the subsequent days are modeled around these results.

## Analysis Links
Here are the links for each day's analysis:  
- [Monday](Monday.md)  
- [Tuesday](Tuesday.md)  
- [Wednesday](Wednesday.md)  
- [Thursday](Thursday.md)  
- [Friday](Friday.md)  
- [Saturday](Saturday.md)  
- [Sunday](Sunday.md)  

## Packages Required
These are the packages required to run the analysis:  
- readr  
- caret  
- knitr  
- corrplot  
- dplyr  
- tidyverse  
- rpart  

## Automation
This is the R script code used for automating the process:  
`library(rmarkdown)  
#save unique days  
weekdays <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")  
#set filenames  
outFile <- paste0(weekdays, ".md")  
#get list for each day with the day parameter  
params = lapply(weekdays, FUN = function(x){list(weekday = x)})  
#create data frame  
reports <- tibble(outFile, params)  
#use data frame to automate R Markdown reports  
apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "Project2_kmlopez_16Oct20.Rmd",
               output_file = x[[1]],
               params = x[[2]])
      })`
