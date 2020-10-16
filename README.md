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
    [Monday]()  
    [Tuesday]()  
    [Wednesday]()  
    [Thursday]()  
    [Friday]()  
    [Saturday]()  
    [Sunday]()  

## Packages Required
These are the packages required to run the analysis:  
    readr  
    caret  
    knitr  
    corrplot  
    dplyr  
    tidyverse  
    rpart  

## Automation
This is the code used for automating the process:  
    (insert render function here)  
    rmarkdown::render(
      input,
      output_file="MondayAnalysis.md",
      params = list(weekday=0)
      envir=globalenv()
    )
