Project 2 - Karen Lopez
================
October 16, 2020

  - [Introduction](#introduction)
  - [Data](#data)
  - [Summarizations](#summarizations)
  - [Modeling](#modeling)
  - [Automation & Blog Post](#automation-blog-post)

## Introduction

This project uses the bike sharing data set, day.csv, that’s located
*[here](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)*
and contains 731 observations with 15 attributes. For modeling, the
response variable is the count of total rental bikes rented (*cnt*) and
12 of the 14 remaining variables will be considered for predictor
variables (*casual* and *registered* are omitted). The 12 remaining
variables include date, season, year, month, holiday, weekday, working
day, weather, temperature, feeling temperature, humidity, and wind
speed.  
Purpose of analysis:  
Methods I’ll use:  
To begin, necessary libraries are loaded so their functions are
accessible and global variables are set.

``` r
library(readr)
library(caret)
library(knitr)
library(corrplot)
library(dplyr)
num <- 12
```

## Data

The dataset is read in using a relative path and saved as an object. The
data is then randomly split into a training and testing set, where 70%
of the data goes into the training set and the remaining 30% goes into
the testing set.

``` r
bikeData <- read_csv("Bike-Sharing-Dataset/day.csv") #read in data
```

    ## Parsed with column specification:
    ## cols(
    ##   instant = col_double(),
    ##   dteday = col_date(format = ""),
    ##   season = col_double(),
    ##   yr = col_double(),
    ##   mnth = col_double(),
    ##   holiday = col_double(),
    ##   weekday = col_double(),
    ##   workingday = col_double(),
    ##   weathersit = col_double(),
    ##   temp = col_double(),
    ##   atemp = col_double(),
    ##   hum = col_double(),
    ##   windspeed = col_double(),
    ##   casual = col_double(),
    ##   registered = col_double(),
    ##   cnt = col_double()
    ## )

``` r
bikeData$cnt <- as.factor(bikeData$cnt) #convert response to factor

#create partitions in data indexes with 70% going in the training set
set.seed(num)
train <- sample(1:nrow(bikeData), size = nrow(bikeData)*0.7)
test <- dplyr::setdiff(1:nrow(bikeData), train)

#create train and test data set with split indexes
bikeDataTrain <- bikeData[train, ]
bikeDataTest <- bikeData[test, ]

#kable(bikeDataTrain) #check for 70% of data -> 511
#kable(bikeDataTest) #check for 30% of data -> 220

kable(head(bikeDataTrain)) #preview train data
```

| instant | dteday     | season | yr | mnth | holiday | weekday | workingday | weathersit |     temp |    atemp |      hum | windspeed | casual | registered | cnt  |
| ------: | :--------- | -----: | -: | ---: | ------: | ------: | ---------: | ---------: | -------: | -------: | -------: | --------: | -----: | ---------: | :--- |
|     450 | 2012-03-25 |      2 |  1 |    3 |       0 |       0 |          0 |          2 | 0.437500 | 0.437488 | 0.880833 |  0.220775 |   1532 |       3464 | 4996 |
|     346 | 2011-12-12 |      4 |  0 |   12 |       0 |       1 |          1 |          1 | 0.238333 | 0.270196 | 0.670833 |  0.063450 |    143 |       3167 | 3310 |
|     336 | 2011-12-02 |      4 |  0 |   12 |       0 |       5 |          1 |          1 | 0.314167 | 0.331433 | 0.625833 |  0.100754 |    268 |       3672 | 3940 |
|     247 | 2011-09-04 |      3 |  0 |    9 |       0 |       0 |          0 |          1 | 0.709167 | 0.665429 | 0.742083 |  0.206467 |   2521 |       2419 | 4940 |
|     174 | 2011-06-23 |      3 |  0 |    6 |       0 |       4 |          1 |          2 | 0.728333 | 0.693833 | 0.703333 |  0.238804 |    746 |       4044 | 4790 |
|     453 | 2012-03-28 |      2 |  1 |    3 |       0 |       3 |          1 |          1 | 0.484167 | 0.470950 | 0.481250 |  0.291671 |    674 |       5024 | 5698 |

``` r
kable(head(bikeDataTest)) #preview test data
```

| instant | dteday     | season | yr | mnth | holiday | weekday | workingday | weathersit |     temp |    atemp |      hum | windspeed | casual | registered | cnt  |
| ------: | :--------- | -----: | -: | ---: | ------: | ------: | ---------: | ---------: | -------: | -------: | -------: | --------: | -----: | ---------: | :--- |
|       4 | 2011-01-04 |      1 |  0 |    1 |       0 |       2 |          1 |          1 | 0.200000 | 0.212122 | 0.590435 | 0.1602960 |    108 |       1454 | 1562 |
|       5 | 2011-01-05 |      1 |  0 |    1 |       0 |       3 |          1 |          1 | 0.226957 | 0.229270 | 0.436957 | 0.1869000 |     82 |       1518 | 1600 |
|       6 | 2011-01-06 |      1 |  0 |    1 |       0 |       4 |          1 |          1 | 0.204348 | 0.233209 | 0.518261 | 0.0895652 |     88 |       1518 | 1606 |
|       7 | 2011-01-07 |      1 |  0 |    1 |       0 |       5 |          1 |          2 | 0.196522 | 0.208839 | 0.498696 | 0.1687260 |    148 |       1362 | 1510 |
|       8 | 2011-01-08 |      1 |  0 |    1 |       0 |       6 |          0 |          2 | 0.165000 | 0.162254 | 0.535833 | 0.2668040 |     68 |        891 | 959  |
|       9 | 2011-01-09 |      1 |  0 |    1 |       0 |       0 |          0 |          1 | 0.138333 | 0.116175 | 0.434167 | 0.3619500 |     54 |        768 | 822  |

## Summarizations

summary statistics and plot with some general explanations

## Modeling

fit some models (two)

## Automation & Blog Post

automate so it creates report for each day of the week
