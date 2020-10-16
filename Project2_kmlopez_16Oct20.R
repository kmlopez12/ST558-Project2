library(rmarkdown)
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
      })