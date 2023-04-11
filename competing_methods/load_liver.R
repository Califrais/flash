# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 09/12/2021

defaultW <- getOption("warn")
options(warn = -1)
library('joineR')

load <- function() {

    data(liver)
    data <- liver
    data$T_long <- data$time
    data$T_survival <- data$survival
    data$delta <- data$cens

    return(data)
}