# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 09/12/2021

defaultW <- getOption("warn")
options(warn = -1)
library("JMbayes")

load <- function() {

    data(aids, package = "JMbayes")
    data <- aids
    data$T_long <- (data$start + data$stop) / 2
    data$T_survival <- data$Time
    data$delta <- data$death
    data$id <- data$patient

    return(data)
}