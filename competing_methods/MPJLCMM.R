# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 09/12/2021

library(lcmm)

# joint model including 2 classes specify longitudinal model for 2 classes,
# without estimation B

MPJLCMM_fit <- function(data, time_dep_feat, time_indep_feat) {
    fixed_form <<- formula(paste(paste(time_dep_feat, collapse = ' + '), " ~ T_long"))
    m1<-do.call(multlcmm, args = list(fixed_form, random = ~T_long, subject = "id",
                            nwg = FALSE, ng = 1, data = data, maxiter = 0))
    long_model <- do.call(multlcmm, args = list(fixed_form, random = ~T_long, subject = "id",
                            nwg = TRUE, ng = 2, mixture = ~T_long,
                            data = data, maxiter = 0, B = m1))
    classmb_form <- as.formula(paste("~ ", paste(time_indep_feat, collapse = ' + ')))
    mixture_form <- vector()
    for(i in 1:length(time_indep_feat)){
        mixture_form[i] <- paste("mixture(", time_indep_feat[i], ")")
    }
    survival_form <- as.formula(paste("Surv(T_survival, delta) ~ ", paste(mixture_form, collapse = ' + ')))
    joint_model <- do.call(mpjlcmm, args = list(longitudinal = list(long_model), subject = "id",
                           ng = 2, data = data, classmb = classmb_form,
                           survival = survival_form, hazard = "Weibull", hazardtype = "PH"))

    return (list(long_model, joint_model))
}


MPJLCMM_score <- function(trained_long_model, trained_joint_model, time_indep_feat, data) {
    # predictive marker
    classmb_form <- as.formula(paste("~ ", paste(time_indep_feat, collapse = ' + ')))
    mixture_form <- vector()
    for(i in 1:length(time_indep_feat)){
        mixture_form[i] <- paste("mixture(", time_indep_feat[i], ")")
    }
    survival_form <- as.formula(paste("Surv(T_max, delta) ~ ", paste(mixture_form, collapse = ' + ')))
    mpjlcmm_pred <- do.call(mpjlcmm, list(longitudinal = list(trained_long_model), B = trained_joint_model$best,
                         maxiter = 0, subject = "id", ng = 2, data = data,
                         classmb = classmb_form, survival = survival_form,
                         hazard = "Weibull", hazardtype = "PH"))

    return(mpjlcmm_pred)
}