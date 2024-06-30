# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 09/12/2021

library(JMbayes)

# estimation
fit <- function(data, time_dep_feat, time_indep_feat) {
    u_cols = c("id", "T_survival", "delta")
    for(i in 1:length(time_indep_feat)){
        u_cols <- append(u_cols,  time_indep_feat[[i]])
    }
    data.id <- unique(data[, u_cols])
    # update account several shared associations
    ## calibration with indepedent estimation of both longitudinal and survival model
    mixed_form <- list()
    families <- list()
    for(i in 1:length(time_dep_feat)){
        mixed_form[[i]] <- as.formula(paste(time_dep_feat[[i]], " ~ T_long + (T_long | id)"))
        families[[i]] <- gaussian
    }
    MixedModelFit <- mvglmer(mixed_form, data = data, families = families)
    surv_form <- as.formula(paste("Surv(T_survival, delta) ~ ", paste(time_indep_feat, collapse = ' + ')))
    CoxFit <- coxph(surv_form, data = data.id, model = TRUE)

    ## estimation of Multivariate Joint Model
    jmbayes_mv <- mvJointModelBayes(MixedModelFit, CoxFit, timeVar = "T_long")
    
    return(jmbayes_mv)
#    Forms <- list()
#    for(i in 1:length(time_dep_feat)){
#        Forms[[time_dep_feat[[i]]]] = "values"
#    #    Forms[[2 * i]] = list(fixed = as.formula("~ 1"), random = as.formula("~ 1"), indFixed = 2, indRandom = 2, name = "slope")
#    #    names(Forms)[c(2 * i - 1, 2 * i)] <- c(time_dep_feat[[i]], time_dep_feat[[i]])    
#    }
#    jmbayes_mv_update <- update(jmbayes_mv, Formulas = Forms)
#    return(jmbayes_mv_update)
}


score <- function(trained_model, data) {
    # predictive marker
    t_max <- data$T_max
    data$delta[which(data$T_survival> t_max)] <- 0
    data$T_survival[which(data$T_survival > t_max)] <- t_max

    JMbayes_marker <- lights_JMbayes_marker(object = trained_model, newdata = data,
    last.time = t_max, survTimes = t_max, idVar = "id", simulate = TRUE)
    # the n-length vector of predictive markers (for the n subjects)
    return(JMbayes_marker$predictive_marker)
}

prob <- function(trained_model, data) {
    # predictive marker
    t_max <- data$T_max
    data$delta[which(data$T_survival> t_max)] <- 0
    data$T_survival[which(data$T_survival > t_max)] <- t_max
    data.id <- data[!duplicated(data$id), ]
    t_max_ <- data.id$T_max
    probs <- survfitJM(object = trained_model, newdata = data,
    last.time = NULL, survTimes = t_max_, idVar = "id", simulate = TRUE)
    return(probs$summaries)
    
    #data.id <- data[!duplicated(data$id), ]
    #survPreds <- vector("list", nrow(data.id))
    #for (i in 1:nrow(data.id)) {
    #    ND <- data[data$id == data.id$id[i],]
    #    survPreds[[i]] <- survfitJM(trained_model, newdata = ND, survTimes = data.id$T_max[i])
    #}
    #return(survPreds)
}

lights_JMbayes_marker <- function (object,
                                   newdata,
                                   survTimes,
                                   last.time,
                                   idVar = "id",
                                   M = 200L,
                                   scale = 1.6,
                                   log = FALSE,
                                   CI.levels = c(0.025, 0.975),
                                   seed = 1L, ...){

  # object    # 'mvJMbayes' objects estimated on training dataset
  # newdata   # dataset including new subjects to get their predictive marker
  # last.time # a scalaire or vector indicating the time to predict (or tmax)
  # survTimes # a scalaire or vector indicating the time to predict (or tmax)
  # other arguments     # see help page of surfitJMbayes function


  # if (!require("JMbayes"))
  #   stop("'JMbayes' is required.\n")
  if (!inherits(object, "mvJMbayes"))
    stop("Use only with 'mvJMbayes' objects.\n")
  if (!is.data.frame(newdata) || nrow(newdata) == 0L)
    stop("'newdata' must be a data.frame with more than one rows.\n")
  if (is.null(newdata[[idVar]]))
    stop("'idVar' not in 'newdata'.\n")
  timeVar <- object$model_info$timeVar
  TermsU <- object$model_info$coxph_components$TermsU
  control <- object$control
  families <- object$model_info$families
  fams <- sapply(families, "[[", "family")
  links <- sapply(families, "[[", "link")
  n_outcomes <- length(object$model_info$families)
  seq_n_outcomes <- seq_len(n_outcomes)
  componentsL <- object$model_info$mvglmer_components
  componentsS <- object$model_info$coxph_components
  TermsL <- componentsL[grep("Terms", names(componentsL),
                             fixed = TRUE)]
  TermsL <- lapply(TermsL, function(x) {
    environment(x) <- parent.frame()
    x
  })
  TermsFormulas_fixed <- object$model_info$coxph_components$TermsFormulas_fixed
  TermsFormulas_random <- object$model_info$coxph_components$TermsFormulas_random
  build_model_matrix <- function(Terms, data) {
    out <- vector("list", length(Terms))
    for (i in seq_along(Terms)) {
      MF <- model.frame(Terms[[i]], data = data, na.action = NULL)
      out[[i]] <- model.matrix(Terms[[i]], MF)
    }
    out
  }
  mfLna <- lapply(TermsL, FUN = model.frame.default, data = newdata)
  mfLnaNULL <- lapply(TermsL, FUN = model.frame.default, data = newdata,
                      na.action = NULL)
  na.inds <- lapply(mfLna, FUN = function(x) as.vector(attr(x,
                                                            "na.action")))
  na.inds <- lapply(na.inds, FUN = function(x) {
    if (is.null(x)) {
      rep(TRUE, nrow(newdata))
    }else{
      !seq_len(nrow(newdata)) %in% x
    }
  })
  na.inds <- na.inds[grep("TermsX", names(na.inds))]
  TermsS <- componentsS[grep("Terms", names(componentsS),
                             fixed = TRUE)][[1]]
  formulasL <- lapply(TermsL, FUN = function(x) formula(x))
  formulasL2 <- lapply(TermsL, FUN = function(x) formula(delete.response(x)))
  formulasS <- formula(delete.response(TermsS))
  allvarsL <- unique(unlist(lapply(formulasL, FUN = all.vars)))
  allvarsS <- all.vars(formulasS)
  allvarsLS <- unique(c(allvarsL, allvarsS))
  respVars <- unlist(componentsL[grep("respVar", names(componentsL),
                                      fixed = TRUE)], use.names = FALSE)
  id <- as.numeric(unclass(newdata[[idVar]]))
  id <- id. <- match(id, unique(id))
  obs.times <- split(newdata[[timeVar]], id)
  obs.times[] <- mapply(function(x, nams) {
    names(x) <- nams
    x
  }, obs.times, split(row.names(newdata), id), SIMPLIFY = FALSE)
  mfL <- lapply(TermsL, FUN = model.frame.default, data = newdata,
                na.action = NULL)
  y <- lapply(mfL[grep("TermsX", names(mfL))], model.response)
  if (any(!allvarsLS %in% names(newdata)))
    stop("The following variable or variables: ", paste(c(allvarsL,
                                                          allvarsS)[which(!c(allvarsL, allvarsS) %in% names(newdata))],
                                                        "\n", sep = " "), "should be included in 'newdata'.\n")
  idNewL <- factor(newdata[[idVar]], levels = unique(newdata[[idVar]]))
  n.NewL <- length(unique(idNewL))
  na.inds2 <- vector("list", n.NewL)
  for (i in 1:n.NewL) {
    for (j in 1:n_outcomes) {
      tmp <- split(na.inds[[j]], id)
      na.inds2[[i]][[j]] <- unname(unlist(tmp[[i]]))
    }
  }
  if (is.null(last.time)) {
    last.time <- tapply(newdata[[timeVar]], idNewL, FUN = max, simplify = FALSE)
  }else{
    last.time <- rep_len(last.time, length.out = length(unique(newdata[[idVar]])))
  }
  Time <- componentsS$Time
  if (is.null(survTimes) || !is.numeric(survTimes)) {
    survTimes <- seq(min(Time), max(Time) + 0.01, length.out = 35L)
  }
  times.to.pred_upper <- lapply(last.time, FUN = function(t) survTimes[survTimes > t])
  times.to.pred_lower <- mapply(FUN = function(t1, t2) as.numeric(c(t1, t2[-length(t2)])), last.time, times.to.pred_upper, SIMPLIFY = FALSE)
  if (control$GQsurv == "GaussKronrod") {
    GQsurv <- JMbayes:::gaussKronrod()
  }else{
    GQsurv <- JMbayes:::gaussLegendre(control$GQsurv.k)
    }
  wk <- GQsurv$wk
  sk <- GQsurv$sk
  K <- length(sk)
  P <- lapply(last.time, FUN = function(x) x/2)
  P1 <- mapply(FUN = function(x, y) (x + y)/2, times.to.pred_upper,
               times.to.pred_lower, SIMPLIFY = FALSE)
  P2 <- mapply(FUN = function(x, y) (x - y)/2, times.to.pred_upper,
               times.to.pred_lower, SIMPLIFY = FALSE)
  GK_points_postRE <- matrix(unlist(lapply(P, FUN = function(x) outer(x, sk + 1))), ncol = K, byrow = TRUE)
  GK_points_CumHaz <- mapply(FUN = function(x, y, sk) outer(y, sk) + x, P1, P2, SIMPLIFY = F, MoreArgs = list(sk = sk))
  idGK <- rep(seq_len(n.NewL), each = K)
  newdata[[idVar]] <- match(newdata[[idVar]], unique(newdata[[idVar]]))
  newdata.GK.postRE <- JMbayes:::right_rows(newdata, newdata[[timeVar]],
                                  newdata[[idVar]], GK_points_postRE)
  newdata.GK.postRE[[timeVar]] <- c(t(GK_points_postRE))
  newdata.GK.postRE[[idVar]] <- match(newdata.GK.postRE[[idVar]],
                                      unique(newdata.GK.postRE[[idVar]]))
  newdata.GK.CumHaz <- vector("list", n.NewL)
  postMeans <- object$statistics$postMeans
  betas <- postMeans[grep("betas", names(postMeans),
                          fixed = TRUE)]
  sigmas <- postMeans[grep("sigma", names(postMeans),
                           fixed = TRUE)]
  sigma <- vector("list", n_outcomes)
  if (any(which_gaussian <- which(fams == "gaussian"))) {
    sigma[which_gaussian] <- sigmas
  }
  D <- postMeans[grep("^D", names(postMeans), fixed = FALSE)]
  gammas <- postMeans[grep("^gammas", names(postMeans),
                           fixed = FALSE)]
  alphas <- postMeans[grep("^alphas", names(postMeans),
                           fixed = FALSE)]
  Bs_gammas <- postMeans[grep("^Bs.*gammas$", names(postMeans),
                              fixed = FALSE)]
  invD <- postMeans[grep("inv_D", names(postMeans))]
  Formulas <- object$model_info$Formulas
  find_outcome <- function(nams_Formulas, repsVars) {
    out <- numeric(length(nams_Formulas))
    for (i in seq_along(respVars)) {
      ind <- grep(respVars[i], nams_Formulas, fixed = TRUE)
      out[ind] <- i
    }
    out
  }
  outcome <- find_outcome(names(Formulas), respVars)
  indFixed <- lapply(Formulas, "[[", "indFixed")
  indRandom <- lapply(Formulas, "[[", "indRandom")
  RE_inds <- object$model_info$RE_inds
  RE_inds2 <- object$model_info$RE_inds2
  Interactions <- object$model_info$Interactions
  trans_Funs <- object$model_info$transFuns
  survMats.last <- vector("list", n.NewL)
  for (i in seq_len(n.NewL)) {
    newdata.GK.postRE.i <- newdata.GK.postRE[newdata.GK.postRE[[idVar]] == i, ]
    newdata.i <- newdata[newdata[[idVar]] == i, ]
    idL.i <- match(newdata.i[[idVar]], unique(newdata.i[[idVar]]))
    idL2 <- rep(list(unique(idL.i)), n_outcomes)
    idL <- rep(list(idL.i), n_outcomes)
    for (j in 1:length(idL)) {
      idL[[j]] <- idL[[j]][na.inds2[[i]][[j]]]
    }
    idGK <- match(newdata.GK.postRE.i[[idVar]], unique(newdata.GK.postRE.i[[idVar]]))
    ids <- rep(list(idGK), n_outcomes)
    idTs <- ids[outcome]
    newdata.i.id <- JMbayes:::last_rows(newdata.i, idL.i)
    newdata.i.id[[timeVar]] <- last.time[[i]]
    Us <- lapply(TermsU, function(term) {
      model.matrix(term, data = newdata.GK.postRE.i)
    })
    mfX <- lapply(TermsL[grep("TermsX", names(TermsL))],
                  FUN = function(x) model.frame.default(delete.response(x),
                                                        data = newdata.GK.postRE.i))
    mfX_long <- lapply(TermsL[grep("TermsX", names(TermsL))],
                       FUN = function(x) model.frame.default(delete.response(x),
                                                             data = newdata.i))
    mfX_long.resp <- lapply(TermsL[grep("TermsX", names(TermsL))],
                            FUN = function(x) model.frame.default(x, data = newdata.i))
    y_long <- lapply(mfX_long.resp, model.response)
    mfZ <- lapply(TermsL[grep("TermsZ", names(TermsL))],
                  FUN = function(x) model.frame.default(x, data = newdata.GK.postRE.i))
    mfZ_long <- lapply(TermsL[grep("TermsZ", names(TermsL))],
                       FUN = function(x) model.frame.default(x, data = newdata.i))
    X_surv_H_postRE <- mapply(FUN = function(x, y) model.matrix.default(x,
                                                                        y), formulasL2[grep("TermsX", names(formulasL2))],
                              mfX, SIMPLIFY = FALSE)
    X_long <- mapply(FUN = function(x, y) model.matrix.default(x,
                                                               y), formulasL2[grep("TermsX", names(formulasL2))],
                     mfX_long.resp, SIMPLIFY = FALSE)
    Xbetas_postRE <- JMbayes:::Xbetas_calc(X_long, betas)
    Z_surv_H_postRE <- mapply(FUN = function(x, y) model.matrix.default(x,
                                                                        y), formulasL2[grep("TermsZ", names(formulasL2))],
                              mfZ, SIMPLIFY = FALSE)
    Z_long <- mapply(FUN = function(x, y) model.matrix.default(x,
                                                               y), formulasL2[grep("TermsZ", names(formulasL2))],
                     mfZ_long, SIMPLIFY = FALSE)
    for (j in seq_len(length(Z_long))) {
      Z_long[[j]] <- Z_long[[j]][na.inds2[[i]][[j]], ,
                                 drop = FALSE]
    }
    XXs <- build_model_matrix(TermsFormulas_fixed, newdata.GK.postRE.i)
    XXsbetas <- JMbayes:::Xbetas_calc(XXs, betas, indFixed, outcome)
    ZZs <- build_model_matrix(TermsFormulas_random, newdata.GK.postRE.i)
    W1s <- splines::splineDesign(control$knots, GK_points_postRE[i,
                                                                 ], ord = control$ordSpline, outer.ok = TRUE)
    W2s <- model.matrix(formulasS, newdata.GK.postRE.i)[,
                                                        -1, drop = FALSE]
    post_b_input_only <- lapply(indRandom, function(x) rbind(rep(0,max(x))))
    Wlongs <- JMbayes:::designMatLong(XXs, betas, ZZs, post_b_input_only,
                            ids, outcome, indFixed, indRandom, Us, trans_Funs)
    Pw <- unlist(P[idGK]) * wk
    P <- unlist(P)
    col_inds = attr(Wlongs, "col_inds")
    row_inds_Us = seq_len(nrow(Wlongs))
    survMats.last[[i]][["idL"]] <- idL
    survMats.last[[i]][["idL2"]] <- lapply(idL, length)
    survMats.last[[i]][["idL3"]] <- unlist(idL)
    survMats.last[[i]][["idL3"]] <- c(unlist(idL)[-length(unlist(idL))] !=
                                        unlist(idL)[-1L], TRUE)
    survMats.last[[i]][["idGK"]] <- idGK
    survMats.last[[i]][["idGK_fast"]] <- c(idGK[-length(idGK)] !=
                                             idGK[-1L], TRUE)
    survMats.last[[i]][["idTs"]] <- idTs
    survMats.last[[i]][["Us"]] <- Us
    factor2numeric <- function(x) {
      if (is.factor(x)) {
        if (length(levels(x)) > 2) {
          stop("Currently only binary outcomes can be considered.")
        }
        as.numeric(x == levels(x)[2L])
      }
      else x
    }
    survMats.last[[i]][["y_long"]] <- lapply(y_long,
                                             factor2numeric)
    survMats.last[[i]][["Xbetas"]] <- Xbetas_postRE
    survMats.last[[i]][["Z_long"]] <- Z_long
    survMats.last[[i]][["X_long"]] <- X_long
    survMats.last[[i]][["RE_inds"]] <- RE_inds
    survMats.last[[i]][["RE_inds2"]] <- RE_inds2
    survMats.last[[i]][["W1s"]] <- W1s
    survMats.last[[i]][["W2s"]] <- W2s
    survMats.last[[i]][["XXs"]] <- XXs
    survMats.last[[i]][["XXsbetas"]] <- XXsbetas
    survMats.last[[i]][["ZZs"]] <- ZZs
    survMats.last[[i]][["col_inds"]] <- col_inds
    survMats.last[[i]][["row_inds_Us"]] <- row_inds_Us
    survMats.last[[i]][["Pw"]] <- Pw
    survMats.last[[i]][["P"]] <- P
  }
  survMats <- vector("list", n.NewL)
  for (i in 1:n.NewL) {
    survMats[[i]] <- vector("list", nrow(GK_points_CumHaz[[i]]))
  }
  modes.b <- matrix(0, n.NewL, ncol(D[[1]]))
  invVars.b <- Vars.b <- vector("list", n.NewL)
  for (i in 1:n.NewL) {
    Data <- list(idL = survMats.last[[i]]$idL, idL2 = survMats.last[[i]]$idL2,
                 idGK = which(survMats.last[[i]]$idGK_fast) - 1, ids = survMats.last[[i]]$ids,
                 idTs = survMats.last[[i]]$idTs, Us = survMats.last[[i]]$Us,
                 y = survMats.last[[i]]$y_long,
                 Xbetas = survMats.last[[i]]$Xbetas,
                 Z = survMats.last[[i]]$Z_long,
                 RE_inds = survMats.last[[i]]$RE_inds,
                 RE_inds2 = survMats.last[[i]]$RE_inds2, W1s = survMats.last[[i]]$W1s,
                 W2s = survMats.last[[i]]$W2s, XXsbetas = survMats.last[[i]]$XXsbetas,
                 ZZs = survMats.last[[i]]$ZZs, col_inds = survMats.last[[i]]$col_inds,
                 row_inds_Us = survMats.last[[i]]$row_inds_Us,
                 Bs_gammas = unlist(Bs_gammas),
                 gammas = unlist(gammas), # coefficients of baseline covariates parameter
                 alphas = unlist(alphas), # coefficients of association parameters
                 fams = fams, links = links, sigmas = sigma, invD = invD[[1]],
                 Pw = survMats.last[[i]]$Pw, trans_Funs = trans_Funs,
                 wk = wk, idL3 = which(survMats.last[[i]]$idGK_fast) -
                   1)
    if (is.null(Data$gammas))
      Data$gammas <- numeric(0)
    ff <- function(b, Data) - JMbayes:::log_post_RE_svft(b, Data = Data)
    gg <- function(b, Data) JMbayes:::cd(b, ff, Data = Data, eps = 0.001)
    start <- rep(0, ncol(D[[1]]))
    opt <- optim(start, ff, gg, Data = Data, method = "BFGS", hessian = TRUE, control = list(maxit = 200, parscale = rep(0.1, ncol(D[[1]]))))
    modes.b[i, ] <- opt$par
    invVars.b[[i]] <- opt$hessian/scale
    Vars.b[[i]] <- scale * solve(opt$hessian)
  }
  res <- list(modes.b = modes.b,
              Vars.b = Vars.b)

  # check the correspondance between coefficents and shared associations
  if(length(names(object$model_info$Formulas))!=length(object$statistics$postModes$alphas))
    stop("Number of shared associations is not equal to the number of coefficients for shared association \n")

  newdata.id <- newdata[!duplicated(newdata$id), ]

  # time-independent design vector for baseline regression
  mfZ <- model.frame(formula(object$model_info$coxph_components$Terms), data = newdata.id)
  Z <- model.matrix(formula(object$model_info$coxph_components$Terms), mfZ)
  Z <- Z[, names(object$statistics$postModes$gammas)]
  # part of marker associated with time-independent covariates
  if (length(object$statistics$postModes$gammas) == 1){
      predictive_marker <- as.vector(exp(object$statistics$postModes$gammas * Z))
  }else{   
      predictive_marker <- as.vector(exp(Z%*%object$statistics$postModes$gammas))
      }
  pred_marker_list <- list(exp_eta_baseline = predictive_marker)

  if(!is.null(survTimes) && !is.null(last.time)){
    if(length(survTimes)==1)
      Tsurv <- rep(survTimes, nrow(newdata.id))
    if(length(survTimes)==nrow(newdata))
      Tsurv <- survTimes[!duplicated(newdata$id)]
    if(length(survTimes)==nrow(newdata.id))
      Tsurv <- survTimes
  }else{
    Tsurv <- as.vector(data.id[all.vars(formula(object$model_info$coxph_components$Terms))][, 1])
  }

  newdata.id[[object$model_info$timeVar]] <- Tsurv
  pred_marker_list$exp_predictor <- matrix(NA,
                                           nrow = nrow(newdata.id),
                                           ncol = length(object$statistics$postModes$alphas))

  # loop on shared association
  for(s in 1:length(object$statistics$postModes$alphas)){
    name_tmp <- names(object$statistics$postModes$alphas)[s]
    for(k in 1:length(respVars)){
      if(grepl(respVars[k], name_tmp, fixed = TRUE))
        index <- k
    }
    # get random effects
    mat_RE <- as.matrix(modes.b[, RE_inds2[[s]]])
    # get estimated coefficients
    beta_tmp <- object$statistics$postModes[which(grepl("betas",
                names(object$statistics$postModes), fixed = TRUE))][[index]]
    # create design matrices
    formFixed <- object$model_info$Formulas[[name_tmp]]$fixed
    mfX.id <- model.frame(formFixed, data = newdata.id)
    Xtime <- model.matrix(formFixed, mfX.id)
    formRandom <- object$model_info$Formulas[[name_tmp]]$random
    mfU.id <- model.frame(formRandom, data = newdata.id)
    Utime <- model.matrix(formRandom, mfU.id)
    # compute marker contribution for the sth shared association
    pred_marker_list$exp_predictor[, s] <- as.vector(exp(object$statistics$postModes$alphas[name_tmp]
                                                         *(Xtime%*%beta_tmp[object$model_info$Formulas[[name_tmp]]$indFixed] +
                                                             rowSums(Utime*mat_RE))
                                                         )
                                                     )
    predictive_marker <- predictive_marker * pred_marker_list$exp_predictor[, s]
  }
  pred_marker_list$predictive_marker <- predictive_marker

  # management of the output
  res$pred_marker_list <- pred_marker_list
  res$predictive_marker <- predictive_marker
  res

}