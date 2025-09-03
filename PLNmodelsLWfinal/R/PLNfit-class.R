#' An R6 Class to represent a PLNfit in a standard, general framework
#'
#' @description The function [PLN()] fit a model which is an instance of a object with class [`PLNfit`].
#' Objects produced by the functions [PLNnetwork()], [PLNPCA()], [PLNmixture()] and [PLNLDA()] also enjoy the methods of [PLNfit()] by inheritance.
#'
#' This class comes with a set of R6 methods, some of them being useful for the user and exported as S3 methods.
#' See the documentation for [coef()], [sigma()], [predict()], [vcov()] and [standard_error()].
#'
#' Fields are accessed via active binding and cannot be changed by the user.
#'
## Parameters common to all PLN-xx-fit methods (shared with PLNfit but inheritance does not work)
#' @param responses the matrix of responses (called Y in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param covariates design matrix (called X in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param offsets offset matrix (called O in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param weights an optional vector of observation weights to be used in the fitting process.
#' @param data an optional data frame, list or environment (or object coercible by as.data.frame to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which PLN is called.
#' @param formula model formula used for fitting, extracted from the formula in the upper-level call
#' @param control a list-like structure for controlling the fit, see [PLN_param()].
#' @param config part of the \code{control} argument which configures the optimizer
#' @param nullModel null model used for approximate R2 computations. Defaults to a GLM model with same design matrix but not latent variable.
#' @param B matrix of regression matrix
#' @param Sigma variance-covariance matrix of the latent variables
#' @param Omega precision matrix of the latent variables. Inverse of Sigma.
#'
#' @inherit PLN details
#'
#' @rdname PLNfit
#' @include PLNfit-class.R
#' @importFrom R6 R6Class
#'
#' @examples
#' \dontrun{
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLN(Abundance ~ 1, data = trichoptera)
#' class(myPLN)
#' print(myPLN)
#' }
PLNfit <- R6Class(
  classname = "PLNfit",
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## PRIVATE MEMBERS ----
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  private = list(
    ## PRIVATE INTERNAL FIELDS
    formula    = NA    , # the formula call for the model as specified by the user
    B          = NA    , # regression parameters of the latent layer
    Sigma      = NA    , # covariance matrix of the latent layer
    Omega      = NA    , # precision matrix of the latent layer. Inverse of Sigma
    S          = NA    , # variational parameters for the variances
    M          = NA    , # variational parameters for the means
    Z          = NA    , # matrix of latent variable
    A          = NA    , # matrix of expected counts (under variational approximation)
    Ji         = NA    , # element-wise approximated loglikelihood
    R2         = NA    , # approximated goodness of fit criterion
    optimizer  = list(), # list of links to the functions doing the optimization
    monitoring = list(), # list with optimization monitoring quantities
    B.new      = NA    , # regression parameters of the latent layer
    Sigma.new  = NA    , # covariance matrix of the latent layer
    Omega.new  = NA    , # precision matrix of the latent layer. Inverse of Sigma
    S.new      = NA    , # variational parameters for the variances
    M.new      = NA    , # variational parameters for the means
    Z.new      = NA    , # matrix of latent variable
    A.new      = NA    , # matrix of expected counts (under variational approximation)
    Ji.new     = NA    , # element-wise approximated loglikelihood

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## PRIVATE TORCH METHODS FOR OPTIMIZATION
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    torch_elbo = function(data, params, index=torch_tensor(1:self$n)) {
      S2 <- torch_square(params$S[index])
      Z  <- data$O[index] + params$M[index] + torch_mm(data$X[index], params$B)
      res <- .5 * sum(data$w[index]) * torch_logdet(private$torch_Sigma(data, params, index)) +
        sum(data$w[index,NULL] * (torch_exp(Z + .5 * S2) - data$Y[index] * Z - .5 * torch_log(S2)))
      res
    },

    torch_Sigma = function(data, params, index=torch_tensor(1:self$n)) {
      ws     <- torch_sqrt(data$w[index, NULL])
      S2_bar <- torch_sum(torch_square(ws * params$S[index]), 1)
      MtM    <- torch_mm(torch_t(ws * params$M[index]), ws * params$M[index])
      (MtM + torch_diag(S2_bar)) / sum(ws*ws)
    },

    torch_Omega = function(data, params) {
      torch::torch_inverse(params$Sigma)
    },

    torch_vloglik = function(data, params) {
      S2    <- torch_square(params$S)
      Ji <- .5 * self$p - rowSums(.logfactorial(as.matrix(data$Y))) + as.numeric(
        .5 * torch_logdet(params$Omega) +
          torch_sum(data$Y * params$Z - params$A + .5 * torch_log(S2), dim = 2) -
          .5 * torch_sum(torch_mm(params$M, params$Omega) * params$M + S2 * torch_diag(params$Omega), dim = 2)
      )
      attr(Ji, "weights") <- as.numeric(data$w)
      Ji
    },

    #' @import torch
    torch_optimize = function(data, params, config) {

      ## Conversion of data and parameters to torch tensors (pointers)
      data   <- lapply(data, torch_tensor)                         # list with Y, X, O, w
      params <- lapply(params, torch_tensor, requires_grad = TRUE) # list with B, M, S

      ## Initialize optimizer
      optimizer <- switch(config$algorithm,
          "RPROP"   = optim_rprop(params  , lr = config$lr, etas = config$etas, step_sizes = config$step_sizes),
          "RMSPROP" = optim_rmsprop(params, lr = config$lr, weight_decay = config$weight_decay, momentum = config$momentum, centered = config$centered),
          "ADAM"    = optim_adam(params   , lr = config$lr, weight_decay = config$weight_decay),
          "ADAGRAD" = optim_adagrad(params, lr = config$lr, weight_decay = config$weight_decay)
      )

      ## Optimization loop
      status <- 5
      num_epoch  <- config$num_epoch
      num_batch  <- config$num_batch
      batch_size <- floor(self$n/num_batch)

      objective <- double(length = config$num_epoch + 1)
      for (iterate in 1:num_epoch) {
        B_old <- as.numeric(optimizer$param_groups[[1]]$params$B)

        # rearrange the data each epoch
        permute <- torch::torch_randperm(self$n) + 1L
        for (batch_idx in 1:num_batch) {
          # here index is a vector of the indices in the batch
          index <- permute[(batch_size*(batch_idx - 1) + 1):(batch_idx*batch_size)]

          ## Optimization
          optimizer$zero_grad() # reinitialize gradients
          loss <- private$torch_elbo(data, params, index) # compute current ELBO
          loss$backward()                   # backward propagation
          optimizer$step()                  # optimization
        }

        ## assess convergence
        objective[iterate + 1] <- loss$item()
        B_new <- as.numeric(optimizer$param_groups[[1]]$params$B)
        delta_f   <- abs(objective[iterate] - objective[iterate + 1]) / abs(objective[iterate + 1])
        delta_x   <- sum(abs(B_old - B_new))/sum(abs(B_new))

        ## display progress
        if (config$trace >  1 && (iterate %% 50 == 0))
          cat('\niteration: ', iterate, 'objective', objective[iterate + 1],
              'delta_f'  , round(delta_f, 6), 'delta_x', ro<und(delta_x, 6))

        ## Check for convergence
        if (delta_f < config$ftol_rel) status <- 3
        if (delta_x < config$xtol_rel) status <- 4
        if (status %in% c(3,4)) {
          objective <- objective[1:iterate + 1]
          break
        }
      }

      params$Sigma <- private$torch_Sigma(data, params)
      params$Omega <- private$torch_Omega(data, params)
      params$Z     <- data$O + params$M + torch_matmul(data$X, params$B)
      params$A     <- torch_exp(params$Z + torch_pow(params$S, 2)/2)

      out <- lapply(params, as.matrix)
      out$Ji <- private$torch_vloglik(data, params)
      out$monitoring <- list(
          objective  = objective,
          iterations = iterate,
          status     = status,
          backend = "torch"
        )
      out
    },


    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## PRIVATE METHODS FOR VARIANCE OF THE ESTIMATORS
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    variance_variational = function(X) {
    ## Variance of B for n data points
      fisher <- Matrix::bdiag(lapply(1:self$p, function(j) {
        crossprod(X, private$A[, j] * X) # t(X) %*% diag(A[, i]) %*% X
      }))
      vcov_B <- tryCatch(Matrix::solve(fisher), error = function(e) {e})
      if (is(vcov_B, "error")) {
        warning(paste("Inversion of the Fisher information matrix failed with following error message:",
                      vcov_B$message, "Returning NA", sep = "\n"))
        vcov_B <- matrix(NA, nrow = self$d, ncol = self$p)
        var_B  <- matrix(NA, nrow = self$d, ncol = self$p)
      } else {
        var_B <- vcov_B %>% diag() %>% matrix(nrow = self$d)
      }
      rownames(vcov_B) <- colnames(vcov_B) <-
        expand.grid(covariates = rownames(private$B),
                    responses  = colnames(private$B)) %>% rev() %>%
        ## Hack to make sure that species is first and varies slowest
        apply(1, paste0, collapse = "_")
      attr(private$B, "vcov_variational") <- vcov_B
      dimnames(var_B) <- dimnames(private$B)
      attr(private$B, "variance_variational") <- var_B

      ## Variance of Omega
      var_Omega <- 2 * outer(diag(private$Omega), diag(private$Omega)) / self$n
      dimnames(var_Omega) <- dimnames(private$Omega)
      attr(private$Omega, "variance_variational") <- var_Omega
      invisible(list(var_B = var_B, var_Omega = var_Omega))
    },

    variance_jackknife = function(Y, X, O, w, config = config_default_nlopt) {
      jacks <- future.apply::future_lapply(seq_len(self$n), function(i) {
        data <- list(Y = Y[-i, , drop = FALSE],
                     X = X[-i, , drop = FALSE],
                     O = O[-i, , drop = FALSE],
                     w = w[-i])
        args <- list(data = data,
                     params = list(B = private$B, M = matrix(0, self$n-1, self$p), S = private$S[-i, ]),
                     config = config)
        optim_out <- do.call(private$optimizer$main, args)
        optim_out[c("B", "Omega")]
      }, future.seed = TRUE)

      B_jack <- jacks %>% map("B") %>% reduce(`+`) / self$n
      var_jack   <- jacks %>% map("B") %>% map(~( (. - B_jack)^2)) %>% reduce(`+`) %>%
        `dimnames<-`(dimnames(private$B))
      B_hat  <- private$B[,] ## strips attributes while preserving names
      attr(private$B, "bias") <- (self$n - 1) * (B_jack - B_hat)
      attr(private$B, "variance_jackknife") <- (self$n - 1) / self$n * var_jack

      Omega_jack <- jacks %>% map("Omega") %>% reduce(`+`) / self$n
      var_jack   <- jacks %>% map("Omega") %>% map(~( (. - Omega_jack)^2)) %>% reduce(`+`) %>%
        `dimnames<-`(dimnames(private$Omega))
      Omega_hat  <- private$Omega[,] ## strips attributes while preserving names
      attr(private$Omega, "bias") <- (self$n - 1) * (Omega_jack - Omega_hat)
      attr(private$Omega, "variance_jackknife") <- (self$n - 1) / self$n * var_jack
    },

    variance_bootstrap = function(Y, X, O, w, n_resamples = 100, config = config_default_nlopt) {
      resamples <- replicate(n_resamples, sample.int(self$n, replace = TRUE), simplify = FALSE)
      boots <- future.apply::future_lapply(resamples, function(resample) {
        data <- list(Y = Y[resample, , drop = FALSE],
                     X = X[resample, , drop = FALSE],
                     O = O[resample, , drop = FALSE],
                     w = w[resample])
        args <- list(data = data,
                     params = list(B = private$B, M = matrix(0,self$n,self$p), S = private$S[resample, ]),
                     config = config)
        optim_out <- do.call(private$optimizer$main, args)
        optim_out[c("B", "Omega", "monitoring")]
      }, future.seed = TRUE)

      B_boots <- boots %>% map("B") %>% reduce(`+`) / n_resamples
      attr(private$B, "variance_bootstrap") <-
        boots %>% map("B") %>% map(~( (. - B_boots)^2)) %>% reduce(`+`)  %>%
          `dimnames<-`(dimnames(private$B)) / n_resamples

      Omega_boots <- boots %>% map("Omega") %>% reduce(`+`) / n_resamples
      attr(private$Omega, "variance_bootstrap") <-
        boots %>% map("Omega") %>% map(~( (. - Omega_boots)^2)) %>% reduce(`+`)  %>%
        `dimnames<-`(dimnames(private$Omega)) / n_resamples
    },

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## PRIVATE METHOD FOR DEVIANCE/R2
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    approx_r2 = function(responses, covariates, offsets, weights, nullModel = NULL) {
      if (is.null(nullModel)) nullModel <- nullModelPoisson(responses, covariates, offsets, weights)
      loglik <- logLikPoisson(responses, self$latent, weights)
      lmin   <- logLikPoisson(responses, nullModel, weights)
      lmax   <- logLikPoisson(responses, log(responses), weights)
      private$R2 <- (loglik - lmin) / (lmax - lmin)
    }

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## END OF PRIVATE METHODS
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  ),
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## PUBLIC MEMBERS
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  public = list(

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## CONSTRUCTOR
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #' @description Initialize a [`PLNfit`] model
    #' @importFrom stats lm.wfit lm.fit poisson residuals coefficients runif
    initialize = function(responses, covariates, offsets, weights, formula, control) {
      ## problem dimensions
      n <- nrow(responses); p <- ncol(responses); d <- ncol(covariates)
      ## set up various quantities
      private$formula <- formula # user formula call
      ## initialize the variational parameters
      if (isPLNfit(control$inception)) {
        if (control$trace > 1) cat("\n User defined inceptive PLN model")
        stopifnot(isTRUE(all.equal(dim(control$inception$model_par$B), c(d,p))))
        stopifnot(isTRUE(all.equal(dim(control$inception$var_par$M)  , c(n,p))))
        private$Sigma <- control$inception$model_par$Sigma
        private$B     <- control$inception$model_par$B
        private$M     <- control$inception$var_par$M
        private$S     <- control$inception$var_par$S
      } else {
        if (control$trace > 1) cat("\n Use LM after log transformation to define the inceptive model")
        fits <- lm.fit(weights * covariates, weights * log((1 + responses)/(1 + exp(offsets))))
        private$B <- matrix(fits$coefficients, d, p)
        private$M <- matrix(fits$residuals, n, p)
        private$S <- matrix(1, n, p)
      }
      private$optimizer$main   <- ifelse(control$backend == "nlopt", nlopt_optimize, private$torch_optimize)
      private$optimizer$vestep <- nlopt_optimize_vestep
    },

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## SETTER METHOD
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #' @description
    #' Update a [`PLNfit`] object
    #' @param M     matrix of variational parameters for the mean
    #' @param S     matrix of variational parameters for the variance
    #' @param Ji    vector of variational lower bounds of the log-likelihoods (one value per sample)
    #' @param R2    approximate R^2 goodness-of-fit criterion
    #' @param Z     matrix of latent vectors (includes covariates and offset effects)
    #' @param A     matrix of fitted values
    #' @param monitoring a list with optimization monitoring quantities
    #' @return Update the current [`PLNfit`] object
    update = function(B=NA, Sigma=NA, Omega=NA, M=NA, S=NA, Ji=NA, R2=NA, Z=NA, A=NA, monitoring=NA) {
      if (!anyNA(B))      private$B  <- B
      if (!anyNA(Sigma))      private$Sigma  <- Sigma
      if (!anyNA(Omega))      private$Omega  <- Omega
      if (!anyNA(M))          private$M      <- M
      if (!anyNA(S))          private$S      <- S
      if (!anyNA(Z))          private$Z      <- Z
      if (!anyNA(A))          private$A      <- A
      if (!anyNA(Ji))         private$Ji     <- Ji
      if (!anyNA(R2))         private$R2     <- R2
      if (!anyNA(monitoring)) private$monitoring <- monitoring
    },

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## GENERIC OPTIMIZER
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #' @description Call to the NLopt or TORCH optimizer and update of the relevant fields
    optimize = function(responses, covariates, offsets, weights, config) {
      args <- list(data   = list(Y = responses, X = covariates, O = offsets, w = weights),
                   params = list(B = private$B, M = private$M, S = private$S),
                   config = config)
      optim_out <- do.call(private$optimizer$main, args)
      do.call(self$update, optim_out)
    },

    #' @description Result of one call to the VE step of the optimization procedure: optimal variational parameters (M, S) and corresponding log likelihood values for fixed model parameters (Sigma, B). Intended to position new data in the latent space.
    #' @param B Optional fixed value of the regression parameters
    #' @param Sigma variance-covariance matrix of the latent variables
    #' @return A list with three components:
    #'  * the matrix `M` of variational means,
    #'  * the matrix `S2` of variational variances
    #'  * the vector `log.lik` of (variational) log-likelihood of each new observation
    optimize_vestep = function(covariates, offsets, responses, weights,
                      B = self$model_par$B,
                      Omega = self$model_par$Omega,
                      control = PLN_param(backend = "nlopt")) {
      n <- nrow(responses); p <- ncol(responses)
      args <- list(data   = list(Y = responses, X = covariates, O = offsets, w = weights),
                   ## Initialize the variational parameters with the new dimension of the data
                   params = list(M = matrix(0, n, p), S = matrix(1, n, p)),
                   B = as.matrix(B),
                   Omega = as.matrix(Omega),
                   config = control$config_optim)
      optim_out <- do.call(private$optimizer$vestep, args)
      optim_out
    },

    #' @description Update R2, fisher and std_err fields after optimization
    #' @param config a list for controlling the post-treatments (optional bootstrap, jackknife, R2, etc.). See details
    #' @details The list of parameters `config` controls the post-treatment processing, with the following entries:
    #' * jackknife boolean indicating whether jackknife should be performed to evaluate bias and variance of the model parameters. Default is FALSE.
    #' * bootstrap integer indicating the number of bootstrap resamples generated to evaluate the variance of the model parameters. Default is 0 (inactivated).
    #' * variational_var boolean indicating whether variational Fisher information matrix should be computed to estimate the variance of the model parameters (highly underestimated). Default is FALSE.
    #' * rsquared boolean indicating whether approximation of R2 based on deviance should be computed. Default is TRUE
    #' * trace integer for verbosity. should be > 1 to see output in post-treatments
    postTreatment = function(responses, covariates, offsets, weights = rep(1, nrow(responses)), config, nullModel = NULL) {
      ## PARAMATERS DIMNAMES
      ## Set names according to those of the data matrices. If missing, use sensible defaults
      if (is.null(colnames(responses)))
        colnames(responses) <- paste0("Y", 1:self$p)
      if (self$d > 0) {
        if (is.null(colnames(covariates))) colnames(covariates) <- paste0("X", 1:self$d)
        #colnames(private$B) <- colnames(responses)
        rownames(private$B) <- colnames(covariates)
      }
      rownames(private$Sigma) <- colnames(private$Sigma) <- colnames(responses)
      rownames(private$Omega) <- colnames(private$Omega) <- colnames(responses)
      rownames(private$M) <- rownames(private$S) <- rownames(responses)
      colnames(private$S) <- 1:self$q

      ## OPTIONAL POST-TREATMENT (potentially costly)
      ## 1. compute and store approximated R2 with Poisson-based deviance
      if (config$rsquared) {
        if(config$trace > 1) cat("\n\tComputing bootstrap estimator of the variance...")
        private$approx_r2(responses, covariates, offsets, weights, nullModel)
      }
      ## 2. compute and store matrix of standard variances for B and Omega with rough variational approximation
      if (config$variational_var) {
        if(config$trace > 1) cat("\n\tComputing variational estimator of the variance...")
        private$variance_variational(covariates)
      }
      ## 3. Jackknife estimation of bias and variance
      if (config$jackknife) {
        if(config$trace > 1) cat("\n\tComputing jackknife estimator of the variance...")
        private$variance_jackknife(responses, covariates, offsets, weights)
      }
      ## 4. Bootstrap estimation of variance
      if (config$bootstrap > 0) {
        if(config$trace > 1) cat("\n\tComputing bootstrap estimator of the variance...")
        private$variance_bootstrap(responses, covariates, offsets, weights, config$bootstrap)
      }
    },

    #' @description Predict position, scores or observations of new data.
    #' @param newdata A data frame in which to look for variables with which to predict. If omitted, the fitted values are used.
    #' @param type Scale used for the prediction. Either `link` (default, predicted positions in the latent space) or `response` (predicted counts).
    #' @param envir Environment in which the prediction is evaluated
    #' @return A matrix with predictions scores or counts.
    predict = function(newdata, type = c("link", "response"), envir = parent.frame()) {

      ## Extract the model matrices from the new data set with initial formula
      X <- model.matrix(formula(private$formula)[-2], newdata, xlev = attr(private$formula, "xlevels"))
      O <- model.offset(model.frame(formula(private$formula)[-2], newdata))

      ## mean latent positions in the parameter space
      EZ <- X %*% private$B
      if (!is.null(O)) EZ <- EZ + O
      EZ <- sweep(EZ, 2, .5 * diag(self$model_par$Sigma), "+")
      colnames(EZ) <- colnames(private$Sigma)

      type <- match.arg(type)
      results <- switch(type, link = EZ, response = exp(EZ))
      attr(results, "type") <- type
      results
    },

    #' @description Predict position, scores or observations of new data, conditionally on the observation of a (set of) variables
    #' @param cond_responses a data frame containing the count of the observed variables (matching the names of the provided as data in the PLN function)
    #' @param newdata a data frame containing the covariates of the sites where to predict
    #' @param type Scale used for the prediction. Either `link` (default, predicted positions in the latent space) or `response` (predicted counts).
    #' @param var_par Boolean. Should new estimations of the variational parameters of mean and variance be sent back, as attributes of the matrix of predictions. Default to \code{FALSE}.
    #' @param envir Environment in which the prediction is evaluated
    #' @return A matrix with predictions scores or counts.
    predict_cond = function(newdata, cond_responses, type = c("link", "response"), var_par = FALSE, envir = parent.frame()){
      type <- match.arg(type)

      # Checks
      Yc <- as.matrix(cond_responses)
      sp_names <- colnames(self$model_par$B)
      if (!any(colnames(cond_responses) %in% sp_names))
        stop("Yc must be a subset of the species in responses")
      if (!nrow(Yc) == nrow(newdata))
        stop("The number of rows of Yc must match the number of rows in newdata")

      # Dimensions and subsets
      n_new <- nrow(Yc)
      cond <- sp_names %in% colnames(Yc)

      ## Extract the model matrices from the new data set with initial formula
      X <- model.matrix(formula(private$formula)[-2], newdata, xlev = attr(private$formula, "xlevels"))
      O <- model.offset(model.frame(formula(private$formula)[-2], newdata))
      if (is.null(O)) O <- matrix(0, n_new, self$p)

      # Compute parameters of the law
      vcov11 <- private$Sigma[cond ,  cond, drop = FALSE]
      vcov22 <- private$Sigma[!cond, !cond, drop = FALSE]
      vcov12 <- private$Sigma[cond , !cond, drop = FALSE]
      prec11 <- solve(vcov11)
      A <- crossprod(vcov12, prec11)
      Sigma21 <- vcov22 - A %*% vcov12

      # Call to VEstep to obtain M1, S1
      VE <- self$optimize_vestep(
              covariates = X,
              offsets    = O[, cond, drop = FALSE],
              responses  = Yc,
              weights    = rep(1, n_new),
              B          = self$model_par$B[, cond, drop = FALSE],
              Omega      = prec11
          )

      M <- tcrossprod(VE$M, A)
      # S <- map(1:n_new, ~crossprod(sqrt(VE$S[., ]) * t(A)) + Sigma21) %>%
      #   simplify2array()
      S <- map(1:n_new, ~crossprod(VE$S[., ] * t(A)) + Sigma21) %>% simplify2array()

      ## mean latent positions in the parameter space
      EZ <- X %*% private$B[, !cond, drop = FALSE] + M + O[, !cond, drop = FALSE]
      colnames(EZ) <- setdiff(sp_names, colnames(Yc))

      # ! We should only add the .5*diag(S2) term only if we want the type="response"
      if (type == "response") {
        if (ncol(EZ) == 1) {
          EZ <- EZ + .5 * S
        } else {
          EZ <- EZ + .5 * t(apply(S, 3, diag))
        }
      }
      results <- switch(type, link = EZ, response = exp(EZ))
      attr(results, "type") <- type
      if (var_par) {
        attr(results, "M") <- M
        attr(results, "S") <- S
      }
      results
    },

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## Print functions -----------------------
    #' @description User friendly print method
    #' @param model First line of the print output
    show = function(model = paste("A multivariate Poisson Lognormal fit with", self$vcov_model, "covariance model.\n")) {
      cat(model)
      cat("==================================================================\n")
      print(as.data.frame(round(self$criteria, digits = 3), row.names = ""))
      cat("==================================================================\n")
      cat("* Useful fields\n")
      cat("    $model_par, $latent, $latent_pos, $var_par, $optim_par\n")
      cat("    $loglik, $BIC, $ICL, $loglik_vec, $nb_param, $criteria\n")
      cat("* Useful S3 methods\n")
      cat("    print(), coef(), sigma(), vcov(), fitted()\n")
      cat("    predict(), predict_cond(), standard_error()\n")
    },

    #' @description User friendly print method
    print = function() { self$show() }

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## Other functions ----------------
  ),

  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ##  ACTIVE BINDINGS ----
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  active = list(
    #' @field n number of samples
    n = function() {nrow(private$M)},
    #' @field q number of dimensions of the latent space
    q = function() {ncol(private$M)},
    #' @field p number of species
    p = function() {ncol(private$B)},
    #' @field d number of covariates
    d = function() {nrow(private$B)},
    #' @field nb_param number of parameters in the current PLN model
    nb_param   = function() {as.integer(self$p * self$d + self$p * (self$p + 1)/2)},
    #' @field model_par a list with the matrices of the model parameters: B (covariates), Sigma (covariance), Omega (precision matrix), plus some others depending on the variant)
    model_par  = function() {list(B = private$B, Sigma = private$Sigma, Omega = private$Omega, Theta = t(private$B), L = private$L, V = private$V,
                                  B.new = private$B.new, Sigma.new = private$Sigma.new, Omega.new = private$Omega.new, Theta.new = t(private$B.new))},
    #' @field var_par a list with the matrices of the variational parameters: M (means) and S2 (variances)
    var_par    = function() {list(M = private$M, S2 = private$S**2, S = private$S,
                                  M.new = private$M.new, S2.new = private$S.new**2, S.new = private$S.new)},
    #' @field optim_par a list with parameters useful for monitoring the optimization
    optim_par  = function() {c(private$monitoring, backend = private$backend)},
    #' @field latent a matrix: values of the latent vector (Z in the model)
    latent     = function() {private$Z},
    #' @field latent_pos a matrix: values of the latent position vector (Z) without covariates effects or offset
    latent_pos = function() {private$M},
    #' @field fitted a matrix: fitted values of the observations (A in the model)
    fitted     = function() {private$A},
    #' @field vcov_coef matrix of sandwich estimator of the variance-covariance of B (need fixed -ie known- covariance at the moment)
    vcov_coef = function() {attr(private$B, "vcov_variational")},
    #' @field vcov_model character: the model used for the residual covariance
    vcov_model = function() {"full"},
    #' @field weights observational weights
    weights     = function() {as.numeric(attr(private$Ji, "weights"))},
    #' @field loglik (weighted) variational lower bound of the loglikelihood
    loglik     = function() {sum(self$weights[self$weights > .Machine$double.eps] * private$Ji[self$weights > .Machine$double.eps]) },
    #' @field loglik_vec element-wise variational lower bound of the loglikelihood
    loglik_vec = function() {private$Ji},
    #' @field BIC variational lower bound of the BIC
    BIC        = function() {self$loglik - .5 * log(self$n) * self$nb_param},
    #' @field entropy Entropy of the variational distribution
    entropy    = function() {.5 * (self$n * self$q * log(2*pi*exp(1)) + sum(log(self$var_par$S2)))},
    #' @field ICL variational lower bound of the ICL
    ICL        = function() {self$BIC - self$entropy},
    #' @field R_squared approximated goodness-of-fit criterion
    R_squared  = function() {private$R2},
    #' @field criteria a vector with loglik, BIC, ICL and number of parameters
    criteria   = function() {data.frame(nb_param = self$nb_param, loglik = self$loglik, BIC = self$BIC, ICL = self$ICL)}
  )
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ##  END OF THE CLASS PLNfit
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  CLASS PLNfit_diagonal
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' An R6 Class to represent a PLNfit in a standard, general framework, with diagonal residual covariance
#'
#' @param responses the matrix of responses (called Y in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param covariates design matrix (called X in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param offsets offset matrix (called O in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param weights an optional vector of observation weights to be used in the fitting process.
#' @param formula model formula used for fitting, extracted from the formula in the upper-level call
#' @param control a list for controlling the optimization. See details.
#'
#' @rdname PLNfit_diagonal
#' @importFrom R6 R6Class
#'
#' @examples
#' \dontrun{
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLN(Abundance ~ 1, data = trichoptera)
#' class(myPLN)
#' print(myPLN)
#' }
PLNfit_diagonal <- R6Class(
  classname = "PLNfit_diagonal",
  inherit = PLNfit,
  public  = list(
    #' @description Initialize a [`PLNfit`] model
    initialize = function(responses, covariates, offsets, weights, formula, control) {
      super$initialize(responses, covariates, offsets, weights, formula, control)
      private$optimizer$main   <- ifelse(control$backend == "nlopt", nlopt_optimize_diagonal, private$torch_optimize)
      private$optimizer$vestep <- nlopt_optimize_vestep_diagonal
    }
  ),
  private = list(
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## PRIVATE TORCH METHODS FOR OPTIMIZATION
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    torch_elbo = function(data, params, index=torch_tensor(1:self$n)) {
      S2 <- torch_square(params$S[index])
      Z <- data$O[index] + params$M[index] + torch_matmul(data$X[index], params$B)
      res <- .5 * sum(data$w[index]) * sum(torch_log(private$torch_sigma_diag(data, params, index))) +
        sum(data$w[index,NULL] * (torch_exp(Z + .5 * S2) - data$Y[index] * Z -  .5 * torch_log(S2)))
      res
    },

    torch_sigma_diag = function(data, params, index=torch_tensor(1:self$n)) {
      torch_sum(data$w[index,NULL] * (torch_square(params$M[index]) + torch_square(params$S[index])), 1) / sum(data$w[index])
    },

    torch_Sigma = function(data, params, index=torch_tensor(1:self$n)) {
      torch_diag(private$torch_sigma_diag(data, params, index))
    },

    torch_vloglik = function(data, params) {
      S2 <- torch_square(params$S)
      omega_diag <- torch_pow(private$torch_sigma_diag(data, params), -1)
      Ji <- .5 * self$p - rowSums(.logfactorial(as.matrix(data$Y))) + as.numeric(
        .5 * sum(torch_log(omega_diag)) +
          torch_sum(data$Y * params$Z - params$A + .5 * torch_log(S2) -
                      .5 * (torch_square(params$M) + S2) * omega_diag[NULL,], dim = 2)
      )
      attr(Ji, "weights") <- as.numeric(data$w)
      Ji
    }
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## END OF TORCH METHODS
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ),
  active = list(
    #' @field nb_param number of parameters in the current PLN model
    nb_param   = function() {as.integer(self$p * self$d + self$p)},
    #' @field vcov_model character: the model used for the residual covariance
    vcov_model = function() {"diagonal"}
  )
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ##  END OF THE CLASS PLNfit_diagonal
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  CLASS PLNfit_spherical
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' An R6 Class to represent a PLNfit in a standard, general framework, with spherical residual covariance
#'
#' @param responses the matrix of responses (called Y in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param covariates design matrix (called X in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param offsets offset matrix (called O in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param weights an optional vector of observation weights to be used in the fitting process.
#' @param formula model formula used for fitting, extracted from the formula in the upper-level call
#' @param control a list for controlling the optimization. See details.
#'
#' @rdname PLNfit_spherical
#' @importFrom R6 R6Class
#'
#' @examples
#' \dontrun{
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLN(Abundance ~ 1, data = trichoptera)
#' class(myPLN)
#' print(myPLN)
#' }
PLNfit_spherical <- R6Class(
  classname = "PLNfit_spherical",
  inherit = PLNfit,
  public  = list(
    #' @description Initialize a [`PLNfit`] model
    initialize = function(responses, covariates, offsets, weights, formula, control) {
      super$initialize(responses, covariates, offsets, weights, formula, control)
      private$optimizer$main   <- ifelse(control$backend == "nlopt", nlopt_optimize_spherical, private$torch_optimize)
      private$optimizer$vestep <- nlopt_optimize_vestep_diagonal
    }
  ),
  private = list(

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## PRIVATE TORCH METHODS FOR OPTIMIZATION
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    torch_elbo = function(data, params, index=torch_tensor(1:self$n)) {
      S2 <- torch_square(params$S[index])
      Z <- data$O[index] + params$M[index] + torch_mm(data$X[index], params$B)
      res <- .5 * sum(data$w[index]) * self$p * torch_log(private$torch_sigma2(data, params, index)) -
        sum(data$w[index,NULL] * (data$Y[index] * Z - torch_exp(Z + .5 * S2) + .5 * torch_log(S2)))
      res
    },

    torch_sigma2 = function(data, params, index=torch_tensor(1:self$n)) {
      sum(data$w[index, NULL] * (torch_square(params$M) + torch_square(params$S))) / (sum(data$w) * self$p)
    },

    torch_Sigma = function(data, params, index=torch_tensor(1:self$n)) {
      torch_eye(self$p) * private$torch_sigma2(data, params, index)
    },

    torch_vloglik = function(data, params) {
      S2 <- torch_pow(params$S, 2)
      sigma2 <- private$torch_sigma2(data, params)
      Ji <- .5 * self$p - rowSums(.logfactorial(as.matrix(data$Y))) + as.numeric(
        torch_sum(data$Y * params$Z - params$A + .5 * torch_log(S2/sigma2) - .5 * (torch_pow(params$M, 2) + S2)/sigma2, dim = 2)
      )
      attr(Ji, "weights") <- as.numeric(data$w)
      Ji
    }
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## END OF TORCH METHODS FOR OPTIMIZATION
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ),
  active = list(
    #' @field nb_param number of parameters in the current PLN model
    nb_param   = function() {as.integer(self$p * self$d + 1)},
    #' @field vcov_model character: the model used for the residual covariance
    vcov_model = function() {"spherical"}
  )
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ##  END OF THE CLASS PLNfit_spherical
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  CLASS PLNfit_fixedcov
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' An R6 Class to represent a PLNfit in a standard, general framework, with fixed (inverse) residual covariance
#'
#' @param responses the matrix of responses (called Y in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param covariates design matrix (called X in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param offsets offset matrix (called O in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param data an optional data frame, list or environment (or object coercible by as.data.frame to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which PLN is called.
#' @param weights an optional vector of observation weights to be used in the fitting process.
#' @param nullModel null model used for approximate R2 computations. Defaults to a GLM model with same design matrix but not latent variable.
#' @param formula model formula used for fitting, extracted from the formula in the upper-level call
#' @param control a list for controlling the optimization. See details.
#' @param config part of the \code{control} argument which configures the optimizer
#'
#' @rdname PLNfit_fixedcov
#' @importFrom R6 R6Class
#'
#' @examples
#' \dontrun{
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLN(Abundance ~ 1, data = trichoptera)
#' class(myPLN)
#' print(myPLN)
#' }
PLNfit_fixedcov <- R6Class(
  classname = "PLNfit_fixedcov",
  inherit = PLNfit,
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## PUBLIC MEMBERS ----
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  public  = list(
    #' @description Initialize a [`PLNfit`] model
    initialize = function(responses, covariates, offsets, weights, formula, control) {
      super$initialize(responses, covariates, offsets, weights, formula, control)
      private$optimizer$main <- ifelse(control$backend == "nlopt", nlopt_optimize_fixed, private$torch_optimize)
      ## ve step is the same as in the fullly parameterized covariance
      private$Omega <- control$Omega
    },
    #' @description Call to the NLopt or TORCH optimizer and update of the relevant fields
    optimize = function(responses, covariates, offsets, weights, config) {
      args <- list(data   = list(Y = responses, X = covariates, O = offsets, w = weights),
                   params = list(B = private$B, M = private$M, S = private$S, Omega = private$Omega),
                   config = config)
      optim_out <- do.call(private$optimizer$main, args)
      do.call(self$update, optim_out)
      private$Sigma <- solve(optim_out$Omega)
    },

    #' @description Update R2, fisher and std_err fields after optimization
    #' @param config a list for controlling the post-treatments (optional bootstrap, jackknife, R2, etc.). See details
    #' @details The list of parameters `config` controls the post-treatment processing, with the following entries:
    #' * trace integer for verbosity. should be > 1 to see output in post-treatments
    #' * jackknife boolean indicating whether jackknife should be performed to evaluate bias and variance of the model parameters. Default is FALSE.
    #' * bootstrap integer indicating the number of bootstrap resamples generated to evaluate the variance of the model parameters. Default is 0 (inactivated).
    #' * variational_var boolean indicating whether variational Fisher information matrix should be computed to estimate the variance of the model parameters (highly underestimated). Default is FALSE.
    #' * rsquared boolean indicating whether approximation of R2 based on deviance should be computed. Default is TRUE
    postTreatment = function(responses, covariates, offsets, weights = rep(1, nrow(responses)), config, nullModel = NULL) {
      super$postTreatment(responses, covariates, offsets, weights, config, nullModel)
      ## 6. compute and store matrix of standard variances for B with sandwich correction approximation
      if (config$sandwich_var) {
        if(config$trace > 1) cat("\n\tComputing sandwich estimator of the variance...")
        private$vcov_sandwich_B(responses, covariates)
      }
    }
  ),
  private = list(
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## PRIVATE TORCH METHODS FOR OPTIMIZATION
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    torch_elbo = function(data, params, index=torch_tensor(1:self$n)) {
      S2 <- torch_square(params$S[index])
      Z <- data$O[index] + params$M[index] + torch_mm(data$X[index], params$B)
      res <- sum(data$w) * torch_trace(torch_mm(private$torch_Sigma(data, params, index), private$torch_Omega(data, params))) +
        sum(data$w[index,NULL] * (torch_exp(Z + .5 * S2) - data$Y[index] * Z - .5 * torch_log(S2)))
      res
    },

    torch_Omega = function(data, params) {
      params$Omega <- torch_tensor(private$Omega)
    },

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## END OF TORCH METHODS FOR OPTIMIZATION
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## PRIVATE METHODS FOR VARIANCE OF THE ESTIMATORS
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    variance_jackknife = function(Y, X, O, w, config = config_default_nlopt) {
      jacks <- future.apply::future_lapply(seq_len(self$n), function(i) {
        args <- list(Y = Y[-i, , drop = FALSE],
                     X = X[-i, , drop = FALSE],
                     O = O[-i, , drop = FALSE],
                     w = w[-i],
                     params = list(B = private$B, Omega = private$Omega, M = private$M[-i, ], S = private$S[-i, ]),
                     config = config)
        optim_out <- do.call(private$optimizer$main, args)
        optim_out[c("B", "Omega")]
      }, future.seed = TRUE)

      B_jack <- jacks %>% map("B") %>% reduce(`+`) / self$n
      var_jack   <- jacks %>% map("B") %>% map(~( (. - B_jack)^2)) %>% reduce(`+`) %>%
        `dimnames<-`(dimnames(private$B))
      B_hat  <- private$B[,] ## strips attributes while preserving names
      attr(private$B, "bias") <- (self$n - 1) * (B_jack - B_hat)
      attr(private$B, "variance_jackknife") <- (self$n - 1) / self$n * var_jack
    },

    vcov_sandwich_B = function(Y, X) {
      getMat_iCnB <- function(i) {
        a_i   <- as.numeric(private$A[i, ])
        s2_i  <- as.numeric(private$S[i, ]**2)
        # omega <- as.numeric(1/diag(private$Sigma))
        # diag_mat_i <- diag(1/a_i + s2_i^2 / (1 + s2_i * (a_i + omega)))
        diag_mat_i <- diag(1/a_i + .5 * s2_i^2)
        solve(private$Sigma + diag_mat_i)
      }
      YmA <- Y - private$A
      Dn <- matrix(0, self$d*self$p, self$d*self$p)
      Cn <- matrix(0, self$d*self$p, self$d*self$p)
      for (i in 1:self$n) {
        xxt_i <- tcrossprod(X[i, ])
        Cn <- Cn - kronecker(getMat_iCnB(i) , xxt_i) / (self$n)
        Dn <- Dn + kronecker(tcrossprod(YmA[i,]), xxt_i) / (self$n)
      }
      Cn_inv <- solve(Cn)
      dim_names <- dimnames(attr(private$B, "vcov_variational"))
      vcov_sand <- ((Cn_inv %*% Dn %*% Cn_inv) / self$n) %>% `dimnames<-`(dim_names)
      attr(private$B, "vcov_sandwich") <- vcov_sand
      attr(private$B, "variance_sandwich") <- matrix(diag(vcov_sand), nrow = self$d, ncol = self$p,
                                                         dimnames = dimnames(private$B))
    }
  ),
  active = list(
    #' @field nb_param number of parameters in the current PLN model
    nb_param   = function() {as.integer(self$p * self$d)},
    #' @field vcov_model character: the model used for the residual covariance
    vcov_model = function() {"fixed"},
    #' @field vcov_coef matrix of sandwich estimator of the variance-covariance of B (needs known covariance at the moment)
    vcov_coef = function() {attr(private$B, "vcov_sandwich")}
  )
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ##  END OF THE CLASS PLNfit_fixedcov
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  CLASS PLNfit_fixedv
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' An R6 Class to represent a PLNfit in a standard, general framework, with fixed eigenvectors of Covariance matrix
#'
#' @param responses the matrix of responses (called Y in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param covariates design matrix (called X in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param offsets offset matrix (called O in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param data an optional data frame, list or environment (or object coercible by as.data.frame to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which PLN is called.
#' @param weights an optional vector of observation weights to be used in the fitting process.
#' @param nullModel null model used for approximate R2 computations. Defaults to a GLM model with same design matrix but not latent variable.
#' @param formula model formula used for fitting, extracted from the formula in the upper-level call
#' @param control a list for controlling the optimization. See details.
#' @param config part of the \code{control} argument which configures the optimizer
#'
#' @rdname PLNfit_fixedv
#' @importFrom R6 R6Class
#'
#' @examples
#' \dontrun{
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLN(Abundance ~ 1, data = trichoptera)
#' class(myPLN)
#' print(myPLN)
#' }
PLNfit_fixedv <- R6Class(
  classname = "PLNfit_fixedv",
  inherit = PLNfit,
  private = list(V = NULL, L = NULL, q = NULL),
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## PUBLIC MEMBERS ----
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  public  = list(
    #' @description Initialize a [`PLNfit_fixedv`] model
    initialize = function(responses, covariates, offsets, weights, formula, control) {
      super$initialize(responses, covariates, offsets, weights, formula, control)
      private$optimizer$main <- ifelse(control$backend == "nlopt", nlopt_optimize_fixedv, private$torch_optimize)
      private$optimizer$vestep <- nlopt_optimize_vestep_fixedv
      private$optimizer$mstep <- nlopt_optimize_mstep_fixedv
      private$q <- ifelse(is.null(control$rank), p, control$rank)
      private$V <- control$V[,1:private$q]
      private$L <- control$config_optim$L0

      if (private$q < self$p) {
        private$M  <- matrix(0, self$n, private$q)
        private$S  <- matrix(1, self$n, private$q)
        private$B  <- matrix(0, ncol(covariates), private$q)
      }
      
    },
    #' @description Update a [`PLNfit_fixedv`] object
    update = function(B=NA, Sigma=NA, Omega=NA, L=NA, M=NA, S=NA, Z=NA, A=NA, Ji=NA, R2=NA, monitoring=NA) {
      super$update(B = B, Sigma = Sigma, Omega = Omega, M = M, S = S, Z = Z, A = A, Ji = Ji, R2 = R2, monitoring = monitoring)
      if (!anyNA(L)) private$L <- L
    },
    #' @description Call to the NLopt or TORCH optimizer and update of the relevant fields
    optimize = function(responses, covariates, offsets, weights, config) {
      args <- list(data   = list(Y = responses, X = covariates, O = offsets, w = weights),
                   params = list(B = private$B, M = private$M, S = private$S, L = private$L, V = private$V),
                   config = config)
      optim_out <- do.call(private$optimizer$main, args)
      do.call(self$update, optim_out)
    },
    
    #' @description Result of one call to the M step of teh optimization procedure: optimal model parameters (C,B) for fixed variational parameters (M,S)
    optimize_mstep = function(responses, covariates, offsets, weights, config) {
      # problem dimension
      n <- nrow(responses); p <- ncol(responses); q <- private$q
      
      # turn offset vector to offset matrix
      offsets <- as.matrix(offsets)
      if (ncol(offsets) == 1) offsets <- matrix(offsets, nrow = n, ncol = p)
      
      # get smoothed M and S
      #self$smooth(Y = responses, M = private$M, S = private$S)
      # Neural network for M
      model.m <- keras_model_sequential()
      model.m %>% 
        layer_dense(units = 30, activation = "relu", input_shape = c(p)) %>%
        layer_dense(units = 30, activation = "relu") %>%
        layer_dense(units = q, activation = "linear")
      model.m %>% compile(loss = "mean_squared_error", optimizer = optimizer_adam(lr = 0.001))
      history.m <- model.m %>% fit(
        log(responses+1), private$M,
        epochs = 150,
        batch_size = 32,
        validation_split = 0.3,
        callbacks = list(callback_early_stopping(monitor = "val_loss", 
                                                 patience = 10, 
                                                 restore_best_weights = TRUE))
      )
      private$M.new <- predict(model.m, log(responses+1))
      
      # Neural network for S
      model.s <- keras_model_sequential()
      model.s %>%
        layer_dense(units = 30, activation = "relu", input_shape = c(p)) %>%
        layer_dense(units = 30, activation = "relu") %>%
        layer_dense(units = q, activation = "softplus")
      model.s %>% compile(loss = "mean_squared_error", optimizer = optimizer_adam(lr = 0.001))
      history.s <- model.s %>% fit(
        log(responses+1), abs(private$S),
        epochs = 150,
        batch_size = 32,
        validation_split = 0.3,
        callbacks = list(callback_early_stopping(monitor = "val_loss",
                                                 patience = 10,
                                                 restore_best_weights = TRUE))
      )
      private$S.new <- predict(model.s, log(responses+1))
      
      args <- list(data   = list(Y = responses, X = covariates, O = offsets, w = weights),
                   ## Initialize the model parameters with the current values
                   params = list(B = private$B, L = private$L, V = private$V),
                   M = private$M.new,
                   S = private$S.new,
                   config = config)
      optim_out <- do.call(private$optimizer$mstep, args)
      private$B.new <- optim_out$B
      private$Sigma.new <- optim_out$Sigma
      
    },
    
    #' @description Result of one call to the VE step of the optimization procedure: optimal variational parameters (M, S) and corresponding log likelihood values for fixed model parameters (C, B). Intended to position new data in the latent space for further use with PCA.
    #' @return A list with three components:
    #'  * the matrix `M` of variational means,
    #'  * the matrix `S2` of variational variances
    #'  * the vector `log.lik` of (variational) log-likelihood of each new observation
    #' @description Result of one call to the VE step of the optimization procedure: optimal variational parameters (M, S) and corresponding log likelihood values for fixed model parameters (Sigma, B). Intended to position new data in the latent space.
    #' @param B Optional fixed value of the regression parameters
    #' @param L diagonal matrix with sqrt(Lambda)
    #' @return A list with three components:
    #'  * the matrix `M` of variational means,
    #'  * the matrix `S2` of variational variances
    #'  * the vector `log.lik` of (variational) log-likelihood of each new observation
    optimize_vestep = function(covariates, offsets, responses, weights,
                               B = self$model_par$B,
                               L = self$model_par$L,
                               control = PLN_param(backend = "nlopt")) {
      n <- nrow(responses); q <- private$q
      args <- list(data   = list(Y = responses, X = covariates, O = offsets, w = weights),
                   ## Initialize the variational parameters with the new dimension of the data
                   params = list(M = matrix(0, n, q), S = matrix(1, n, q), V = private$V),
                   B = as.matrix(B),
                   L = as.matrix(L),
                   config = control$config_optim)
      optim_out <- do.call(private$optimizer$vestep, args)
      optim_out
    },
    
    #' @description Update R2, fisher and std_err fields after optimization
    #' @param config a list for controlling the post-treatments (optional bootstrap, jackknife, R2, etc.). See details
    #' @details The list of parameters `config` controls the post-treatment processing, with the following entries:
    #' * trace integer for verbosity. should be > 1 to see output in post-treatments
    #' * jackknife boolean indicating whether jackknife should be performed to evaluate bias and variance of the model parameters. Default is FALSE.
    #' * bootstrap integer indicating the number of bootstrap resamples generated to evaluate the variance of the model parameters. Default is 0 (inactivated).
    #' * variational_var boolean indicating whether variational Fisher information matrix should be computed to estimate the variance of the model parameters (highly underestimated). Default is FALSE.
    #' * rsquared boolean indicating whether approximation of R2 based on deviance should be computed. Default is TRUE
    postTreatment = function(responses, covariates, offsets, weights = rep(1, nrow(responses)), config, nullModel = NULL) {
      super$postTreatment(responses, covariates, offsets, weights, config, nullModel)
      ## 6. compute and store matrix of standard variances for B with sandwich correction approximation
      if (config$sandwich_var) {
        if(config$trace > 1) cat("\n\tComputing sandwich estimator of the variance...")
        private$vcov_sandwich_B(responses, covariates)
      }
    }
  ),
  #private = list(
  # ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # ## PRIVATE TORCH METHODS FOR OPTIMIZATION
  # ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # torch_elbo = function(data, params, index=torch_tensor(1:self$n)) {
  #   S2 <- torch_square(params$S[index])
  #   Z <- data$O[index] + params$M[index] + torch_mm(data$X[index], params$B)
  #   res <- sum(data$w) * torch_trace(torch_mm(private$torch_Sigma(data, params, index), private$torch_Omega(data, params))) +
  #     sum(data$w[index,NULL] * (torch_exp(Z + .5 * S2) - data$Y[index] * Z - .5 * torch_log(S2)))
  #   res
  # },
  # 
  # torch_Omega = function(data, params) {
  #   params$Omega <- torch_tensor(private$Omega)
  # },
  
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## END OF TORCH METHODS FOR OPTIMIZATION
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  active = list(
    #' @field nb_param number of parameters in the current PLN model
    nb_param   = function() {as.integer(self$p * self$d + self$p)},
    #' @field vcov_model character: the model used for the residual covariance
    vcov_model = function() {"fixedV"},
    #' @field vcov_coef matrix of sandwich estimator of the variance-covariance of B (needs known covariance at the moment)
    vcov_coef = function() {attr(private$B, "vcov_sandwich")},
    #' @field model_par a list with the matrices associated with the estimated parameters of the pPCA model: B (covariates), Sigma (covariance), Omega (precision) and C (loadings)
    model_par = function() {
      par <- super$model_par
      par$L <- private$L
      par
    }
  )
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ##  END OF THE CLASS PLNfit_fixedv
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
)


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  CLASS PLNfit_lowrank
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' An R6 Class to represent a PLNfit in a standard, general framework, with fixed eigenvectors of Covariance matrix
#'
#' @param responses the matrix of responses (called Y in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param covariates design matrix (called X in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param offsets offset matrix (called O in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param data an optional data frame, list or environment (or object coercible by as.data.frame to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which PLN is called.
#' @param weights an optional vector of observation weights to be used in the fitting process.
#' @param nullModel null model used for approximate R2 computations. Defaults to a GLM model with same design matrix but not latent variable.
#' @param formula model formula used for fitting, extracted from the formula in the upper-level call
#' @param control a list for controlling the optimization. See details.
#' @param config part of the \code{control} argument which configures the optimizer
#'
#' @rdname PLNfit_lowrank
#' @importFrom R6 R6Class
#'
#' @examples
#' \dontrun{
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLN(Abundance ~ 1, data = trichoptera)
#' class(myPLN)
#' print(myPLN)
#' }
PLNfit_lowrank <- R6Class(
  classname = "PLNfit_lowrank",
  inherit = PLNfit,
  private = list(C = NULL, svdCM = NULL, q=NULL),
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## PUBLIC MEMBERS ----
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  public  = list(
    #' @description Initialize a [`PLNfit_lowrank`] model
    initialize = function(responses, covariates, offsets, weights, formula, control) {
      super$initialize(responses, covariates, offsets, weights, formula, control)
      private$optimizer$main   <- nlopt_optimize_rank
      private$optimizer$vestep <- nlopt_optimize_vestep_rank
      private$q <- control$rank

      if (!is.null(control$svdM)) {
        svdM <- control$svdM
      } else {
        svdM <- svd(private$M, nu = private$q, nv = self$p)
      }
      ## TODO: check that it is really better than initializing with zeros...
      private$M  <- svdM$u[, 1:private$q, drop = FALSE] %*% diag(svdM$d[1:private$q], nrow = private$q, ncol = private$q) %*% t(svdM$v[1:private$q, 1:private$q, drop = FALSE])
      private$S  <- matrix(0.1, self$n, private$q)
      private$C  <- svdM$v[, 1:private$q, drop = FALSE] %*% diag(svdM$d[1:private$q], nrow = private$q, ncol = private$q)/sqrt(self$n)
      private$B  <- matrix(0, ncol(covariates), private$q)
    },
    #' @description Update a [`PLNfit_lowrank`] object
    update = function(B=NA, Sigma=NA, Omega=NA, C=NA, M=NA, S=NA, Z=NA, A=NA, Ji=NA, R2=NA, monitoring=NA) {
      super$update(B = B, Sigma = Sigma, Omega = Omega, M = M, S = S, Z = Z, A = A, Ji = Ji, R2 = R2, monitoring = monitoring)
      if (!anyNA(C)) private$C <- C
    },
    #' @description Call to the NLopt or TORCH optimizer and update of the relevant fields
    optimize = function(responses, covariates, offsets, weights, config) {
      args <- list(data   = list(Y = responses, X = covariates, # covariates here already include groups
                                 O = offsets, w = weights),
                   params = list(B = private$B, C = private$C, M = private$M, S = private$S),
                   config = config)
      optim_out <- do.call(private$optimizer$main, args)
      do.call(self$update, optim_out)
    },
    

    #' @description Result of one call to the VE step of the optimization procedure: optimal variational parameters (M, S) and corresponding log likelihood values for fixed model parameters (C, B). Intended to position new data in the latent space for further use with PCA.
    #' @return A list with three components:
    #'  * the matrix `M` of variational means,
    #'  * the matrix `S2` of variational variances
    #'  * the vector `log.lik` of (variational) log-likelihood of each new observation
    optimize_vestep = function(covariates, offsets, responses, weights = rep(1, self$n),
                               control = PLNPCA_param(backend = "nlopt")) {
      
      # problem dimension
      n <- nrow(responses); p <- ncol(responses); q <- self$q
      
      # turn offset vector to offset matrix
      offsets <- as.matrix(offsets)
      if (ncol(offsets) == 1) offsets <- matrix(offsets, nrow = n, ncol = p)
      
      ## Not completely naive starting values for M: SVD on the residuals of
      ## a linear regression on the log-counts (+1 to deal with 0s)
      log_responses <- log(responses+1)
      residuals <- lm.wfit(covariates, y = log_responses, w = weights, offset = offsets)$residuals
      svd_residuals <- svd(residuals, nu = q, nv = p)
      M_init <- svd_residuals$u[, 1:q, drop = FALSE] %*% diag(svd_residuals$d[1:q], nrow = q, ncol = q) %*% t(svd_residuals$v[1:q, 1:q, drop = FALSE])
      
      ## Initialize the variational parameters with the appropriate new dimension of the data
      args <- list(data   = list(Y = responses, X = covariates, O = offsets, w = weights),
                   ## Initialize the variational parameters with the new dimension of the data
                   params = list(M = M_init, S = matrix(1, n, q)),
                   B = private$B,
                   C = private$C,
                   config = control$config_optim)
      optim_out <- do.call(private$optimizer$vestep, args)
      optim_out
    },
    
    
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## Post treatment --------------------
    #' @description Compute PCA scores in the latent space and update corresponding fields.
    #' @param scale.unit Logical. Should PCA scores be rescaled to have unit variance
    setVisualization = function(scale.unit = FALSE) {
      private$svdCM <- svd(scale(self$latent_pos, TRUE, scale.unit), nv = self$rank)
    },
    
    #' @description Update R2, fisher, std_err fields and set up visualization
    #' @details The list of parameters `config` controls the post-treatment processing, with the following entries:
    #' * jackknife boolean indicating whether jackknife should be performed to evaluate bias and variance of the model parameters. Default is FALSE.
    #' * bootstrap integer indicating the number of bootstrap resamples generated to evaluate the variance of the model parameters. Default is 0 (inactivated).
    #' * variational_var boolean indicating whether variational Fisher information matrix should be computed to estimate the variance of the model parameters (highly underestimated). Default is FALSE.
    #' * rsquared boolean indicating whether approximation of R2 based on deviance should be computed. Default is TRUE
    #' * trace integer for verbosity. should be > 1 to see output in post-treatments
    postTreatment = function(responses, covariates, offsets, weights, config, nullModel) {
      #super$postTreatment(responses, covariates, offsets, weights, config, nullModel)
      #colnames(private$C) <- colnames(private$M) <- 1:self$q
      #rownames(private$C) <- colnames(responses)
      #self$setVisualization()
    }
  ),
  #private = list(
  # ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # ## PRIVATE TORCH METHODS FOR OPTIMIZATION
  # ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # torch_elbo = function(data, params, index=torch_tensor(1:self$n)) {
  #   S2 <- torch_square(params$S[index])
  #   Z <- data$O[index] + params$M[index] + torch_mm(data$X[index], params$B)
  #   res <- sum(data$w) * torch_trace(torch_mm(private$torch_Sigma(data, params, index), private$torch_Omega(data, params))) +
  #     sum(data$w[index,NULL] * (torch_exp(Z + .5 * S2) - data$Y[index] * Z - .5 * torch_log(S2)))
  #   res
  # },
  # 
  # torch_Omega = function(data, params) {
  #   params$Omega <- torch_tensor(private$Omega)
  # },
  
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## END OF TORCH METHODS FOR OPTIMIZATION
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  active = list(
    #' @field rank the dimension of the current model
    rank = function() {self$q},
    #' @field vcov_model character: the model used for the residual covariance
    vcov_model = function() {"rank"},
    #' @field nb_param number of parameters in the current PLN model
    nb_param = function() {self$p * (self$d + self$q) - self$q * (self$q - 1)/2},
    #' @field entropy entropy of the variational distribution
    entropy  = function() {.5 * (self$n * self$q * log(2*pi*exp(1)) + sum(log(self$var_par$S2)))},
    #' @field latent_pos a matrix: values of the latent position vector (Z) without covariates effects or offset
    latent_pos = function() {tcrossprod(private$M, private$C)},
    #' @field model_par a list with the matrices associated with the estimated parameters of the pPCA model: B (covariates), Sigma (covariance), Omega (precision) and C (loadings)
    model_par = function() {
      par <- super$model_par
      par$C <- private$C
      par
    },
    #' @field percent_var the percent of variance explained by each axis
    percent_var = function() {
      eigen.val <- private$svdCM$d[1:self$rank]^2
      setNames(round(eigen.val/sum(eigen.val)*self$R_squared,4), paste0("PC", 1:self$rank))
    },
    #' @field corr_circle a matrix of correlations to plot the correlation circles
    corr_circle = function() {
      corr <- private$svdCM$v[, 1:self$rank] * matrix(private$svdCM$d[1:self$rank], byrow = TRUE, nrow = self$p, ncol = self$rank)
      corr <- corr/sqrt(rowSums(corr^2))
      rownames(corr) <- rownames(private$Sigma)
      colnames(corr) <- paste0("PC", 1:self$rank)
      corr
    },
    #' @field scores a matrix of scores to plot the individual factor maps (a.k.a. principal components)
    scores     = function() {
      scores <- private$svdCM$u[, 1:self$rank] * matrix(private$svdCM$d[1:self$rank], byrow = TRUE, nrow = self$n, ncol = self$rank)
      rownames(scores) <- rownames(private$M)
      colnames(scores) <- paste0("PC", 1:self$rank)
      scores
    },
    #' @field rotation a matrix of rotation of the latent space
    rotation   = function() {
      rotation <- private$svdCM$v[, 1:self$rank, drop = FALSE]
      rownames(rotation) <- rownames(private$Sigma)
      colnames(rotation) <- paste0("PC", 1:self$rank)
      rotation
    },
    #' @field eig description of the eigenvalues, similar to percent_var but for use with external methods
    eig = function() {
      eigen.val <- private$svdCM$d[1:self$rank]^2
      matrix(
        c(eigen.val,                                                # eigenvalues
          100 * self$R_squared * eigen.val / sum(eigen.val),        # percentage of variance
          100 * self$R_squared * cumsum(eigen.val) / sum(eigen.val) # cumulative percentage of variance
        ),
        ncol = 3,
        dimnames = list(paste("comp", 1:self$rank), c("eigenvalue", "percentage of variance", "cumulative percentage of variance"))
      )
    },
    #' @field var a list of data frames with PCA results for the variables: `coord` (coordinates of the variables), `cor` (correlation between variables and dimensions), `cos2` (Cosine of the variables) and `contrib` (contributions of the variable to the axes)
    var = function() {
      coord  <- private$svdCM$v[, 1:self$rank] * matrix(private$svdCM$d[1:self$rank], ncol = self$rank, nrow = self$p, byrow = TRUE)
      ## coord[j, k] = d[k] * v[j, k]
      var_sd <- sqrt(rowSums(coord^2))
      coord  <- coord / var_sd
      cor    <- coord
      cos2 <- cor^2
      contrib <- 100 * private$svdCM$v[, 1:self$rank, drop = FALSE]^2
      dimnames(coord) <- dimnames(cor) <- dimnames(cos2) <- dimnames(contrib) <- list(rownames(private$Sigma), paste0("Dim.", 1:self$rank))
      list(coord   = coord,
           cor     = cor,
           cos2    = cos2,
           contrib = contrib)
    },
    #' @field ind a list of data frames with PCA results for the individuals: `coord` (coordinates of the individuals), `cos2` (Cosine of the individuals), `contrib` (contributions of individuals to an axis inertia) and `dist` (distance of individuals to the origin).
    ind = function() {
      coord  <- private$svdCM$u[, 1:self$rank] * matrix(private$svdCM$d[1:self$rank], ncol = self$rank, nrow = self$n, byrow = TRUE)
      ## coord[i, k] = d[k] * v[i, k]
      dist_origin <- sqrt(rowSums(coord^2))
      cos2 <- coord^2 / dist_origin^2
      contrib <- 100 * private$svdCM$u[, 1:self$rank, drop = FALSE]^2
      dimnames(coord) <- dimnames(cos2) <- dimnames(contrib) <- list(rownames(private$M), paste0("Dim.", 1:self$rank))
      names(dist_origin) <- rownames(private$M)
      list(coord   = coord,
           cos2    = cos2,
           contrib = contrib,
           dist    = dist_origin)
    },
    #' @field call Hacky binding for compatibility with factoextra functions
    call = function() {
      list(scale.unit = FALSE)
    }
  )
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ##  END OF THE CLASS PLNfit_fixedv
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  CLASS PLNfit_fixedc
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' An R6 Class to represent a PLNfit in a standard, general framework, with fixed eigenvectors of Covariance matrix
#'
#' @param responses the matrix of responses (called Y in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param covariates design matrix (called X in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param offsets offset matrix (called O in the model). Will usually be extracted from the corresponding field in PLNfamily-class
#' @param data an optional data frame, list or environment (or object coercible by as.data.frame to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which PLN is called.
#' @param weights an optional vector of observation weights to be used in the fitting process.
#' @param nullModel null model used for approximate R2 computations. Defaults to a GLM model with same design matrix but not latent variable.
#' @param formula model formula used for fitting, extracted from the formula in the upper-level call
#' @param control a list for controlling the optimization. See details.
#' @param config part of the \code{control} argument which configures the optimizer
#'
#' @rdname PLNfit_fixedc
#' @importFrom R6 R6Class
#'
#' @examples
#' \dontrun{
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLN(Abundance ~ 1, data = trichoptera)
#' class(myPLN)
#' print(myPLN)
#' }
PLNfit_fixedc <- R6Class(
  classname = "PLNfit_fixedc",
  inherit = PLNfit,
  private = list(C = NULL, q = NULL),
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## PUBLIC MEMBERS ----
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  public  = list(
    #' @description Initialize a [`PLNfit_fixedc`] model
    initialize = function(responses, covariates, C, offsets, weights, formula, control) {
      super$initialize(responses, covariates, offsets, weights, formula, control)
      private$optimizer$main <- ifelse(control$backend == "nlopt", nlopt_optimize_fixedc, private$torch_optimize)
      private$optimizer$vestep <- nlopt_optimize_vestep_fixedc
      private$C <- C
      private$q <- ncol(private$C)
      
      if (private$q < self$p) {
        private$M  <- matrix(0, self$n, private$q)
        private$S  <- matrix(1, self$n, private$q)
      }
    },

    #' @description Call to the NLopt or TORCH optimizer and update of the relevant fields
    optimize = function(responses, covariates, C, offsets, weights, config) {
      args <- list(data   = list(Y = responses, X = covariates, O = offsets, w = weights),
                   params = list(B = private$B, M = private$M, S = private$S, C = C),
                   config = config)
      optim_out <- do.call(private$optimizer$main, args)
      do.call(self$update, optim_out)
    },
    
    
    #' @description Update R2, fisher and std_err fields after optimization
    #' @param config a list for controlling the post-treatments (optional bootstrap, jackknife, R2, etc.). See details
    #' @details The list of parameters `config` controls the post-treatment processing, with the following entries:
    #' * trace integer for verbosity. should be > 1 to see output in post-treatments
    #' * jackknife boolean indicating whether jackknife should be performed to evaluate bias and variance of the model parameters. Default is FALSE.
    #' * bootstrap integer indicating the number of bootstrap resamples generated to evaluate the variance of the model parameters. Default is 0 (inactivated).
    #' * variational_var boolean indicating whether variational Fisher information matrix should be computed to estimate the variance of the model parameters (highly underestimated). Default is FALSE.
    #' * rsquared boolean indicating whether approximation of R2 based on deviance should be computed. Default is TRUE
    postTreatment = function(responses, covariates, offsets, weights = rep(1, nrow(responses)), config, nullModel = NULL) {
      super$postTreatment(responses, covariates, offsets, weights, config, nullModel)
      ## 6. compute and store matrix of standard variances for B with sandwich correction approximation
      if (config$sandwich_var) {
        if(config$trace > 1) cat("\n\tComputing sandwich estimator of the variance...")
        private$vcov_sandwich_B(responses, covariates)
      }
    }
  ),
  
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## END OF TORCH METHODS FOR OPTIMIZATION
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  active = list(
    #' @field nb_param number of parameters in the current PLN model
    nb_param   = function() {as.integer(self$p * self$d)},
    #' @field vcov_model character: the model used for the residual covariance
    vcov_model = function() {"fixedC"},
    #' @field vcov_coef matrix of sandwich estimator of the variance-covariance of B (needs known covariance at the moment)
    vcov_coef = function() {attr(private$B, "vcov_sandwich")},
    #' @field model_par a list with the matrices associated with the estimated parameters of the pPCA model: B (covariates), Sigma (covariance), Omega (precision) and C (loadings)
    model_par = function() {
      par <- super$model_par
      par
    }
  )
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ##  END OF THE CLASS PLNfit_fixedc
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
)
