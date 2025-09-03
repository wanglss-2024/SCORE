## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  CLASS PLNfixedVsupfit
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
#' @rdname PLNfixedVsupfit
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
PLNfixedVsupfit <- R6Class(
  classname = "PLNfixedVsupfit",
  inherit = PLNfit_fixedv,
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## PUBLIC MEMBERS ----
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  public  = list(
    #' @description Initialize a [`PLNfixedVsupfit`] model
    initialize = function(grouping, responses, covariates, offsets, weights, formula, control) {
      private$grouping <- grouping
      super$initialize(responses, cbind(covariates, grouping), offsets, weights, formula, control)
      # super$initialize(responses, cbind(covariates, model.matrix( ~ grouping + 0)), offsets, weights, formula, control)
      # private$optimizer$main <- nlopt_optimize_fixedv
      # private$optimizer$vestep <- nlopt_optimize_vestep_fixedv
      # #private$optimizer$mstep <- nlopt_optimize_mstep_fixedv
      # private$grouping <- grouping
      # private$q <- ifelse(is.null(control$rank), p, control$rank)
      # private$V <- control$V[,1:private$q]
      # private$L <- diag(10, private$q)
      # 
      # if (private$q < self$p) {
      #   private$M  <- matrix(0, self$n, private$q)
      #   private$S  <- matrix(1, self$n, private$q)
      # }
      
    },
    #' @description Update a [`PLNfixedVsupfit`] object
    update = function(B=NA, Sigma=NA, Omega=NA, L=NA, M=NA, S=NA, Z=NA, A=NA, Ji=NA, R2=NA, monitoring=NA) {
      super$update(B = B, Sigma = Sigma, Omega = Omega, M = M, S = S, Z = Z, A = A, Ji = Ji, R2 = R2, monitoring = monitoring)
      if (!anyNA(L)) private$L <- L
    },
    
    #' @description Call to the NLopt or TORCH optimizer and update of the relevant fields
    optimize = function(grouping, responses, covariates, offsets, weights, config) {
      super$optimize(responses, cbind(covariates, grouping), offsets, weights, config)
      #design_group <- model.matrix( ~ grouping + 0)
      
      # ## extract group means
      # if (ncol(covariates) > 0) {
      #   proj_orth_X <- (diag(self$n) - covariates %*% solve(crossprod(covariates)) %*% t(covariates))
      #   P <- proj_orth_X %*% ((cbind(covariates, design_group) %*% private$B) + private$M)
      #   Mu <- t(rowsum(P, private$grouping) / tabulate(private$grouping))
      # } else {
      #   Mu <- t(private$B)
      # }
      # colnames(Mu) <- colnames(design_group)
      # rownames(Mu) <- colnames(private$B)
      # private$Mu <- Mu
      # nk <- table(private$grouping)
      # Mu_bar <- as.vector(Mu %*% nk / self$n)
      # private$C <- Mu %*% diag(nk) %*% t(Mu) / self$n - Mu_bar %o% Mu_bar
    },
    
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## Post treatment --------------------
  #' @description  Update R2, fisher and std_err fields and visualization
  #' @param config list controlling the post-treatment
  postTreatment = function(grouping, responses, covariates, offsets, config) {
    covariates <- cbind(covariates, grouping)
    super$postTreatment(responses, covariates, offsets, config = config)
    rownames(private$C) <- colnames(private$C) <- colnames(responses)
    colnames(private$S) <- 1:self$q
    if (config$trace > 1) cat("\n\tCompute LD scores for visualization...")
    #self$setVisualization()
  },
  
  #' @description Compute LDA scores in the latent space and update corresponding fields.
  #' @param scale.unit Logical. Should LDA scores be rescaled to have unit variance
  setVisualization = function(scale.unit = FALSE) {
    Wm1C <- solve(private$Sigma) %*% private$C
    private$svdLDA <- svd(scale(Wm1C,TRUE, scale.unit), nv = self$rank)
    P <- private$M + tcrossprod(model.matrix( ~ private$grouping + 0), private$Mu) ## P = M + G Mu
    private$P <- scale(P, TRUE, scale.unit)
  },
  
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## Graphical methods -----------------
  #' @description Plot the factorial map of the LDA
  # @inheritParams plot.PLNLDAfit
  #' @return a [`ggplot`] graphic
  plot_individual_map = function(axes = 1:min(2,self$rank), main = "Individual Factor Map", plot = TRUE) {
    
    .scores <- data.frame(self$scores[,axes, drop = FALSE])
    colnames(.scores) <- paste("a",1:ncol(.scores),sep = "")
    .scores$labels <- private$grouping
    .scores$names <- rownames(private$M)
    axes_label <- paste(paste("axis",axes), paste0("(", round(100*self$percent_var,3)[axes], "%)"))
    
    p <- get_ggplot_ind_map(.scores, axes_label, main)
    if (plot) print(p)
    invisible(p)
  },
  
  #' @description Plot the correlation circle of a specified axis for a [`PLNLDAfit`] object
  # @inheritParams plot.PLNLDAfit
  #' @param cols a character, factor or numeric to define the color associated with the variables. By default, all variables receive the default color of the current palette.
  #' @return a [`ggplot`] graphic
  plot_correlation_map = function(axes=1:min(2,self$rank), main="Variable Factor Map", cols = "default", plot=TRUE) {
    
    ## data frame with correlations between variables and PCs
    correlations <- as.data.frame(self$corr_map[, axes, drop = FALSE])
    colnames(correlations) <- paste0("axe", 1:length(axes))
    correlations$labels <- cols
    correlations$names  <- rownames(correlations)
    axes_label <- paste(paste("axis",axes), paste0("(", round(100*self$percent_var,3)[axes], "%)"))
    
    p <- get_ggplot_corr_square(correlations, axes_label, main)
    
    if (plot) print(p)
    invisible(p)
  },
  
  #' @description Plot a summary of the [`PLNLDAfit`] object
  # @inheritParams plot.PLNLDAfit
  #' @importFrom gridExtra grid.arrange arrangeGrob
  #' @importFrom grid nullGrob textGrob
  #' @return a [`grob`] object
  plot_LDA = function(nb_axes = min(3, self$rank), var_cols = "default", plot = TRUE) {
    
    axes <- 1:nb_axes
    if (nb_axes > 1) {
      pairs.axes <- combn(axes, 2, simplify = FALSE)
      
      ## get back all individual maps
      ind.plot <- lapply(pairs.axes, function(pair) {
        ggobj <- self$plot_individual_map(axes = pair, plot = FALSE, main="") + ggplot2::theme(legend.position="none")
        ggplot2::ggplotGrob(ggobj)
      })
      
      ## get back all correlation circle
      cor.plot <- lapply(pairs.axes, function(pair) {
        ggobj <- self$plot_correlation_map(axes = pair, plot = FALSE, main = "", cols = var_cols)
        ggplot2::ggplotGrob(ggobj)
      })
      
      ## plot that appear on the diagonal
      crit <- setNames(c(NA,NA,NA), c("loglikelihood", "BIC", "ICL"))
      criteria.text <- paste("Model Selection\n\n", paste(names(crit), round(crit, 2), sep=" = ", collapse="\n"))
      percentV.text <- paste("Axes contribution\n\n", paste(paste("axis",axes), paste0(": ", round(100*self$percent_var[axes],3), "%"), collapse="\n"))
      
      diag.grobs <- list(textGrob(percentV.text),
                         g_legend(self$plot_individual_map(plot=FALSE) + ggplot2::guides(colour = ggplot2::guide_legend(nrow = 4, title="classification"))),
                         textGrob(criteria.text))
      if (nb_axes > 3)
        diag.grobs <- c(diag.grobs, rep(list(nullGrob()), nb_axes - 3))
      
      
      grobs <- vector("list", nb_axes^2)
      i.cor <- 1; i.ind <- 1; i.dia <- 1
      ind <- 0
      for (i in 1:nb_axes) {
        for (j in 1:nb_axes) {
          ind <- ind+1
          if (j > i) { ## upper triangular  -> cor plot
            grobs[[ind]] <- cor.plot[[i.ind]]
            i.ind <- i.ind + 1
          } else if (i == j) { ## diagonal
            grobs[[ind]] <- diag.grobs[[i.dia]]
            i.dia <- i.dia + 1
          } else {
            grobs[[ind]] <- ind.plot[[i.cor]]
            i.cor <- i.cor + 1
          }
        }
      }
      p <- arrangeGrob(grobs = grobs, ncol = nb_axes)
    } else {
      p <- arrangeGrob(grobs = list(
        self$plot_individual_map(plot = FALSE),
        self$plot_correlation_map(plot = FALSE)
      ), ncol = 1)
    }
    if (plot)
      grid.arrange(p)
    
    invisible(p)
  },
  
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## Prediction methods --------------------
  #' @description Predict group of new samples
  # @inheritParams predict.PLNLDAfit
  #' @param newdata A data frame in which to look for variables, offsets and counts  with which to predict.
  #' @param type The type of prediction required. The default are posterior probabilities for each group (in either unnormalized log-scale or natural probabilities, see "scale" for details), "response" is the group with maximal posterior probability and "scores" is the average score along each separation axis in the latent space, with weights equal to the posterior probabilities.
  #' @param scale The scale used for the posterior probability. Either log-scale ("log", default) or natural probabilities summing up to 1 ("prob").
  #' @param prior User-specified prior group probabilities in the new data. If NULL (default), prior probabilities are computed from the learning set.
  #' @param control a list for controlling the optimization. See [PLN()] for details.
  #' @param envir Environment in which the prediction is evaluated
  predict = function(newdata,
                     type = c("posterior", "group", "latent"),
                     scale = c("log", "prob"),
                     prior = NULL,
                     control = PLN_param(backend="nlopt"), envir = parent.frame()) {
    
    type <- match.arg(type)
    
    if (type == "scores") scale <- "prob"
    scale <- match.arg(scale)
    
    ## Extract the model matrices from the new data set with initial formula
    args <- extract_model(call = call("PLNfixedVsup", formula = private$formula, data = newdata), envir = envir)
    ## Remove intercept to prevent interference with binary coding of the grouping factor
    args$X <- args$X[ , colnames(args$X) != "(Intercept)", drop = FALSE]
    
    ## Problem dimensions
    n.new  <- nrow(args$Y)
    p      <- ncol(args$Y)
    groups <- levels(private$grouping)
    K <- length(groups)
 
    ## Initialize priors
    if (is.null(prior)) {
      prior <- table(private$grouping)
    } else {
      names(prior) <- groups
    }
    if (any(prior <= 0) || anyNA(prior)) stop("Prior group proportions should be positive.")
    prior <- prior / sum(prior)
    
    ## Compute conditional log-likelihoods of new data, using previously estimated parameters
    cond.log.lik <- matrix(0, n.new, K)
    ve_step_list <- NULL
    for (k in 1:K) { ## One VE-step to estimate the conditional (variational) likelihood of each group
      grouping <- factor(rep(groups[k], n.new), levels = groups)
      X <- cbind(args$X, grouping)
      ve_step_list[[k]] <- super$optimize_vestep(X, args$O, args$Y, args$w,
                                       B = self$model_par$B,
                                       L = self$model_par$L,
                                       control = control)
      cond.log.lik[, k] <- ve_step_list[[k]]$Ji
    }
    
    ## Compute (unnormalized) posterior probabilities
    log.prior <- rep(1, n.new) %o% log(prior)
    log.posterior <- cond.log.lik + log.prior
    
    res <- log.posterior
    ## trick to avoid rounding errors before exponentiation
    row_max <- apply(res, 1, max)
    res <- exp(sweep(res, 1, row_max, "-"))
    res <- sweep(res, 1, rowSums(res), "/")
    rownames(res) <- rownames(newdata)
    colnames(res) <- groups
    g <- apply(res, 1, which.max)-1

    if (type == "posterior") {
      return (res)
    }
    if (type == "group") {
      return (g)
    }
    
    if (type == "latent") {
      latent <- matrix(NA, nrow=n.new, ncol=p)
      for (i in 1:n.new){
        if (g[i]==0) {latent[i,] <- ve_step_list[[1]]$Z[i,]}
        else {latent[i,] <- ve_step_list[[2]]$Z[i,]}
      }
      return (latent)
    }

  },
  
  ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ## Print methods -------------------------
  #' @description User friendly print method
  show = function() {
    super$show(paste0("Linear Discriminant Analysis for Poisson Lognormal distribution\n"))
    cat("* Additional fields for LDA\n")
    cat("    $percent_var, $corr_map, $scores, $group_means\n")
    cat("* Additional S3 methods for LDA\n")
    cat("    plot.PLNLDAfit(), predict.PLNLDAfit()\n")
  }
),

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## PRIVATE MEMBERS
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
private = list(
  C        = NULL,
  P        = NULL,
  Mu       = NULL,
  grouping = NULL,
  svdLDA   = NULL
),

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  ACTIVE BINDINGS ----
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
active = list(
  #' @field rank the dimension of the current model
  rank = function() {nlevels(private$grouping) - 1},
  #' @field nb_param number of parameters in the current PLN model
  nb_param = function() {self$p * (self$d + self$rank)},
  #' @field model_par a list with the matrices associated with the estimated parameters of the PLN model: B (covariates), Sigma (latent covariance), C (latent loadings), P (latent position) and Mu (group means)
  model_par = function() {
    par <- super$model_par
    par$C  <- private$C
    par$P  <- private$P
    par$Mu <- private$Mu
    par
  },
  #' @field percent_var the percent of variance explained by each axis
  percent_var = function() {
    eigen.val <- private$svdLDA$d[1:self$rank]^2
    setNames(round(eigen.val/sum(eigen.val)*self$R_squared,4), paste0("LD", 1:self$rank))
  },
  #' @field corr_map a matrix of correlations to plot the correlation circles
  corr_map = function() {
    corr <- cor(private$P, self$scores)
    rownames(corr) <- rownames(private$C)
    colnames(corr) <- paste0("LD", 1:self$rank)
    corr
  },
  #' @field scores a matrix of scores to plot the individual factor maps
  scores     = function() {
    scores <- private$P %*% t(t(private$svdLDA$u[, 1:self$rank]) * private$svdLDA$d[1:self$rank])
    rownames(scores) <- rownames(private$M)
    colnames(scores) <- paste0("LD", 1:self$rank)
    scores
  },
  #' @field group_means a matrix of group mean vectors in the latent space.
  group_means = function() {
    self$model_par$Mu
  }
)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##  END OF THE CLASS ----
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
)