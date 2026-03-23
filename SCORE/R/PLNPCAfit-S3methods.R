## =========================================================================================
##
## PUBLIC S3 METHODS FOR PLNPCAfit
##
## =========================================================================================

## Auxiliary functions to check the given class of an objet
isPLNPCAfit <- function(Robject) {inherits(Robject, "PLNPCAfit"       )}

#' PCA visualization (individual and/or variable factor map(s)) for a [`PLNPCAfit`] object
#'
#' @name plot.PLNPCAfit
#'
#' @param x an R6 object with class PLNPCAfit
#' @param map the type of output for the PCA visualization: either "individual", "variable" or "both". Default is "both".
#' @param nb_axes scalar: the number of axes to be considered when `map = "both"`. The default is `min(3,rank)`.
#' @param axes numeric, the axes to use for the plot when `map = "individual"` or `map = "variable"`. Default it `c(1,min(rank))`
#' @param ind_cols a character, factor or numeric to define the color associated with the individuals. By default, all variables receive the default color of the current palette.
#' @param var_cols a character, factor or numeric to define the color associated with the variables. By default, all variables receive the default color of the current palette.
#' @param plot logical. Should the plot be displayed or sent back as [`ggplot`] object
#' @param main character. A title for the single plot (individual or variable factor map). If NULL (the default), an hopefully appropriate title will be used.
#' @param ... Not used (S3 compatibility).
#'
#' @return displays an individual and/or variable factor maps for the corresponding axes, and/or sends back a [`ggplot`] or gtable object
#' @examples
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPCAs <- PLNPCA(Abundance ~ 1 + offset(log(Offset)), data = trichoptera, ranks = 1:5)
#' myPCA <- getBestModel(myPCAs)
#' \dontrun{
#' plot(myPCA, map = "individual", nb_axes=2, ind_cols = trichoptera$Group)
#' plot(myPCA, map = "variable", nb_axes=2)
#' plot(myPCA, map = "both", nb_axes=2, ind_cols = trichoptera$Group)
#' }
#' @export
plot.PLNPCAfit <-
  function(x,
           map      = c("both", "individual", "variable"),
           nb_axes  = min(3, x$rank),
           axes     = seq.int(min(2,x$rank)),
           ind_cols = "ind_colors",
           var_cols = "var_colors",
           plot     = TRUE,
           main = NULL,
           ...) {

    stopifnot(isPLNPCAfit(x))

    map <- match.arg(map)

    if (is.null(main)) {
      if (map == "individual")
        main <- "Individual factor map"
      if (map == "variable")
        main <- "Variable factor map"
    }

    if (map == "individual")
      p <- x$plot_individual_map(axes = axes, cols = ind_cols, main = main, plot = plot)
    if (map == "variable")
      p <- x$plot_correlation_circle(axes = axes, cols = var_cols, main = main, plot = plot)
    if (map == "both")
      p <- x$plot_PCA(nb_axes = nb_axes, ind_cols = ind_cols, var_cols = var_cols, plot = plot)

    invisible(p)
  }

#' @describeIn standard_error Component-wise standard errors of B in [`PLNPCAfit`] (not implemented yet)
#' @export
standard_error.PLNPCAfit <- function(object, type = c("variational", "jackknife", "sandwich"), parameter = c("B", "Omega")) {
  par  <- match.arg(parameter)
  if (par == "Omega")
    stop("Omega is not estimated as such in PLNPCA models")
  if (par == "B") {
    warning("Standard error of B is not implemented yet for PLNPCA models")
    theta_sd <- coef.PLNfit(object)
    theta_sd[ , ] <- NA
    theta_sd
  }
}

#' Predict group of new samples
#'
#' @name predict.PLNPCAfit
#'
#' @param object an R6 object with class [`PLNPCAfit`]
#' @param newdata A data frame in which to look for variables, offsets and counts  with which to predict.
#' @param type The type of prediction required. The default are posterior probabilities for each group (in either unnormalized log-scale or natural probabilities, see "scale" for details), "response" is the group with maximal posterior probability and "scores" is the average score along each separation axis in the latent space, with weights equal to the posterior probabilities.
#' @param scale The scale used for the posterior probability. Either log-scale ("log", default) or natural probabilities summing up to 1 ("prob").
#' @param prior User-specified prior group probabilities in the new data. If NULL (default), prior probabilities are computed from the learning set.
#' @param control a list for controlling the optimization. See [PLN()] for details.
#' @param ... additional parameters for S3 compatibility. Not used
#' @return A matrix of posterior probabilities for each group (if type = "posterior"), a matrix of (average) scores in the latent space (if type = "scores") or a vector of predicted groups (if type = "response").
#' @export
predict.PLNPCAfit <- function(object, newdata,
                              type = c("posterior", "group", "latent"),
                              prior = NULL,
                              control = PLN_param(backend="nlopt"), ...) {
  stopifnot(isPLNPCAfit(object))
  object$project(newdata, type, prior, control, parent.frame())
}

