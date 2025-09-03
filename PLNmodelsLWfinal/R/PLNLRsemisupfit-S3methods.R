## =========================================================================================
##
## PUBLIC S3 METHODS FOR PLNLRsemisupfit
##
## =========================================================================================

## Auxiliary functions to check the given class of an objet
isPLNLRsemisupfit <- function(Robject) {inherits(Robject, "PLNLRsemisupfit"       )}

#' Mixture visualization of a [`PLNLRsemisupfit`] object
#'
#' Represent the result of the clustering either by coloring the individual in a two-dimension PCA factor map,
#' or by representing the expected matrix  of count reorder according to the clustering.
#'
#' @name plot.PLNLRsemisupfit
#'
#' @param x an R6 object with class [`PLNLRsemisupfit`]
#' @param type character for the type of plot, either "pca", for or "matrix". Default is `"pca"`.
#' @param main character. A title for the  plot. If NULL (the default), an hopefully appropriate title will be used.
#' @param plot logical. Should the plot be displayed or sent back as [`ggplot`] object
#' @param ... Not used (S3 compatibility).
#'
#' @return a [`ggplot`] graphic
#' @examples
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLNmixture(Abundance ~ 1 + offset(log(Offset)),
#'            data = trichoptera, control = PLNmixture_param(smoothing = "none"))  %>% getBestModel()
#' \dontrun{
#' plot(myPLN, "pca")
#' plot(myPLN, "matrix")
#' }
#' @export
plot.PLNLRsemisupfit <-
  function(x,
           type           = c("pca", "matrix"),
           main           = NULL,
           plot           = TRUE, ...) {
    
    if (is.null(main))
      main <- switch(match.arg(type),
                     "pca"    = "Clustering labels in Individual Factor Map",
                     "matrix" = "Expected counts reorder by clustering")
    p <- switch(match.arg(type),
                "pca"    = x$plot_clustering_pca(main = main, plot = FALSE),
                "matrix" = x$plot_clustering_data(main = main, plot = FALSE))
    if (plot) print(p)
    invisible(p)
  }


#' Prediction for a [`PLNLRsemisupfit`] object
#'
#' Predict either posterior probabilities for each group or latent positions based on new samples
#'
#' @param object an R6 object with class [`PLNLRsemisupfit`]
#' @param newdata A data frame in which to look for variables, offsets and counts with which to predict.
#' @param type The type of prediction required. The default `posterior` are posterior probabilities for each group ,
#'  `response` is the group with maximal posterior probability and `latent` is the averaged latent in the latent space,
#'  with weights equal to the posterior probabilities.
#' @param prior User-specified prior group probabilities in the new data. The default uses a uniform prior.
#' @param control a list-like structure for controlling the fit. See [PLN_param()] for details.
#' @param ... additional parameters for S3 compatibility. Not used
#' @return A matrix of posterior probabilities for each group (if type = "posterior"), a matrix of (average) position in the
#' latent space (if type = "position") or a vector of predicted groups (if type = "response").
#' @export
#' @examples
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLNmixture(Abundance ~ 1 + offset(log(Offset)),
#'            data = trichoptera, control = PLNmixture_param(smoothing = "none"))  %>% getBestModel()
#' predict(myPLN, trichoptera, "posterior")
#' predict(myPLN, trichoptera, "position")
#' predict(myPLN, trichoptera, "response")
predict.PLNLRsemisupfit <-
  function(object, newdata,
           type = c("posterior", "group", "latent"),
           prior = matrix(rep(1/object$k, object$k), nrow(newdata), object$k, byrow = TRUE),
           control = PLN_param(), ...) {
    
    stopifnot(isPLNLRsemisupfit(object))
    object$predict(newdata, type, prior, control, parent.frame())
    
  }

#' Extract model coefficients
#'
#' @description Extracts model coefficients from objects returned by [PLN()] and its variants
#'
#' @name coef.PLNLRsemisupfit
#'
#' @param object an R6 object with class [`PLNLRsemisupfit`]
#' @param type type of parameter that should be extracted. Either "main" (default) for \deqn{\Theta},
#' "means" for \deqn{\mu}, "mixture" for \deqn{\pi} or "covariance" for \deqn{\Sigma}
#' @param ... additional parameters for S3 compatibility. Not used
#' @return A matrix of coefficients extracted from the PLNfit model.
#'
#' @seealso [sigma.PLNLRsemisupfit()]
#'
#' @export
#' @examples
#' data(trichoptera)
#' trichoptera <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
#' myPLN <- PLNmixture(Abundance ~ 1 + offset(log(Offset)),
#'            data = trichoptera, control = PLNmixture_param(smoothing = "none"))  %>% getBestModel()
#' coef(myPLN) ## Theta - empty here
#' coef(myPLN, type = "mixture") ## pi
#' coef(myPLN, type = "means") ## mu
#' coef(myPLN, type = "covariance") ## Sigma
coef.PLNLRsemisupfit <- function(object, type = c("main", "means", "covariance", "mixture"), ...) {
  stopifnot(isPLNLRsemisupfit(object))
  switch(match.arg(type),
         main       = object$model_par$Theta,
         means      = object$group_means,
         mixture    = object$mixtureParam,
         covariance = object$model_par$Sigma)
}

