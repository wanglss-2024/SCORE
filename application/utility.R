library(Matrix)
library(genlasso)
library(glmnet)
expit = function (x) {return (exp(x)/(1+exp(x)))}

ROC <- function(true, pred, cutoffs=seq(0,1,0.0001)) {
  TPR = FPR = PPV = rep(NA, length(cutoffs))
  prev = mean(true)
  for (i in 1:length(cutoffs)){
    c = cutoffs[i]
    TPR[i] = sum((pred >= c) * true) / sum(true)
    FPR[i] = sum((pred >= c) * (1-true)) / sum(1-true)
    PPV[i] = ifelse( FPR[i]+TPR[i] == 0, 1, TPR[i] * prev / (FPR[i] * (1-prev) + TPR[i] * prev))
  }
  return (list(TPR=TPR, FPR=FPR, PPV=PPV))
}

PLOT_ROC <- function(FPR, TPR){
  AUC = COMP_AUC(FPR, TPR)
  plt = ggplot() + 
    geom_path(aes(x=FPR, y=TPR), size=1.1) +
    labs(x = "FPR", y = "TPR") +
    geom_abline(slope=1, intercept=0, linetype=3, color="darkgrey") +
    xlim(0,1) +
    ylim(0,1) +
    theme_minimal() +
    ggtitle(paste0("ROC (AUC=", round(AUC,3), ")"))
  return (plt)
}

COMP_AUC <- function(FPR, TPR) {
  if (length(FPR) != length(TPR)) {print("Lengths don't match")}
  sums = 0
  for (i in 2:length(FPR)) {
    sums = sums + (TPR[i-1]+TPR[i]) * abs(FPR[i-1] - FPR[i])/2
  }
  return(sums)
}

COMP_PRAUC <- function(TPR, PPV) {
  if (length(TPR) != length(PPV)) {print("Lengths don't match")}
  sums = 0
  for (i in 2:length(TPR)) {
    sums = sums + (PPV[i]+PPV[i-1]) * abs(TPR[i] - TPR[i-1])/2
  }
  return(sums)
}

# Compute cutoff value based on specificity = 0.95
COMP_CUT = function(FPR, cutoffs=seq(0,1,0.0001), a=0.05) {
  return (cutoffs[FPR <= a][1])
}

# Compute TPR value based on specificity=1-a
COMP_TPR = function(FPR, TPR, a=0.05) {
  return (TPR[FPR <= a][1])
}

# FPR at cutoff (should be close to a)
COMP_FPR = function(FPR, a=0.05) {
  return (FPR[FPR <= a][1])
}

# Transforms vector vc into matrix with desired number of rows, dm #
VTM <- function(vc, dm){
  matrix(vc, ncol=length(vc), nrow = dm, byrow = T)
}

# Performs summations efficiently based on ranks #
sum.I <- function(yy, FUN, Yi, Vi = NULL){
  # for each element of a vector yy, sums over all Vi with (yy FUN Yi) true
  # i.e. if FUN  is '<' then sum.I(yy, '<', Yi, Vi) returns
  # for each yy the sum of all Vi with yy < Yi
  if (FUN=="<"|FUN==">=") { yy <- -yy; Yi <- -Yi}
  pos <- rank(c(yy,Yi),ties.method='f')[1:length(yy)]-rank(yy,ties.method='f')
  if (substring(FUN,2,2)=="=") pos <- length(Yi)-pos
  if (!is.null(Vi)) {
    if(substring(FUN,2,2)=="=") tmpind <- order(-Yi) else  tmpind <- order(Yi)
    Vi <- apply(as.matrix(Vi)[tmpind,,drop=F],2,cumsum)
    return(rbind(0,Vi)[pos+1,])
  } else return(pos)
}