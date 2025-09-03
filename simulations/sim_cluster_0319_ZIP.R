args = commandArgs(trailingOnly=T)
n = as.integer(args[1])
N = as.integer(args[2])
p = as.integer(args[3])
q = as.integer(args[4])
b = as.integer(args[5])
type = as.integer(args[6])

set.seed(b)

library(PLNmodels)
setwd("~/PLNSemisupervised/simulations")
source("utility.R")
library(MASS)
library(lsa)
ar_cor <- function(n, rho) {
  exponent <- abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - 
                    (1:n - 1))
  rho^exponent
}

# set.seed(0)
# N = 1000; n = 100
# p = 400; q = 20
# type = 1

N; n; p; q; type

if (type == 1) {
  # medium
  cor.mat = ar_cor(p, rho=0.5)  # pxp
  ev = eigen(cor.mat)
  V = ev$vectors
  #V.tild = sqrt(p) * V[,1:q]
  V.tild = V[,1:q]
}
if (type == 2) {
  # sparse
  #V = V.tild = sqrt(p)/q * (eigen(bdiag(lapply(1:q, function(j){matrix(1, nrow=p/q, ncol=p/q)})))$vectors[,1:q] != 0)
  #V = V.tild = 1*(eigen(bdiag(lapply(1:q, function(j){matrix(1, nrow=p/q, ncol=p/q)})))$vectors[,1:q] != 0)
  V = V.tild = eigen(bdiag(lapply(1:q, function(j){matrix(1, nrow=p/q, ncol=p/q)})))$vectors[,1:q]
}
if (type == 3) {
  # dense
  #V = V.tild = sqrt(p/q) * eigen(ar_cor(p, rho=0.5))$vectors[,1:q]
  V = V.tild = eigen(ar_cor(p, rho=0.9))$vectors[,1:q]
}

# # check column norms
# colSums((sqrt(q/p)*V.tild)^2)

# Generate L
var.mat = diag(4,q)
L = sqrt(var.mat) %*% ar_cor(q, rho=0.1) %*% sqrt(var.mat)
W = mvrnorm(n+N, rep(0,q), Sigma=L)
#B = cbind(rep(0, q), 0.2, c(rep(0.8, 4), rep(0.4, 4), rep(0, q-8)))
B = cbind(rep(0, q), 0.2, rep(0.8, q))
U = log(rpois(n+N, 2)*10 + 1)
group = rbinom(n+N, 1, 0.4)
X = cbind(1, U, group); colnames(X) = c("X1", "X2", "X3")
# (X %*% t(B) %*% t(V.tild))[1:3, 1:3]
# (W %*% t(V.tild))[1:3, 1:3]
Z = (X %*% t(B) + W) %*% t(V.tild)
#Y = apply(Z, c(1,2), FUN=function(z){rpois(1, exp(z))})
Y = apply(Z, c(1,2), FUN=function(z){rbinom(1, 1, 0.95) * rpois(1, exp(z))})


X.l = X[1:n,]; Y.l = Y[1:n,]; dat.l = prepare_data(Y.l, X.l)
X.u = X[(n+1):(n+N),]; Y.u = Y[(n+1):(n+N),]; dat.u = prepare_data(Y.u, X.u)
Z.l = Z[1:n,]; Z.u = Z[(n+1):(n+N),]

###-------------- Supervised -------------------
lb = list(B = matrix(-Inf, nrow=3, ncol=q),
          L = matrix(-Inf, nrow=q, ncol=q), 
          M = matrix(-Inf, nrow=n, ncol=q),
          S = matrix(0, nrow=n, ncol=q))
# supervised fixed V
pln.fixedv.sup = PLNfixedVsup(Abundance ~ X1 + X2, grouping = X3, data  = dat.l,
                              control=PLN_param(V=V.tild, rank=q,
                                                config_optim = list("algorithm" = "MMA",
                                                                    "lower_bounds" = lb,
                                                                    "L0" = L)))

B.fixedv.sup = sqrt(sum((pln.fixedv.sup$model_par$B[-1,] - t(B[,-1]))^2) / sum(B[,-1]^2))
L.fixedv.sup = sqrt(sum((pln.fixedv.sup$model_par$L - L)^2)/sum(L^2))

#B.fixedv.sup; L.fixedv.sup

fixedv.sup.pred = predict(pln.fixedv.sup, dat.u, type="posterior")[,2]
ROC.fixedv.sup = ROC(dat.u[,"X3"], fixedv.sup.pred)
AUC.fixedv.sup = COMP_AUC(ROC.fixedv.sup$FPR, ROC.fixedv.sup$TPR)
PRAUC.fixedv.sup = COMP_PRAUC(ROC.fixedv.sup$TPR, ROC.fixedv.sup$PPV)

Z.fixedv.sup = predict(pln.fixedv.sup, dat.u, "latent")
cos.fixedv.sup = mean(sapply(1:N, function(i){cosine(Z.u[i,], Z.fixedv.sup[i,])}))

###-------------- Unsupervised -------------------
lb = list(B = matrix(-Inf, nrow=3, ncol=q),
          L = matrix(0, nrow=q, ncol=q), 
          M = matrix(-Inf, nrow=N, ncol=q),
          S = matrix(0, nrow=N, ncol=q))
# Unsupervised fixed V
pln.fixedv.unsup = PLNfixedVunsup(Abundance ~ X1 + X2, data  = dat.u, control=PLN_param(V=V.tild, rank=q, 
                                                                                        config_optim = list("algorithm" = "MMA",
                                                                                                            "lower_bounds" = lb,
                                                                                                            "ftol_out" = 1e-6,
                                                                                                            "maxit_out" = 10,
                                                                                                            "L0" = L,
                                                                                                            "tau0" = matrix(0.5, ncol=2, nrow=N))))
B.fixedv.unsup = sqrt(sum((pln.fixedv.unsup$model_par$B[-1,] - t(B[,-1]))^2) / sum(B[,-1]^2))
L.fixedv.unsup = sqrt(sum((pln.fixedv.unsup$model_par$L - L)^2)/sum(L^2))

#B.fixedv.unsup; L.fixedv.unsup

fixedv.unsup.pred = predict(pln.fixedv.unsup, dat.u, type="posterior")[,2]
ROC.fixedv.unsup = ROC(dat.u[,"X3"], fixedv.unsup.pred)
AUC.fixedv.unsup = COMP_AUC(ROC.fixedv.unsup$FPR, ROC.fixedv.unsup$TPR)
PRAUC.fixedv.unsup = COMP_PRAUC(ROC.fixedv.unsup$TPR, ROC.fixedv.unsup$PPV)

Z.fixedv.unsup = predict(pln.fixedv.unsup, dat.u, "latent")
cos.fixedv.unsup = mean(sapply(1:N, function(i){cosine(Z.u[i,], Z.fixedv.unsup[i,])}))

###-------------- Semisupervised -------------------
Y.all = rbind(Y.l, Y.u)
X.all = rbind(X.l, X.u)
X.all[(n+1):(n+N),3] = 3
dat.all = prepare_data(Y.all, X.all)
lb = list(B = matrix(-Inf, nrow=3, ncol=q),
          L = matrix(0, nrow=q, ncol=q), 
          M = matrix(-Inf, nrow=n+N, ncol=q),
          S = matrix(0, nrow=n+N, ncol=q))
tau.l = cbind(1-X.l[,3], X.l[,3])
#tau.init = rbind(tau.l, predict(pln.fixedv.sup, dat.u, type="posterior"))
tau.init = rbind(cbind(1-X.l[,3], X.l[,3]), cbind(1-X.u[,3], X.u[,3]))
rownames(tau.init) = NULL
# Semisupervised fixed V
pln.fixedv.semisup = PLNfixedVsemisup(Abundance ~ X1 + X2, grouping=X3, data  = dat.all, control=PLN_param(V=V.tild, 
                                                                                                           rank=q, 
                                                                                                           config_optim = list("algorithm" = "MMA",
                                                                                                                               "lower_bounds" = lb,
                                                                                                                               "ftol_out" = 1e-6,
                                                                                                                               "maxit_out" = 10,
                                                                                                                               "L0" = L,
                                                                                                                               "B0" = pln.fixedv.sup$model_par$B,
                                                                                                                               "tau0" = tau.init)))

B.fixedv.semisup = sqrt(sum((pln.fixedv.semisup$model_par$B[-1,] - t(B[,-1]))^2) / sum(B[,-1]^2))
L.fixedv.semisup = sqrt(sum((pln.fixedv.semisup$model_par$L - L)^2)/sum(L^2))

#B.fixedv.semisup; L.fixedv.semisup

fixedv.semisup.pred = predict(pln.fixedv.semisup, dat.u, type="posterior")[,2]
ROC.fixedv.semisup = ROC(dat.u[,"X3"], fixedv.semisup.pred)
AUC.fixedv.semisup = COMP_AUC(ROC.fixedv.semisup$FPR, ROC.fixedv.semisup$TPR)
PRAUC.fixedv.semisup = COMP_PRAUC(ROC.fixedv.semisup$TPR, ROC.fixedv.semisup$PPV)

Z.fixedv.semisup = predict(pln.fixedv.semisup, dat.u, "latent")
cos.fixedv.semisup = mean(sapply(1:N, function(i){cosine(Z.u[i,], Z.fixedv.semisup[i,])}))

save(B.fixedv.sup, L.fixedv.sup, AUC.fixedv.sup, PRAUC.fixedv.sup, cos.fixedv.sup,
     B.fixedv.unsup, L.fixedv.unsup, AUC.fixedv.unsup, PRAUC.fixedv.unsup, cos.fixedv.unsup,
     B.fixedv.semisup, L.fixedv.semisup, AUC.fixedv.semisup, PRAUC.fixedv.semisup, cos.fixedv.semisup,
     file=paste0("results_0319_new/ZIP05_N", N, "_n", n, "_p", p, "_q", q, "_b", b, "_type", type ,".RData"))
