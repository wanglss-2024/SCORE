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
# 
# N; n; p; q; type

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
Y = apply(Z, c(1,2), FUN=function(z){rpois(1, exp(z))})

X.l = X[1:n,]; Y.l = Y[1:n,]; dat.l = prepare_data(Y.l, X.l)
X.u = X[(n+1):(n+N),]; Y.u = Y[(n+1):(n+N),]; dat.u = prepare_data(Y.u, X.u)
Z.l = Z[1:n,]; Z.u = Z[(n+1):(n+N),]

# Xiaoou's two-step SVD
sv = svd(Y.l); #plot(sv$d)
eta = 0.01
threshold = (1 + eta) * sqrt(n)
#index = 1 : max(which(sv$d >= threshold), q + 1); print(index)
index = 1:(q+1)
Y.l.tild = sv$u[,index] %*% diag(sv$d[index]) %*% t(sv$v[,index]); hist(Y.l.tild); min(Y.l.tild)
Y.l.tild[Y.l.tild<=0.001] = 0.001
Y.l.tild.log = log(Y.l.tild)
B.hat = sapply(1:p, FUN=function(j){ coefficients(lm(Y.l.tild.log[,j] ~ 0 + X.l))})
mu.hat = sapply(1:p, FUN=function(j){ predict(lm(Y.l.tild.log[,j] ~ 0 + X.l))})
V.hat = eigen(cov(Y.l.tild.log - mu.hat))$vectors[,1:q]
L.hat = eigen(cov(Y.l.tild.log - mu.hat))$values[1:q]/sqrt(n)
#Sigma.hat = 1/n * V.hat %*% diag(L.hat) %*% t(V.hat)
L.svd = sqrt(sum((L-diag(L.hat))^2)/sum(L^2)); L.svd
B.svd = sqrt(sum((B.hat[-1,] %*% V.hat - t(B[,-1]))^2)/sum(B[,-1]^2))


# original lowrank PLN
pln.pca.sup = PLNPCA(Abundance ~ X1 + X2, grouping = X3, data  = dat.l, ranks=q)
pln.lr.sup = getBestModel(pln.pca.sup)
V.hat = eigen(pln.lr.sup$model_par$Sigma)$vectors[,1:q]
sum((pln.lr.sup$model_par$B %*% V.hat - t(B))^2)
B.lr.sup = sqrt(sum((pln.lr.sup$model_par$B[-1,] %*% V.hat - t(B[,-1]))^2) / sum(B[,-1]^2))
L.lr.sup = sqrt(sum((diag(eigen(pln.lr.sup$model_par$Sigma)$values[1:q]) - L)^2) / sum(L^2))
lr.sup.pred = predict(pln.lr.sup, dat.u, type="posterior")[,2]
ROC.lr.sup = ROC(dat.u[,"X3"], lr.sup.pred)
AUC.lr.sup = COMP_AUC(ROC.lr.sup$FPR, ROC.lr.sup$TPR); AUC.lr.sup
PRAUC.lr.sup = COMP_PRAUC(ROC.lr.sup$TPR, ROC.lr.sup$PPV)
Z.lr.sup = predict(pln.lr.sup, dat.u, "latent")
cos.lr.sup = mean(sapply(1:N, function(i){cosine(Z.u[i,], Z.lr.sup[i,])}))

# original lowrank PLN with fixedV
lb = list(B = matrix(-Inf, nrow=3, ncol=p),
          L = matrix(0, nrow=q, ncol=q), 
          M = matrix(-Inf, nrow=n, ncol=q),
          S = matrix(0, nrow=n, ncol=q))
pln.fixedv.sup = PLNfixedVsup(Abundance ~ X1 + X2, grouping = X3, data  = dat.l,
                              control=PLN_param(V=V.tild, rank=q,
                                                config_optim = list("algorithm" = "MMA",
                                                                    "lower_bounds" = lb,
                                                                    "L0" = L)))

B.fixedv.sup = sqrt(sum((pln.fixedv.sup$model_par$B[-1,] %*% V.tild - t(B[,-1]))^2) / sum(B[,-1]^2))
L.fixedv.sup = sqrt(sum((pln.fixedv.sup$model_par$L - L)^2)/sum(L^2))
fixedv.sup.pred = predict(pln.fixedv.sup, dat.u, type="posterior")[,2]
ROC.fixedv.sup = ROC(dat.u[,"X3"], fixedv.sup.pred)
AUC.fixedv.sup = COMP_AUC(ROC.fixedv.sup$FPR, ROC.fixedv.sup$TPR)
PRAUC.fixedv.sup = COMP_PRAUC(ROC.fixedv.sup$TPR, ROC.fixedv.sup$PPV)
Z.fixedv.sup = predict(pln.fixedv.sup, dat.u, "latent")
cos.fixedv.sup = mean(sapply(1:N, function(i){cosine(Z.u[i,], Z.fixedv.sup[i,])}))


save(B.fixedv.sup, L.fixedv.sup, AUC.fixedv.sup, PRAUC.fixedv.sup, cos.fixedv.sup,
     B.lr.sup, L.lr.sup, AUC.lr.sup, PRAUC.lr.sup, cos.lr.sup,
     B.svd, L.svd,
     file=paste0("results_0319_new/N", N, "_n", n, "_p", p, "_q", q, "_b", b, "_type", type ,"_others.RData"))
