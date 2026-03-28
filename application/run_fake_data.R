library(PLNmodels)
library(pROC)
library(dplyr)
library(wordcloud2)
library(scales)
library(readxl)
library(fmsb)
library(glmnet)
source("utility.R")

# Manually selected features
feat.sel = c("C0043143", # wheelchair
             "C0025353", # mental health
             "RXNORM:897018", # dalfampridine
             "C3714552", # weakness
             "PheCode:315.2") #speech and language disorder

# Load data
load("data/Fake_MS_pt_all.Rdata")
V = eigen(cov(t(embed.features[colnames(dat.L.train$Abundance),])))$vectors; dim(V)
r = length(feat.sel)
V.sel = matrix(0, nrow=p, ncol=r)
for (j in 1:r){
  feat = feat.sel[j]
  i = which(colnames(dat.L.train$Abundance) == feat)
  V.sel[i,j] = 1
}
V.new = cbind(V.sel, V)

# Run models
n.list = c(50, 100, 150, 200)
AUC.all.list = PRAUC.all.list = n.all.list = n.visits.all.list = type.all.list = NULL
b = 1234
for (n in n.list) {
  n.pt = n
  print(n); print(n.pt)
  set.seed(b)
  
  L.id = sample(unique(d.upmc$patient_num), n.pt, replace=F)
  U.id = unique(d.upmc$patient_num)[!unique(d.upmc$patient_num) %in% L.id]
  
  d.U = d.upmc %>% filter(patient_num %in% U.id)
  d.L = d.upmc %>% filter(patient_num %in% L.id)
  
  train.id = L.id
  test.id = U.id
  
  X.U = d.U %>% dplyr::select(patient_num, date, count, PDDS) %>% 
    mutate(Intercept=1, PDDS = ifelse(PDDS < 4, 0, 1))
  Y.U = d.U %>% dplyr::select(all_of(features.nm)) %>% replace(is.na(.), 0)
  X.L.train = d.L %>% filter(patient_num %in% train.id) %>% 
    dplyr::select(patient_num, date, count, PDDS) %>% 
    mutate(Intercept=1, PDDS = ifelse(PDDS < 4, 0, 1))
  Y.L.train = d.L %>% filter(patient_num %in% train.id) %>% 
    dplyr::select(all_of(features.nm)) %>% replace(is.na(.), 0)
  X.L.test = d.U %>% filter(patient_num %in% test.id) %>% 
    dplyr::select(patient_num, date, count, PDDS) %>% 
    mutate(Intercept=1, PDDS = ifelse(PDDS < 4, 0, 1))
  Y.L.test = d.U %>% filter(patient_num %in% test.id) %>% 
    dplyr::select(all_of(features.nm)) %>% replace(is.na(.), 0)
  
  X.L.train = X.L.train[1:n,]; Y.L.train = Y.L.train[1:n,]
  
  n=nrow(X.L.train)
  N=nrow(X.U)
  p=length(features.nm)
  q=10
  
  dat.L.train = prepare_data(Y.L.train[,-c(1,2)], X.L.train); nrow(dat.L.train)
  dat.L.test = prepare_data(Y.L.test[,-c(1,2)], X.L.test); nrow(dat.L.test)
  dat.U = prepare_data(Y.U[,-c(1,2)], X.U); nrow(dat.U)
  dat.U$PDDS = 3
  dat.all.train = rbind(dat.L.train, dat.U)
  
  dat.L.train = dat.L.train[rowSums(dat.L.train$Abundance)>0,]
  dat.L.test = dat.L.test[rowSums(dat.L.test$Abundance)>0,]
  dat.U = dat.U[rowSums(dat.U$Abundance)>0,]
  n = nrow(dat.L.train); n
  N = nrow(dat.U); N
  
  ## Lasso
  x.train = cbind(log(cbind(dat.L.train$Abundance, dat.L.train$count)+1)); colnames(x.train)[ncol(x.train)] = "count"
  y.train = dat.L.train$PDDS
  x.test = cbind(log(cbind(dat.L.test$Abundance, dat.L.test$count)+1)); colnames(x.test)[ncol(x.test)] = "count"
  y.test = dat.L.test$PDSS
  
  lasso.cv = cv.glmnet(x.train, y.train,family="binomial")
  coef(lasso.cv, s = "lambda.min")
  lasso.pred = predict(lasso.cv, newx = x.test, s="lambda.min", type="response")
  ROC.lasso = ROC(dat.L.test$PDDS, lasso.pred)
  AUC.lasso = COMP_AUC(ROC.lasso$FPR, ROC.lasso$TPR); print(AUC.lasso)
  PRAUC.lasso = COMP_PRAUC(ROC.lasso$TPR, ROC.lasso$PPV); print(PRAUC.lasso)
  
  ## RF
  library(randomForest)
  library(caret)
  set.seed(0)
  rf = randomForest(y=factor(y.train), x=x.train, ntree=100)
  rf.pred = predict(rf, newdata = x.test, type="prob")[,2]
  ROC.rf = ROC(dat.L.test$PDDS, rf.pred)
  AUC.rf = COMP_AUC(ROC.rf$FPR, ROC.rf$TPR); print(AUC.rf)
  PRAUC.rf = COMP_PRAUC(ROC.rf$TPR, ROC.rf$PPV); print(PRAUC.rf)
  
  ## XGBoost
  library(xgboost)
  xgb = xgboost(data = log(cbind(dat.L.train$Abundance, dat.L.train$count)+1), label = dat.L.train$PDDS, nrounds=100, gamma=10, max_depth=100, objective = "binary:logistic", verbose=F)
  xgb.pred = predict(xgb, newdata = cbind(log(cbind(dat.L.test$Abundance, dat.L.test$count)+1)), type="prob")
  ROC.xgb = ROC(dat.L.test$PDDS, xgb.pred)
  AUC.xgb = COMP_AUC(ROC.xgb$FPR, ROC.xgb$TPR); print(AUC.xgb)
  PRAUC.xgb = COMP_PRAUC(ROC.xgb$TPR, ROC.xgb$PPV); print(PRAUC.xgb)
  
  ## supervised SCORE
  lb = list(B = matrix(-Inf, nrow=3, ncol=r+q),
            L = matrix(0, nrow=r+q, ncol=r+q),
            M = matrix(-Inf, nrow=n, ncol=r+q),
            S = matrix(0, nrow=n, ncol=r+q))
  
  pln.fixedv.sup = PLNfixedVsup(Abundance ~ Intercept + log(count+1), grouping = PDDS, data  = dat.L.train,
                                control=PLN_param(V=V.new[,1:(r+q)],
                                                  rank=r+q,
                                                  config_optim = list("algorithm" = "MMA",
                                                                      "lower_bounds" = lb,
                                                                      "L0" = sqrt(diag(1, r+q))
                                                  )))
  fixedv.sup.pred = predict(pln.fixedv.sup, dat.L.test, type="posterior")[,2]
  ROC.fixedv.sup = ROC(dat.L.test$PDDS, fixedv.sup.pred)
  AUC.fixedv.sup = COMP_AUC(ROC.fixedv.sup$FPR, ROC.fixedv.sup$TPR); print(AUC.fixedv.sup)
  PRAUC.fixedv.sup = COMP_PRAUC(ROC.fixedv.sup$TPR, ROC.fixedv.sup$PPV); print(PRAUC.fixedv.sup)
  
  ## Semisupervised fixed V
  lb = list(B = matrix(-Inf, nrow=3, ncol=r+q),
            L = matrix(0, nrow=r+q, ncol=r+q),
            M = matrix(-Inf, nrow=n+N, ncol=r+q),
            S = matrix(0, nrow=n+N, ncol=r+q))
  tau.L.train = cbind(1-dat.L.train$PDDS, dat.L.train$PDDS)
  tau.init = rbind(tau.L.train, predict(pln.fixedv.sup, dat.U, type="posterior"))
  rownames(tau.init) = NULL
  w = c(rep(0.5, n), rep(0.5, N))
  L.init = sqrt(diag(eigen(pln.fixedv.sup$model_par$Sigma)$values[1:(r+q)]))
  B.init = pln.fixedv.sup$model_par$B
  pln.fixedv.semisup = PLNfixedVsemisup(Abundance ~ Intercept + log(count+1), grouping=PDDS, weights=w, data=dat.all.train, control=PLN_param(V=V.new[,1:(r+q)],
                                                                                                                                              rank=r+q,
                                                                                                                                              config_optim = list("algorithm" = "MMA",
                                                                                                                                                                  "lower_bounds" = lb,
                                                                                                                                                                  "ftol_out" = 1e-6,
                                                                                                                                                                  "maxit_out" = 10,
                                                                                                                                                                  "L0" = L.init,
                                                                                                                                                                  "B0" = B.init,
                                                                                                                                                                  "tau0" = tau.init)))
  
  fixedv.semisup.pred = predict(pln.fixedv.semisup, dat.L.test, type="posterior")[,2]
  ROC.fixedv.semisup = ROC(dat.L.test$PDDS, fixedv.semisup.pred)
  AUC.fixedv.semisup = COMP_AUC(ROC.fixedv.semisup$FPR, ROC.fixedv.semisup$TPR); print(AUC.fixedv.semisup)
  PRAUC.fixedv.semisup = COMP_PRAUC(ROC.fixedv.semisup$TPR, ROC.fixedv.semisup$PPV); print(PRAUC.fixedv.semisup)
  
  
  AUC.all.list = c(AUC.all.list, AUC.lasso, AUC.rf, AUC.xgb, AUC.fixedv.sup, AUC.fixedv.semisup)
  PRAUC.all.list = c(PRAUC.all.list, PRAUC.lasso, PRAUC.rf, PRAUC.xgb, PRAUC.fixedv.sup, PRAUC.fixedv.semisup)
  n.all.list = c(n.all.list, rep(n.pt, 1))
  n.visits.all.list = c(n.visits.all.list, rep(n, 1))
  type.all.list = c(type.all.list, c("Lasso", "Random Forest", "XGBoost", "fixedV(sup)", "fixedV(semisup)"))
  
  save(lasso.cv, rf, xgb, PRAUC.fixedv.sup, PRAUC.fixedv.semisup, file=paste0("results/MS_models_n",n, "_b", b, ".RData"))
}

AUC.df = data.frame(AUC = AUC.all.list, PRAUC = PRAUC.all.list, n.pt = n.all.list, n.visits = n.visits.all.list, type = type.all.list)
write.csv(AUC.df, file = "results/MS_models_AUC.csv")