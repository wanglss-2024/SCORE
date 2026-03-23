#pragma once

#include <RcppArmadillo.h>
//#include <RcppEigen.h>

inline arma::vec logfact(arma::mat y) {
    y.replace(0., 1.);
    return sum(y % log(y) - y + log(8 * pow(y, 3) + 4 * pow(y, 2) + y + 1. / 30.) / 6. + std::log(M_PI) / 2., 1);
}

inline arma::mat logfact_mat(arma::mat y) {
    y.replace(0., 1.);
    return y % log(y) - y + log(8 * pow(y, 3) + 4 * pow(y, 2) + y + 1. / 30.) / 6. + std::log(M_PI) / 2.;
}

inline arma::vec ki(arma::mat y) {
    arma::uword p = y.n_cols;
    return -logfact(std::move(y)) + 0.5 * double(p) ;
}

inline arma::mat lambda(arma::mat S2, arma::mat M) {
  arma::mat L; L.zeros(S2.n_cols, S2.n_cols);
  int n = S2.n_rows;
  for (int i = 0; i < n; i++) {
    arma::mat temp = diagmat(S2.row(i)) + M.row(i).t() * M.row(i);
    L = L + temp;
  }
  L = L / n;
  return L;
}

inline arma::vec ElogP(arma::mat L, arma::mat M, arma::mat S2){
  arma::mat Linv = inv(L);
  double logdL = log_det_sympd(L);
  arma::vec ElogP; ElogP.zeros(S2.n_rows);
  int n = S2.n_rows;
  for (int i = 0; i < n; i++) {
    ElogP(i) = - 0.5 * (logdL + as_scalar(M.row(i) * Linv * M.row(i).t()) + trace(Linv * diagmat(S2.row(i))));
  }
  return ElogP;
}
