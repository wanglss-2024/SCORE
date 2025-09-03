#include "RcppArmadillo.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(nloptr)]]

#include "nlopt_wrapper.h"
#include "packing.h"
#include "utils.h"

// ---------------------------------------------------------------------------------------
// Rank-constrained covariance

// Rank (q) is already determined by param dimensions ; not passed anywhere

// [[Rcpp::export]]
Rcpp::List nlopt_optimize_fixedv_unsup(
    const Rcpp::List & data  , // List(Y, X0, X1, O, w, w0, w1)
    const Rcpp::List & params, // List(M, S, B, L, V)
    const Rcpp::List & config  // List of config values
) {
    // Conversion from R, prepare optimization
    const arma::mat & Y  = Rcpp::as<arma::mat>(data["Y"]); // responses (n,p)
    const arma::mat & X0 = Rcpp::as<arma::mat>(data["X0"]); // covariates (n,d)
    const arma::mat & X1 = Rcpp::as<arma::mat>(data["X1"]); // covariates (n,d)
    const arma::mat & O  = Rcpp::as<arma::mat>(data["O"]); // offsets (n,p)
    const arma::vec & w1 = Rcpp::as<arma::vec>(data["w1"]); // weights (n)
    const arma::vec & w0 = Rcpp::as<arma::vec>(data["w0"]);
    const arma::vec & w  = Rcpp::as<arma::vec>(data["w"]);
    const auto init_B = Rcpp::as<arma::mat>(params["B"]); // (d,p)
    const auto init_L = Rcpp::as<arma::mat>(params["L"]); // (q,q)
    const auto init_M = Rcpp::as<arma::mat>(params["M"]); // (n,q)
    const auto init_S = Rcpp::as<arma::mat>(params["S"]); // (n,q)
    const auto  V = Rcpp::as<arma::mat>(params["V"]); //


    const auto metadata = tuple_metadata(init_B, init_L, init_M, init_S);
    enum { B_ID, L_ID, M_ID, S_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<B_ID>(parameters.data()) = init_B;
    //metadata.map<L_ID>(parameters.data()) = init_L;
    metadata.map<M_ID>(parameters.data()) = init_M;
    metadata.map<S_ID>(parameters.data()) = init_S;

    auto optimizer = new_nlopt_optimizer(config, parameters.size());
    if(config.containsElementNamed("xtol_abs")) {
        SEXP value = config["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto per_param_list = Rcpp::as<Rcpp::List>(value);
            auto packed = std::vector<double>(metadata.packed_size);
            set_from_r_sexp(metadata.map<B_ID>(packed.data()), per_param_list["B"]);
            //set_from_r_sexp(metadata.map<L_ID>(packed.data()), per_param_list["L"]);
            set_from_r_sexp(metadata.map<M_ID>(packed.data()), per_param_list["M"]);
            set_from_r_sexp(metadata.map<S_ID>(packed.data()), per_param_list["S"]);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    if (config.containsElementNamed("lower_bounds")) {
      SEXP value = config["lower_bounds"];
      auto per_param_list = Rcpp::as<Rcpp::List>(value);
      auto packed = std::vector<double>(metadata.packed_size);
      set_from_r_sexp(metadata.map<B_ID>(packed.data()), per_param_list["B"]);
      //set_from_r_sexp(metadata.map<L_ID>(packed.data()), per_param_list["L"]);
      set_from_r_sexp(metadata.map<M_ID>(packed.data()), per_param_list["M"]);
      set_from_r_sexp(metadata.map<S_ID>(packed.data()), per_param_list["S"]);
      set_lower_bounds(optimizer.get(), packed);
    }

    // Optimize
    auto objective_and_grad = [&metadata, &O, &X0, &X1, &Y, &w, &w0, &w1, &V](const double * params, double * grad) -> double {
        const arma::mat B = metadata.map<B_ID>(params);
        //const arma::mat L = metadata.map<L_ID>(params);
        const arma::mat M = metadata.map<M_ID>(params);
        const arma::mat S = metadata.map<S_ID>(params);

        //Rcpp::Rcout << "The value of B : " << B << "\n";

        arma::mat S2 = S % S;
        //arma::mat C = V * abs(L);
        arma::mat Z0 = (X0 * B + M) * V.t();
        arma::mat Z1 = (X1 * B + M) * V.t();
        arma::mat A0 = exp(Z0 + 0.5 * S2 * (V % V).t());
        arma::mat A1 = exp(Z1 + 0.5 * S2 * (V % V).t());
        arma::mat L = lambda(S2, M);
        //double objective = accu(diagmat(w % w1) * (A1 - Y % Z1)) + accu(diagmat(w % w0) * (A0 - Y % Z0)) + 0.5 * accu(diagmat(w) * (M % M + S2 - log(S2) - 1.));
        double objective = accu(diagmat(w % w1) * (A1 - Y % Z1)) + accu(diagmat(w % w0) * (A0 - Y % Z0)) - arma::as_scalar(w.t() * ElogP(L, M, S2)) - 0.5 * accu(diagmat(w) * (log(S2)+1.));
        // Rcpp::Rcout << "The value of objective : " << objective << "\n";

        metadata.map<B_ID>(grad) = (X1.each_col() % (w % w1)).t() * (A1 - Y) * V + (X0.each_col() % (w % w0)).t() * (A0 - Y) * V;
        //arma::mat F1 = V.t() * ((diagmat(w % w1) * (A1 - Y)).t() * (X0 * B + M) + (A1.t() * (S2.each_col() % (w % w1))) % C);
        //arma::mat F0 = V.t() * ((diagmat(w % w0) * (A0 - Y)).t() * (X1 * B + M) + (A0.t() * (S2.each_col() % (w % w0))) % C);
        //arma::mat F = F1 + F0;
        //metadata.map<L_ID>(grad) = diagmat(F.diag());
        metadata.map<M_ID>(grad) = diagmat(w % w1) * ((A1 - Y) * V + M * inv(L)) + diagmat(w % w0) * ((A0 - Y) * V + M * inv(L));
        metadata.map<S_ID>(grad) = diagmat(w % w1) * (S * inv(L) - 1. / S + A1 * (V % V) % S) + diagmat(w % w0) * (S * inv(L) - 1. / S + A0 * (V % V) % S);
        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // Model and variational parameters
    arma::mat B = metadata.copy<B_ID>(parameters.data());
    //arma::mat L = metadata.copy<L_ID>(parameters.data());
    arma::mat M = metadata.copy<M_ID>(parameters.data());
    arma::mat S = metadata.copy<S_ID>(parameters.data());
    arma::mat S2 = S % S;
    arma::mat L = lambda(S2, M);
    //arma::mat C = V * abs(L);
    arma::mat Sigma = V * (M.t() * (M.each_col() % w) + diagmat(sum(S2.each_col() % w, 0))) * V.t() / accu(w);
    arma::mat Omega = V * inv_sympd((M.t() * (M.each_col() % w) + diagmat(sum(S2.each_col() % w, 0)))/accu(w))  * V.t() ;
    arma::mat Z0 = (X0 * B + M) * V.t();
    arma::mat Z1 = (X1 * B + M) * V.t();
    arma::mat A0 = exp(Z0 + 0.5 * S2 * (V % V).t());
    arma::mat A1 = exp(Z1 + 0.5 * S2 * (V % V).t());
    //arma::mat loglik0 = arma::sum(Y % Z0 - A0, 1) - 0.5 * sum(M % M + S2 - log(S2) - 1., 1) + ki(Y);
    //arma::mat loglik1 = arma::sum(Y % Z1 - A1, 1) - 0.5 * sum(M % M + S2 - log(S2) - 1., 1) + ki(Y);
    arma::mat loglik0 = arma::sum(Y % Z0 - A0, 1) + ElogP(L, M, S2) + 0.5 * sum(log(S2) + 1., 1) + ki(Y);
    arma::mat loglik1 = arma::sum(Y % Z1 - A1, 1) + ElogP(L, M, S2) + 0.5 * sum(log(S2) + 1., 1) + ki(Y);

    Rcpp::NumericVector Ji0 = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik0));
    Rcpp::NumericVector Ji1 = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik1));

    return Rcpp::List::create(
        Rcpp::Named("B", B),
        Rcpp::Named("L", L),
        Rcpp::Named("M", M),
        Rcpp::Named("S", S),
        Rcpp::Named("A0", A0),
        Rcpp::Named("A1", A1),
        Rcpp::Named("Z0", Z0),
        Rcpp::Named("Z1", Z1),
        Rcpp::Named("Sigma", Sigma),
        Rcpp::Named("Omega", Omega),
        Rcpp::Named("Ji0", Ji0),
        Rcpp::Named("Ji1", Ji1),
        Rcpp::Named("monitoring", Rcpp::List::create(
            Rcpp::Named("status", static_cast<int>(result.status)),
            Rcpp::Named("backend", "nlopt"),
            Rcpp::Named("iterations", result.nb_iterations)
        ))
    );
}

// ---------------------------------------------------------------------------------------
// VE fixedV
// Covariance with fixed eigenvectors

// [[Rcpp::export]]
Rcpp::List nlopt_optimize_vestep_fixedv_unsup(
        const Rcpp::List & data  , // List(Y, X0, X1, O, w, w0, w1)
        const Rcpp::List & params, // List(M, S, V)
        const arma::mat & B,       // (d,p)
        const arma::mat & L,       // (q,q)
        const Rcpp::List & config  // List of config values
) {
    // Conversion from R, prepare optimization
    const arma::mat & Y  = Rcpp::as<arma::mat>(data["Y"]); // responses (n,p)
    const arma::mat & X0 = Rcpp::as<arma::mat>(data["X0"]); // covariates (n,d)
    const arma::mat & X1 = Rcpp::as<arma::mat>(data["X1"]); // covariates (n,d)
    const arma::mat & O  = Rcpp::as<arma::mat>(data["O"]); // offsets (n,p)
    const arma::vec & w1 = Rcpp::as<arma::vec>(data["w1"]); // weights (n)
    const arma::vec & w0 = Rcpp::as<arma::vec>(data["w0"]);
    const arma::vec & w  = Rcpp::as<arma::vec>(data["w"]);
    const auto init_M = Rcpp::as<arma::mat>(params["M"]); // (n,q)
    const auto init_S = Rcpp::as<arma::mat>(params["S"]); // (n,q)
    const auto  V = Rcpp::as<arma::mat>(params["V"]); //

    const auto metadata = tuple_metadata(init_M, init_S);
    enum { M_ID, S_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<M_ID>(parameters.data()) = init_M;
    metadata.map<S_ID>(parameters.data()) = init_S;

    auto optimizer = new_nlopt_optimizer(config, parameters.size());
    if(config.containsElementNamed("xtol_abs")) {
        SEXP value = config["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto per_param_list = Rcpp::as<Rcpp::List>(value);
            auto packed = std::vector<double>(metadata.packed_size);
            set_from_r_sexp(metadata.map<M_ID>(packed.data()), per_param_list["M"]);
            set_from_r_sexp(metadata.map<S_ID>(packed.data()), per_param_list["S"]);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    // Optimize
    auto objective_and_grad = [&metadata, &O, &X0, &X1, &Y, &w, &w0, &w1, &B, &L, &V](const double * params, double * grad) -> double {
        const arma::mat M = metadata.map<M_ID>(params);
        const arma::mat S = metadata.map<S_ID>(params);

        arma::mat S2 = S % S;
        //arma::mat C = V * abs(L);
        arma::mat Z0 = (X0 * B + M) * V.t();
        arma::mat Z1 = (X1 * B + M) * V.t();
        arma::mat A0 = exp(Z0 + 0.5 * S2 * (V % V).t());
        arma::mat A1 = exp(Z1 + 0.5 * S2 * (V % V).t());
        double objective = accu(diagmat(w % w1) * (A1 - Y % Z1)) + accu(diagmat(w % w0) * (A0 - Y % Z0)) - arma::as_scalar(w.t() * ElogP(L, M, S2)) - 0.5 * accu(diagmat(w) * (log(S2)+1.));

        metadata.map<M_ID>(grad) = diagmat(w % w1) * ((A1 - Y) * V + M * inv(L)) + diagmat(w % w0) * ((A0 - Y) * V + M * inv(L));
        metadata.map<S_ID>(grad) = diagmat(w % w1) * (S * inv(L) - 1. / S + A1 * (V % V) % S) + diagmat(w % w0) * (S * inv(L) - 1. / S + A0 * (V % V) % S);
        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // Model and variational parameters
    arma::mat M = metadata.copy<M_ID>(parameters.data());
    arma::mat S = metadata.copy<S_ID>(parameters.data());
    arma::mat S2 = S % S;
    //arma::mat C = V * abs(L);
    arma::mat Z0 = (X0 * B + M) * V.t();
    arma::mat Z1 = (X1 * B + M) * V.t();
    arma::mat A0 = exp(Z0 + 0.5 * S2 * (V % V).t());
    arma::mat A1 = exp(Z1 + 0.5 * S2 * (V % V).t());
    //arma::mat loglik0 = arma::sum(Y % Z0 - A0, 1) - 0.5 * sum(M % M + S2 - log(S2) - 1., 1) + ki(Y);
    //arma::mat loglik1 = arma::sum(Y % Z1 - A1, 1) - 0.5 * sum(M % M + S2 - log(S2) - 1., 1) + ki(Y);
    arma::mat loglik0 = arma::sum(Y % Z0 - A0, 1) + ElogP(L, M, S2) + 0.5 * sum(log(S2) + 1., 1) + ki(Y);
    arma::mat loglik1 = arma::sum(Y % Z1 - A1, 1) + ElogP(L, M, S2) + 0.5 * sum(log(S2) + 1., 1) + ki(Y);

    Rcpp::NumericVector Ji0 = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik0));
    Rcpp::NumericVector Ji1 = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik1));

    return Rcpp::List::create(
        Rcpp::Named("Z0") = Z0,
        Rcpp::Named("Z1") = Z1,
        Rcpp::Named("M") = M,
        Rcpp::Named("S") = S,
        Rcpp::Named("Ji0") = Ji0,
        Rcpp::Named("Ji1") = Ji1);
}
