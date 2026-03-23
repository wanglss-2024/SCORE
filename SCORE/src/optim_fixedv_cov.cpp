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
Rcpp::List nlopt_optimize_fixedv(
    const Rcpp::List & data  , // List(Y, X, O, w)
    const Rcpp::List & params, // List(M, S, B, L, V)
    const Rcpp::List & config  // List of config values
) {
    // Conversion from R, prepare optimization
    const arma::mat & Y = Rcpp::as<arma::mat>(data["Y"]); // responses (n,p)
    const arma::mat & X = Rcpp::as<arma::mat>(data["X"]); // covariates (n,d)
    const arma::mat & O = Rcpp::as<arma::mat>(data["O"]); // offsets (n,p)
    const arma::vec & w = Rcpp::as<arma::vec>(data["w"]); // weights (n)
    const auto init_B = Rcpp::as<arma::mat>(params["B"]); // (d,q)
    const auto init_L = Rcpp::as<arma::mat>(params["L"]); // (q,q)
    const auto init_M = Rcpp::as<arma::mat>(params["M"]); // (n,q)
    const auto init_S = Rcpp::as<arma::mat>(params["S"]); // (n,q)
    const auto  V = Rcpp::as<arma::mat>(params["V"]); //


    const auto metadata = tuple_metadata(init_B, init_M, init_S);
    enum { B_ID, M_ID, S_ID }; // Names for metadata indexes

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
    auto objective_and_grad = [&metadata, &O, &X, &Y, &w, &V](const double * params, double * grad) -> double {
        const arma::mat B = metadata.map<B_ID>(params);
        //const arma::mat L = metadata.map<L_ID>(params);
        const arma::mat M = metadata.map<M_ID>(params);
        const arma::mat S = metadata.map<S_ID>(params);
        //Rcpp::Rcout << "The value of B : " << B << "\n";

        arma::mat S2 = S % S;
        arma::mat Z = O + (X * B + M) * V.t();
        arma::mat A = exp(Z + 0.5 * S2 * (V % V).t());
        arma::mat L = lambda(S2, M);
        //double objective = accu(diagmat(w) * (A - Y % Z)) + 0.5 * accu(diagmat(w) * (M % M + S2 - log(S2) - 1.));
        double objective = accu(diagmat(w) * (A - Y % Z)) - arma::as_scalar(w.t() * ElogP(L, M, S2)) - 0.5 * accu(diagmat(w) * (log(S2)+1.));


        metadata.map<B_ID>(grad) = (X.each_col() % w).t() * (A - Y) * V;
        metadata.map<M_ID>(grad) = diagmat(w) * ((A - Y) * V + M * inv(L));
        metadata.map<S_ID>(grad) = diagmat(w) * (S * inv(L) - 1. / S + A * (V % V) % S);
        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // Model and variational parameters
    arma::mat B = metadata.copy<B_ID>(parameters.data());
    arma::mat M = metadata.copy<M_ID>(parameters.data());
    arma::mat S = metadata.copy<S_ID>(parameters.data());
    arma::mat S2 = S % S;
    arma::mat L = lambda(S2, M);
    arma::mat Sigma = V * (M.t() * (M.each_col() % w) + diagmat(sum(S2.each_col() % w, 0))) * V.t() / accu(w);
    arma::mat Omega = V * inv_sympd((M.t() * (M.each_col() % w) + diagmat(sum(S2.each_col() % w, 0)))/accu(w))  * V.t() ;
    // Element-wise log-likelihood
    arma::mat Z = O + (X * B + M) * V.t();
    arma::mat A = exp(Z + 0.5 * S2 * (V % V).t());
    arma::mat loglik = arma::sum(Y % Z - A, 1) + ElogP(L, M, S2) + 0.5 * sum(log(S2) + 1., 1) + ki(Y);

    Rcpp::NumericVector Ji = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik));
    Ji.attr("weights") = w;
    return Rcpp::List::create(
        Rcpp::Named("B", B),
        Rcpp::Named("L", L),
        Rcpp::Named("M", M),
        Rcpp::Named("S", S),
        Rcpp::Named("Z", Z),
        Rcpp::Named("A", A),
        Rcpp::Named("Sigma", Sigma),
        Rcpp::Named("Omega", Omega),
        Rcpp::Named("Ji", Ji),
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
Rcpp::List nlopt_optimize_vestep_fixedv(
        const Rcpp::List & data  , // List(Y, X, O, w)
        const Rcpp::List & params, // List(M, S, V)
        const arma::mat & B,       // (d,p)
        const arma::mat & L,       // (q,q)
        const Rcpp::List & config  // List of config values
) {
    // Conversion from R, prepare optimization
    const arma::mat & Y = Rcpp::as<arma::mat>(data["Y"]); // responses (n,p)
    const arma::mat & X = Rcpp::as<arma::mat>(data["X"]); // covariates (n,d)
    const arma::mat & O = Rcpp::as<arma::mat>(data["O"]); // offsets (n,p)
    const arma::vec & w = Rcpp::as<arma::vec>(data["w"]); // weights (n)
    const auto init_M = Rcpp::as<arma::mat>(params["M"]); // (n,q)
    const auto init_S = Rcpp::as<arma::mat>(params["S"]); // (n,q)
    const auto  V = Rcpp::as<arma::mat>(params["V"]); // covinv (p,q)

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
    auto objective_and_grad = [&metadata, &O, &X, &Y, &w, &B, &L, &V](const double * params, double * grad) -> double {
        const arma::mat M = metadata.map<M_ID>(params);
        const arma::mat S = metadata.map<S_ID>(params);

        // arma::mat S2 = S % S;
        // arma::mat C = V * abs(L);
        // arma::mat Z = O + (X * B + M) * C.t();
        // arma::mat A = exp(Z + 0.5 * S2 * (C % C).t());
        // arma::mat nSigma = M.t() * (M.each_col() % w) + diagmat(w.t() * S2) ;
        // double objective = accu(diagmat(w) * (A - Y % Z)) + 0.5 * accu(diagmat(w) * (M % M + S2 - log(S2) - 1.));

        arma::mat S2 = S % S;
        //arma::mat C = V * abs(L);
        arma::mat Z = O + (X * B + M) * V.t();
        arma::mat A = exp(Z + 0.5 * S2 * (V % V).t());
        arma::mat L = lambda(S2, M);
        double objective = accu(diagmat(w) * (A - Y % Z)) - arma::as_scalar(w.t() * ElogP(L, M, S2)) - 0.5 * accu(diagmat(w) * (log(S2)+1.));

        // metadata.map<M_ID>(grad) = diagmat(w) * ((A - Y) * C + M);
        // metadata.map<S_ID>(grad) = diagmat(w) * (S - 1. / S + A * (C % C) % S);
        metadata.map<M_ID>(grad) = diagmat(w) * ((A - Y) * V + M * inv(L));
        metadata.map<S_ID>(grad) = diagmat(w) * (S * inv(L) - 1. / S + A * (V % V) % S);
        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // // Model and variational parameters
    // arma::mat M = metadata.copy<M_ID>(parameters.data());
    // arma::mat S = metadata.copy<S_ID>(parameters.data());
    // arma::mat S2 = S % S;
    // arma::mat C = V * abs(L);
    // // Element-wise log-likelihood
    // arma::mat Z = O + (X * B + M) * C.t();
    // arma::mat A = exp(Z + 0.5 * S2 * (C % C).t());
    // arma::mat loglik = arma::sum(Y % Z - A, 1) - 0.5 * sum(M % M + S2 - log(S2) - 1., 1) + ki(Y);


    // Model and variational parameters
    arma::mat M = metadata.copy<M_ID>(parameters.data());
    arma::mat S = metadata.copy<S_ID>(parameters.data());
    arma::mat S2 = S % S;
    // Element-wise log-likelihood
    arma::mat Z = O + (X * B + M) * V.t();
    arma::mat A = exp(Z + 0.5 * S2 * (V % V).t());
    arma::mat loglik = arma::sum(Y % Z - A, 1) + ElogP(lambda(S2, M), M, S2) + 0.5 * sum(log(S2) + 1., 1) + ki(Y);


    Rcpp::NumericVector Ji = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik));
    Ji.attr("weights") = w;
    return Rcpp::List::create(
        Rcpp::Named("Z") = Z,
        Rcpp::Named("M") = M,
        Rcpp::Named("S") = S,
        Rcpp::Named("Ji") = Ji);
}

// M step fixedV
// [[Rcpp::export]]
Rcpp::List nlopt_optimize_mstep_fixedv(
  const Rcpp::List & data  , // List(Y, X, O, w)
  const Rcpp::List & params, // List(B, L, V)
  const arma::mat & M,       // (n,q)
  const arma::mat & S,       // (n,q)
  const Rcpp::List & config  // List of config values
) {

    // Conversion from R, prepare optimization
    const arma::mat & Y = Rcpp::as<arma::mat>(data["Y"]); // responses (n,p)
    const arma::mat & X = Rcpp::as<arma::mat>(data["X"]); // covariates (n,d)
    const arma::mat & O = Rcpp::as<arma::mat>(data["O"]); // offsets (n,p)
    const arma::vec & w = Rcpp::as<arma::vec>(data["w"]); // weights (n)
    const auto init_B = Rcpp::as<arma::mat>(params["B"]); // (d,p)
    const auto init_L = Rcpp::as<arma::mat>(params["L"]); // (q,q)
    const auto  V = Rcpp::as<arma::mat>(params["V"]); // covinv (q,q)

    const auto metadata = tuple_metadata(init_B, init_L);
    enum { B_ID, L_ID }; // Names for metadata indexes

    auto parameters = std::vector<double>(metadata.packed_size);
    metadata.map<B_ID>(parameters.data()) = init_B;
    metadata.map<L_ID>(parameters.data()) = init_L;


    auto optimizer = new_nlopt_optimizer(config, parameters.size());
    if(config.containsElementNamed("xtol_abs")) {
        SEXP value = config["xtol_abs"];
        if(Rcpp::is<double>(value)) {
            set_uniform_xtol_abs(optimizer.get(), Rcpp::as<double>(value));
        } else {
            auto per_param_list = Rcpp::as<Rcpp::List>(value);
            auto packed = std::vector<double>(metadata.packed_size);
            set_from_r_sexp(metadata.map<B_ID>(packed.data()), per_param_list["B"]);
            set_from_r_sexp(metadata.map<L_ID>(packed.data()), per_param_list["L"]);
            set_per_value_xtol_abs(optimizer.get(), packed);
        }
    }

    // Optimize
    auto objective_and_grad = [&metadata, &O, &X, &Y, &w, &V, &M, &S](const double * params, double * grad) -> double {
        const arma::mat B = metadata.map<B_ID>(params);
        //const arma::mat L = metadata.map<L_ID>(params);
        //Rcpp::Rcout << "The value of B : " << B << "\n";

        // arma::mat S2 = S % S;
        // arma::mat C = V * abs(L);
        // arma::mat Z = O + (X * B + M) * C.t();
        // arma::mat A = exp(Z + 0.5 * S2 * (C % C).t());
        // double objective = accu(diagmat(w) * (A - Y % Z)) + 0.5 * accu(diagmat(w) * (M % M + S2 - log(S2) - 1.));

        arma::mat S2 = S % S;
        //arma::mat C = V * abs(L);
        arma::mat Z = O + (X * B + M) * V.t();
        arma::mat A = exp(Z + 0.5 * S2 * (V % V).t());
        arma::mat L = lambda(S2, M);
        double objective = accu(diagmat(w) * (A - Y % Z)) - as_scalar(w * ElogP(L, M, S2)) - 0.5 * accu(diagmat(w) * (log(S2)+1.));


        // metadata.map<B_ID>(grad) = (X.each_col() % w).t() * (A - Y) * C;
        // const arma::mat F = V.t() * ((diagmat(w) * (A - Y)).t() * (X * B + M) + (A.t() * (S2.each_col() % w)) % C);
        // metadata.map<L_ID>(grad) = diagmat(F.diag());
        metadata.map<B_ID>(grad) = (X.each_col() % w).t() * (A - Y) * V;

        return objective;
    };
    OptimizerResult result = minimize_objective_on_parameters(optimizer.get(), objective_and_grad, parameters);

    // // Model and variational parameters
    // arma::mat B = metadata.copy<B_ID>(parameters.data());
    // //arma::mat L = metadata.copy<L_ID>(parameters.data());
    // arma::mat S2 = S % S;
    // arma::mat C = V * abs(L);
    // arma::mat Sigma = C * (M.t() * (M.each_col() % w) + diagmat(sum(S2.each_col() % w, 0))) * C.t() / accu(w);
    // arma::mat Omega = C * inv_sympd((M.t() * (M.each_col() % w) + diagmat(sum(S2.each_col() % w, 0)))/accu(w))  * C.t() ;
    // // Element-wise log-likelihood
    // arma::mat Z = O + (X * B + M) * C.t();
    // arma::mat A = exp(Z + 0.5 * S2 * (C % C).t());
    // arma::mat loglik = arma::sum(Y % Z - A, 1) - 0.5 * sum(M % M + S2 - log(S2) - 1., 1) + ki(Y);

    // Model and variational parameters
    arma::mat B = metadata.copy<B_ID>(parameters.data());
    arma::mat S2 = S % S;
    arma::mat L = lambda(S2, M);
    arma::mat Sigma = V * (M.t() * (M.each_col() % w) + diagmat(sum(S2.each_col() % w, 0))) * V.t() / accu(w);
    arma::mat Omega = V * inv_sympd((M.t() * (M.each_col() % w) + diagmat(sum(S2.each_col() % w, 0)))/accu(w))  * V.t() ;
    // Element-wise log-likelihood
    arma::mat Z = O + (X * B + M) * V.t();
    arma::mat A = exp(Z + 0.5 * S2 * (V % V).t());
    //arma::mat loglik = arma::sum(Y % Z - A, 1) - 0.5 * sum(M % M + S2 - log(S2) - 1., 1) + ki(Y);
    arma::mat loglik = arma::sum(Y % Z - A, 1) + ElogP(L, M, S2) + 0.5 * sum(log(S2) + 1., 1) + ki(Y);


    Rcpp::NumericVector Ji = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(loglik));
    Ji.attr("weights") = w;
    return Rcpp::List::create(
        Rcpp::Named("B", B),
        Rcpp::Named("L", L),
        Rcpp::Named("M", M),
        Rcpp::Named("S", S),
        Rcpp::Named("Z", Z),
        Rcpp::Named("A", A),
        Rcpp::Named("Sigma", Sigma),
        Rcpp::Named("Omega", Omega),
        Rcpp::Named("Ji", Ji),
        Rcpp::Named("monitoring", Rcpp::List::create(
            Rcpp::Named("status", static_cast<int>(result.status)),
            Rcpp::Named("backend", "nlopt"),
            Rcpp::Named("iterations", result.nb_iterations)
        ))
    );
}
