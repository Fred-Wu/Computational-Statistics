//[[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace arma;
using Rcpp::_;

// [[Rcpp::export]]
Rcpp::List lmRcpp(const mat &X, const vec &y) {
  
  mat Q;
  mat R;
  qr_econ(Q, R, X);
  mat xTxInv = square(inv(trimatu(R)));
  vec coef = solve(trimatu(R), Q.t() * y);
  vec res = y - X * coef;
  int df = X.n_rows - X.n_cols;
  double residVar = arma::dot(res, res) / (double) df;
  vec coefStdErr = sqrt(residVar * sum(xTxInv, 1));


  return Rcpp::List::create(_["coef"] = coef,
                      _["coefStdErr"] = coefStdErr,
                      _["df"] = df);

}