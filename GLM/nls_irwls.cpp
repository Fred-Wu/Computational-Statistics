// [[Rcpp::depends(RcppArmadillo)]]
#include <Rcpp.h>
#include <RcppArmadillo.h>

using namespace arma;
using namespace Rcpp;

// [[Rcpp::plugin(cpp14)]]

// [[Rcpp::export]]

NumericVector update_parm(
    NumericVector X, 
    NumericVector y,
    NumericVector (*fp)()
    
) {
    
}