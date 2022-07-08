# My implementation of IRWLS and 3-step GLS algorithms for non-linear model using analytic derivative

library(Deriv)


update_parm <- function(X, y, fun, dfun, parm, theta, wt) {
    
    #' wt: weights here is the square root of the weights to be used
    #' for solving weighted least square "||sqrt(W) * X - sqrt(W) * y||"
    
    len_y <- length(y)
    len_parm <- length(parm)
    
    call_arg <- c(X, as.list(parm))
    mean_fun <- do.call(fun, call_arg)
    
    # wt is either be logical or vector of weights
    if (is.logical(wt)) {
        if (wt) {
            var_fun <- exp(theta * log(mean_fun))
            sqrtW <- 1 / as.numeric(sqrt(var_fun^2))
        } else {
            sqrtW = 1
        }
        
    } else if (is.vector(wt) && is.numeric(wt)) {
        sqrtW <- wt
    }
    
    gradX <- do.call(dfun, call_arg)
    gradX <- matrix(gradX, len_y, len_parm)
    weighted_gradX <- sqrtW * gradX
    
    z <- gradX %*% parm + (y - mean_fun)
    weighted_z <- sqrtW * z

    qr_gradX <- qr(weighted_gradX)
    qr_R <- qr.R(qr_gradX)
    xTxInv <- chol2inv(qr_R)
    new_parm <- qr.coef(qr_gradX, weighted_z)
    new_parm
}


nls_irwls <- function(X, y, fun, dfun, init, theta = 1, tol = 1e-8, maxiter = 500) {
    
    #' fun: object function
    #' dfun: first derivate of the object function
    
    X <- list(as.matrix(X))
    len_y <- length(y)
    len_parm <- length(init)
    
    old_parm <- init
    
    iter = 0
    
    while (iter < maxiter) {
        
        new_parm <- update_parm(X = X, y = y, fun = fun, dfun = dfun, 
                                parm = old_parm, theta = theta, wt = TRUE)
        
        parm_diff <- max(abs(new_parm - old_parm) / abs(old_parm))
        print(parm_diff)
        
        if (parm_diff < tol) {
            break
        } else {
            old_parm <- new_parm
            iter = iter + 1
        }
    }
    if (iter == maxiter) {
        cat("The algorithm failed to converge\n")
    } else {
        call_arg <- c(X, as.list(new_parm))
        mean_fun <- do.call(fun, call_arg)
        var_fun <- exp(theta * log(mean_fun))
        W <- 1 / as.numeric(var_fun^2)
        est_sigma2 <- sum(W * (y - mean_fun)^2) / (len_y - len_parm)
        return(list(estimate = new_parm, `sigma^2` = est_sigma2))
    }
}

nls_gls <- function(X, y, fun, dfun, init, theta = 1, tol = 1e-8, maxiter = 500) {
    
    X <- list(as.matrix(X))
    len_y <- length(y)
    len_parm <- length(init)
    old_parm <- init
    outIter = 0
    
    # loop to find the OLS estimate using the initial values. 
    
    olsIter <- 0
    while (olsIter < maxiter) {
        
        new_parm <- update_parm(
            X = X, y = y, fun = fun, dfun = dfun,
            parm = old_parm, theta = theta, wt = FALSE
        )

        if (max(abs(new_parm - old_parm) / abs(old_parm)) < tol) {
            gls_old <- new_parm  # set as OLS estimate as initial value for WLS
            break
        } else {
            old_parm <- new_parm
            olsIter <- olsIter + 1
        }
    }

    # update WLS estimates and update weights
    while (outIter < maxiter) {

        # fix weights
        call_arg <- c(X, as.list(gls_old))
        mean_fun <- do.call(fun, call_arg)
        var_fun <- exp(theta * log(mean_fun))
        sqrtW <- 1 / as.numeric(sqrt(var_fun^2))
        
        # update WLS estimates with fixed weights
        old_parm <- gls_old
        inIter = 0
        while (inIter < maxiter) {
            
            new_parm <- update_parm(
                X = X, y = y, fun = fun, dfun = dfun,
                parm = old_parm, theta = theta, wt = sqrtW
            )
            
            if (max(abs(new_parm - old_parm) / abs(old_parm)) < tol) {
                gls_new <- new_parm
                break
            } else {
                old_parm <- new_parm
                inIter <- inIter + 1
            }
        }

        if (max(abs(gls_new - gls_old) / abs(gls_old)) < tol) {
            break
        } else {
            gls_old <- gls_new
            outIter = outIter + 1
        }
    }
    if (outIter == maxiter) {
        cat("The algorithm failed to converge\n")
    } else {
        call_arg <- c(X, as.list(gls_new))
        mean_fun <- do.call(fun, call_arg)
        var_fun <- exp(theta * log(mean_fun))
        W <- 1 / as.numeric(var_fun^2)
        est_sigma2 <- sum(W * (y - mean_fun)^2) / (len_y - len_parm)
        return(list(estimate = gls_new, `sigma^2` = est_sigma2))
    }

}


# Example 1

x <- c(0.25, 0.5, 0.75, 1, 1.25, 2, 3, 4, 5, 6, 8)
y <- c(2.05, 1.04, 0.81, 0.39, 0.30, 0.23, 0.13, 0.11, 0.08, 0.10, 0.06)

model <- function(x, b1, b2, b3, b4) {
    comp1 <- exp(b1)
    comp2 <- exp(-exp(b2) * x)
    comp3 <- exp(b3)
    comp4 <- exp(-exp(b4) * x)
    comp1 * comp2 + comp3 * comp4
}

parm <- c("b1", "b2", "b3", "b4")
init <- c(0.69, 0.69, -1.6, -1.6)

dModel <- Deriv(model, parm, nderiv = 1)

nls_irwls(x, y, model, dModel, init, theta = 1, tol = 1e-10)
nls_gls(x, y, model, dModel, init, theta = 1)


# Example 2
dat <- read.table("GLM/theo10.dat")
t <- dat[, 1]
y <- dat[, 2]

model2 <- function(t, b1, b2, b3) {
    D <- 5.50
    cl <- exp(b1)
    v <- exp(b2)
    ka <- exp(b3)

    comp1 <- (D * ka) / (v * (ka - cl / v))
    comp2 <- exp(-t * cl / v) - exp(-ka * t) 
    # there is an error in the homework sheet, in which the first exp() should 
    # only be applied to the first term in comp2
    
    comp1 * comp2
}

parm <- c("b1", "b2", "b3")
init <- log(c(0.05, 0.50, 0.75))

grad <- Deriv(model2, parm, nderiv = 1)

# pass derivate of the function as the function argument
# which occupies quite a bit time in evaluating the derivate
nls_irwls(t, y, model2, grad, init, theta = 0.5)
nls_gls(t, y, model2, grad, init, theta = 0.5)

microbenchmark::microbenchmark(
    nls_irwls(t, y, model2, grad, init, theta = 0.5), 
    nls_gls(t, y, model2, grad, init, theta = 0.5), 
    times = 1000L)
