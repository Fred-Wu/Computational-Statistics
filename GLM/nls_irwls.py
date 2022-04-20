
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as ola
from jax import config
config.update('jax_enable_x64', True)


def update_parm(X, y, fun, dfun, parm, theta, wt):
    len_y = len(y)
    mean_fun = fun(X, parm)

    if (type(wt) == bool):
        if (wt):
            var_fun = np.exp(theta * np.log(mean_fun))
            sqrtW = 1 / np.sqrt(var_fun ** 2)
        else:
            sqrtW = 1
    else:
        sqrtW = wt
        
    gradX = dfun(x, parm)
    weighted_X = sqrtW.reshape(len_y, 1) * gradX
    z = gradX @ parm + (y - mean_fun)
    weighted_z = sqrtW * z
    qr_gradX = ola.qr(weighted_X, mode="economic")
    Q = qr_gradX[0]
    R = qr_gradX[1]
    new_parm = ola.solve_triangular(R, np.dot(Q.T, weighted_z))
    
    return new_parm


def nls_irwls(X, y, fun, dfun, init, theta = 1, tol = 1e-8, maxiter = 500):

    old_parm = init
    iter = 0

    while (iter < maxiter):
        new_parm = update_parm(X, y, fun, dfun, parm=old_parm, theta=theta, wt=True)
        parm_diff = np.max(np.abs(new_parm - old_parm) / np.abs(old_parm))
        if (parm_diff < tol) :
            break
        else:
            old_parm = new_parm
            iter += 1
            print(parm_diff)
            print(new_parm)

    if (iter == maxiter):
        print("The algorithm failed to converge")
    else:
        return {"Estimated coefficient": new_parm}


x = np.array([0.25, 0.5, 0.75, 1, 1.25, 2, 3, 4, 5, 6, 8])
y = np.array([2.05, 1.04, 0.81, 0.39, 0.30, 0.23, 0.13, 0.11, 0.08, 0.10, 0.06])

def model(x, W):
    comp1 = jnp.exp(W[0])
    comp2 = jnp.exp(-jnp.exp(W[1]) * x)
    comp3 = jnp.exp(W[2])
    comp4 = jnp.exp(-jnp.exp(W[3]) * x)
    return comp1 * comp2 + comp3 * comp4

def dModel(x, W):
    e1 = np.exp(W[1])
    e2 = np.exp(W[3])
    e5 = np.exp(-(x * e1))
    e6 = np.exp(-(x * e2))
    e7 = np.exp(W[0])
    e8 = np.exp(W[2])
    b1 = e5 * e7
    b2 = -(x * e5 * e7 * e1) 
    b3 = e6 * e8 
    b4 = -(x * e6 * e8 * e2)

    return np.array([b1, b2, b3, b4]).T

init = np.array([0.69, 0.69, -1.6, -1.6])
model_grad = jax.jit(jax.jacfwd(model, argnums=1))

nls_irwls(x, y, model, dModel, init=init, theta=1, tol=1e-8)


init=np.matrix([1, 2, 3, 4])
