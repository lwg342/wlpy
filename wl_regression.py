# %%
import scipy.interpolate as interpolate
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats as ST
from scipy import linalg as LA

# # OLS Regression Class
# Return OLS estimate of the conditional mean as a col.vector %% OLS parameter estimate


class Regression():
    """
    This defines a class for regression models and estimation
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def plot_function(self, f, x_eval = None):
        '''
        Plot either a function or a list of functions
        '''
        if x_eval == None:
            x_eval = np.linspace(min(self.X), max(self.X), num= 100)
        if isinstance(f, list):
            fig, ax = plt.subplots()
            for i in f:
                ax.plot(x_eval, i(x_eval))
        else:
            fig, ax = plt.subplots()
            ax.plot(x_eval, f(x_eval))


class OLS(Regression):
    '''
    Here we have an OLS/GLS regression
    Input: X, Y in a linear regression Y = X @ beta + epsilon
    Return: \n
        - beta_hat(intercept = 1 or 0): estimated beta
        - y_hat(intercept = 1 or 0): estimated y
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert(X.shape[0] == Y.shape[0])

    def beta_hat(self, intercept=1):
        if intercept == 1:
            beta = sm.OLS(self.Y, sm.add_constant(self.X)).fit().params.T
        elif intercept == 0:
            beta = sm.OLS(self.Y, self.X).fit().params.T
        else:
            print('Intercept error.')
        return beta

    def y_hat(self, intercept=1):
        if intercept == 1:
            m = sm.OLS(self.Y, sm.add_constant(self.X)).fit().predict()
        elif intercept == 0:
            m = sm.OLS(self.Y, self.X).fit().predict()
        return m

# %%
# -> Created on 28 October 2020
# -> Author: Weiguang Liu


class BSpline(OLS):
    """
    This is the class of B-Spline Models
    """
    import scipy.interpolate as interpolate

    def __init__(self, X, Y, n_degree=3):
        self.X = X
        self.Y = Y
        assert(X.shape[0] == Y.shape[0])
        self.n_degree = n_degree
        self.knots = np.linspace(min(self.X), max(self.X), int(len(self.X)**0.45))
        n = len(self.knots) + self.n_degree
        self.dim_basis = n

    def univariate_bspline_basis(self):
        basis_functions = [interpolate.BSpline(
            self.knots, np.eye(self.dim_basis)[i], self.n_degree) for i in range(self.dim_basis)]
        # b_eval = np.array([[basis_functions[i](x) for x in x_eval] for i in range(n)])
        return basis_functions

    def basis_function_evaluate(self, x_eval=None):
        """
        Evaluate the basis functions at given points
        """
        if x_eval == None:
            x_eval = self.X
        X_mat = np.array([i(x_eval)
                          for i in self.univariate_bspline_basis()]).transpose()
        return X_mat

    def fit(self):
        """
        Generates fit and prediction of a b-spline model
        """
        X_mat = self.basis_function_evaluate()
        _M = OLS(X_mat, self.Y)
        self.coeff = _M.beta_hat()
        self.predict = _M.y_hat()
        return self.predict



# %% The local linear regression
# We take Gaussian Kernel
# Bandwidth is choosen as 1/T^0.2
# It can be used for multidimensional case. The plot is different


def loc_poly(Y, X, X_eval):
    N = X.shape[0]
    k = np.linalg.matrix_rank(X)
    m = np.empty(len(X_eval))
    i = 0
    h = 1/(N**(1/5))
    # grid = np.arange(start=X.min(), stop=X.max(), step=np.ptp(X)/200)
    for x in X_eval:
        Xx = X - (np.ones([N, 1]))*x
        Xx1 = sm.add_constant(Xx)
        Wx = np.diag(ST.norm.pdf(Xx.T/h)[0])
        Sx = ((Xx1.T)@Wx@Xx1 + 1e-90*np.eye(k))
        m[i] = ((LA.inv(Sx)) @ (Xx1.T) @ Wx @ Y)[0]
        i = i + 1
    # plt.figure()
    # plt.scatter(X, Y)
    # plt.plot(grid, m, color= 'red')
    # plt.scatter(X, m, color='red')
    return m

# %%


def matlocl(data, x_eval):
    """
    This is a univariate local linear regression
    Get the weights at evaluation points 
    """
    n = len(data)
    ne = len(x_eval)
    h = 1.06*np.std(data)*(n ** -0.2)
    h = h/5
    m = (np.outer(x_eval, np.ones(n)) - np.outer(np.ones(ne), data))/h
    K1 = ST.norm.pdf(m)
    K2 = K1 * m
    K3 = K2 * m
    K = np.diag(K3.sum(axis=1)) @ K1 - np.diag(K2.sum(axis=1)) @ K2
    K = np.diag(1/K.sum(axis=1)) @ K
    return K

# %% Test of B-Spline

# n = 1000
# self.X = np.linspace(0, 10, n)
# y = np.sin(obs_x) * obs_x + obs_x ** 0.5 + \
#     np.random.default_rng().standard_normal(n)
# M = BSpline(obs_x, y)
# M.fit()
# plt.plot(obs_x, y)
# plt.plot(obs_x, M.predict)
# M.Y

# %%
