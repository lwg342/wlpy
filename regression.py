# %%
import scipy.interpolate as interpolate
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats as ST
from scipy import linalg as LA
import torch
# %% Kernel Functions


def gaussian_pdf(x, device="cpu"):
    if device == "cpu":
        p = ST.norm.pdf(x)
    else:
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        p = 1/np.sqrt(2 * np.pi)*torch.exp(-0.5*(x**2))
    return p


def Epanechnikov(z: np.array) -> np.array:
    """Generate Epanechnikov Kernel evaluation at z

    Args:
        z (np.array): The locations at which to evaluate the Epanechnikov function

    Returns:
        np.array: Return E(z)
    """
    K = ((1 - z**2).clip(0)) * 0.75
    return K


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



class OLS():
    def __init__(self, X, Y, N, k, device = "cpu"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.N = N
        self.k = k
        self.device = device
        if device == "cpu":
            self.X = np.array(X).reshape([N, k])
            self.Y = np.array(Y).reshape([N, 1])
            self.IX = self.add_constant(self.X, self.N)
        elif device == "cuda":
            self.X = torch.tensor(X).reshape([N, k])
            self.Y = torch.tensor(Y).reshape([N, 1])
            self.IX = self.add_constant(self.X, self.N)
            
    def add_constant(self, X, N):
        if self.device == "cpu":
            IX = np.concatenate([np.ones([N, 1]), X], 1)
        elif self.device == "cuda":
            IX = torch.cat(
                [torch.ones([N, 1], device=self.device), X], 1)
        return IX
    
    def beta_hat(self, add_intercept=True):
        if add_intercept:
            X = self.IX
        else:
            X = self.X
        Y = self.Y
        if self.device == "cpu":
            beta = LA.inv(X.transpose()@X)@X.transpose()@Y
        elif self.device == "cuda":
            V = torch.inverse(X.transpose(1, 0)@X)
            W = X.transpose(1, 0)@Y
            beta = torch.matmul(V, W)
        self.beta_est = beta
        return beta

    def y_hat(self, add_intercept=True):
        if add_intercept:
            X = self.IX
        else:
            X = self.X
        beta = self.beta_hat()
        
        y_hat = X@beta
        return y_hat

# %%
# -> Created on 28 October 2020
# -> Author: Weiguang Liu


class BSpline(OLS):
    """
    This is the class of B-Spline Models
    """

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


class LocLin(OLS):
    def __init__(self,X, Y, N, k, return_derivative=False, device='cpu'):
        super().__init__(X, Y, N, k, device)
        self.return_derivative = return_derivative
        
    def fit(self, xe: np.array):
        """Fit loc-polynomial of order p at evaluation points xe

        See page 298 of Fan and Gijbels
        Args:
            xe (np.array): evaluation points of shape L*k
        """
        XE = self.X - np.outer(np.ones(self.N),xe)
        IXE = self.add_constant(XE, self.N)
        
        h = 1/(self.N**(1/(4+self.k))) * 1.06 * self.X.std(0)
        # W = np.diag(gaussian_pdf(XE/h).prod(1)/h.prod())
        W = np.diag(gaussian_pdf(XE/h).prod(1)/h.prod())
        beta_hat =LA.inv(IXE.T@W@IXE) @ IXE.T@W@self.Y
        return beta_hat
    
    def vec_fit(self, vec_xe, L):
        list_beta_hat = np.concatenate(
            [self.fit(xe).tolist() for xe in vec_xe], axis = 1)
        return list_beta_hat            

    # Perhaps we can speed up by using einsum 
    def _temp(self, xe: np.array, L):
        XE = np.einsum("nk,l -> lnk" , self.X, np.ones(L)) - np.einsum("lk,n -> lnk", xe, np.ones(self.N))
        IXE = np.concatenate([np.ones([20, 1000, 1]), XE], 2)
        h = 1/(self.N**(1/(4+self.k))) * 1.06 * self.X.std(0)
        W = np.einsum('lnk -> ln', Epanechnikov(XE/h))/h.prod()
        W = np.einsum('ln, nk -> lnk', W, np.eye(self.N))
        # beta_hat = self.N * np.inv(IXE.T@W@IXE/self.N) @ IXE.T@W@self.Y
        
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

