# %%
import scipy.interpolate as interpolate
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as ST
from scipy import linalg as LA
import torch
import pandas as pd

# %% Kernel Functions


def gaussian_pdf(x, device="cpu"):
    if device == "cpu":
        p = ST.norm.pdf(x)
    else:
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        p = 1 / np.sqrt(2 * np.pi) * torch.exp(-0.5 * (x**2))
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


class Regression:
    """
    This defines a class for regression models and estimation
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def plot_function(self, f, x_eval=None):
        """
        Plot either a function or a list of functions
        """
        if x_eval == None:
            x_eval = np.linspace(min(self.X), max(self.X), num=100)
        if isinstance(f, list):
            fig, ax = plt.subplots()
            for i in f:
                ax.plot(x_eval, i(x_eval))
        else:
            fig, ax = plt.subplots()
            ax.plot(x_eval, f(x_eval))


# %%
class OLS:
    def __init__(self, X, Y, N=None, k=None, num_y=1, device="cpu"):
        """Perform an OLS regression

        Args:
            X (array, tensor or dataframe): The covariates, of shape N*k
            Y (numpy.array): The explained variable of shape N*num_y
            N (int): sample size
            k (int): column rank of X
            num_y (int, optional): [description]. Defaults to 1.
            device (str, optional): Whether use numpy or tensor on CPU or GPU. Defaults to "cpu".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if N == None:
            N = X.shape[0]
        if k == None:
            k = X.shape[1]
        self.N = N
        self.k = k
        self.device = device
        self.isDF = type(X) == pd.DataFrame

        if self.isDF:
            xcol = X.columns
            ycol = Y.columns
            xindex = X.index
            ## Unfinished

        if device == "cpu":
            self.X = np.array(X).reshape([N, k])
            self.Y = np.array(Y).reshape([N, num_y])
            self.IX = self.add_constant(self.X, self.N)
        elif device == "cuda":
            self.X = torch.tensor(X).reshape([N, k])
            self.Y = torch.tensor(Y).reshape([N, 1])
            self.IX = self.add_constant(self.X, self.N)

    def add_constant(self, X, N):
        if self.device == "cpu":
            IX = np.concatenate([np.ones([N, 1]), X], 1)
        elif self.device == "cuda":
            IX = torch.cat([torch.ones([N, 1], device=self.device), X], 1)
        return IX

    def beta_hat(self, add_intercept=True):
        if add_intercept:
            X = self.IX
        else:
            X = self.X
        Y = self.Y
        if self.device == "cpu":
            beta = LA.inv(X.transpose() @ X) @ X.transpose() @ Y
        elif self.device == "cuda":
            V = torch.inverse(X.transpose(1, 0) @ X)
            W = X.transpose(1, 0) @ Y
            beta = torch.matmul(V, W)
        self.beta_est = beta
        return beta

    def y_hat(self, add_intercept=True):
        if add_intercept:
            X = self.IX
        else:
            X = self.X
        beta = self.beta_hat()

        y_hat = X @ beta
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
        assert X.shape[0] == Y.shape[0]
        self.n_degree = n_degree
        self.knots = np.linspace(min(self.X), max(self.X), int(len(self.X) ** 0.45))
        n = len(self.knots) + self.n_degree
        self.dim_basis = n

    def univariate_bspline_basis(self):
        basis_functions = [
            interpolate.BSpline(self.knots, np.eye(self.dim_basis)[i], self.n_degree)
            for i in range(self.dim_basis)
        ]
        # b_eval = np.array([[basis_functions[i](x) for x in x_eval] for i in range(n)])
        return basis_functions

    def basis_function_evaluate(self, x_eval=None):
        """
        Evaluate the basis functions at given points
        """
        if x_eval == None:
            x_eval = self.X
        X_mat = np.array(
            [i(x_eval) for i in self.univariate_bspline_basis()]
        ).transpose()
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
# We take Gaussian Kernel, Epanechnikov Kernel seems to give singular weighting matrix.


class LocLin(OLS):
    def __init__(
        self, X, Y, N=None, k=None, num_y=1, return_derivative=False, device="cpu"
    ):
        super().__init__(X, Y, N, k, num_y, device)
        self.return_derivative = return_derivative

    def fit(self, xe: np.array):
        """Fit a local polynomial of order p at evaluation points xe.

        See page 298 of Fan and Gijbels.

        This method fits a local polynomial of order p at the evaluation points xe, using the data stored in the object. The method first constructs a design matrix IXE based on the evaluation points xe and the data X stored in the object. It then constructs a weight matrix W based on a Gaussian kernel function and the bandwidth h. Finally, it computes the beta estimate using the weighted least squares method.

        Args:
            xe (np.array): Evaluation point.

        Returns:
            y_prediction (float): The predicted value of the response variable.
        """
        XE = self.X - xe[np.newaxis, :]
        IXE = self.add_constant(XE, self.N)

        h = 1 / (self.N ** (1 / (4 + self.k))) * 1.06 * self.X.std(0)
        W = np.diag(gaussian_pdf(XE / h).prod(1) / h.prod())
        beta_estimate = LA.inv(IXE.T @ W @ IXE) @ IXE.T @ W @ self.Y

        y_prediction = beta_estimate[0]
        return y_prediction

    def vec_fit(self, vec_xe: np.array):
        """Fit a local polynomial of order p at multiple evaluation points.

        This method fits a local polynomial of order p at multiple evaluation points, using the data stored in the object. The method calls the `fit` method for each evaluation point in `vec_xe`, and returns an array of predicted values.

        Args:
            vec_xe (np.array): Evaluation points of shape L*k.

        Returns:
            y_prediction (np.array): An array of predicted values of the response variable.

        Raises:
            ValueError: If the evaluation points vec_xe have the wrong shape.
        """
        if vec_xe.shape[1] != self.k:
            raise ValueError("Evaluation points have the wrong shape.")

        L = vec_xe.shape[0]
        y_prediction = np.array([self.fit(vec_xe[i]) for i in range(L)])
        return y_prediction

    # Perhaps we can speed up by using einsum
    def _temp(self, xe: np.array):
        XE = np.einsum("nk,l -> lnk", self.X, np.ones(L)) - np.einsum(
            "lk,n -> lnk", xe, np.ones(self.N)
        )
        IXE = np.concatenate([np.ones([20, 1000, 1]), XE], 2)
        h = 1 / (self.N ** (1 / (4 + self.k))) * 1.06 * self.X.std(0)
        W = np.einsum("lnk -> ln", Epanechnikov(XE / h)) / h.prod()
        W = np.einsum("ln, nk -> lnk", W, np.eye(self.N))
        # beta_hat = self.N * np.inv(IXE.T@W@IXE/self.N) @ IXE.T@W@self.Y
        return None


# %% Test case of Local Linear
# n = 1000
# k = 3
# x = np.random.normal(0, 1, [n, k])
# param = np.arange(k).reshape(-1, 1) + 2.0
# print(param)
# y = x @ param
# #  + np.random.normal(0, 0.001, [n, 1])
# import matplotlib.pyplot as plt

# # plt.plot(x, y, "o")

# xe = np.array([2.0, 3.0, 4.0])
# XE = x - np.outer(np.ones(n), xe)
# IXE = np.concatenate([np.ones([n, 1]), XE], 1)
# h = 1 / (n ** (1 / (4 + k))) * 1.06 * x.std(0)
# W = np.diag(gaussian_pdf(XE / h).prod(1) / (h.prod()))
# beta_estimate = LA.inv(IXE.T @ W @ IXE) @ IXE.T @ W @ y
# print(beta_estimate)

# mm = LocLin(x, y)
# # y_prediction = [mm.fit(xe) for xe in x]
# # plt.plot(y, y_prediction, "o")
# vec_xe = x
# y_prediction = mm.vec_fit(vec_xe)
# plt.plot(y, y_prediction, "o")

# %%


def matlocl(data, x_eval):
    """
    This is a univariate local linear regression
    Get the weights at evaluation points
    """
    n = len(data)
    ne = len(x_eval)
    h = 1.06 * np.std(data) * (n**-0.2)
    h = h / 5
    m = (np.outer(x_eval, np.ones(n)) - np.outer(np.ones(ne), data)) / h
    K1 = ST.norm.pdf(m)
    K2 = K1 * m
    K3 = K2 * m
    K = np.diag(K3.sum(axis=1)) @ K1 - np.diag(K2.sum(axis=1)) @ K2
    K = np.diag(1 / K.sum(axis=1)) @ K
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
