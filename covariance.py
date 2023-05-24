# -> Created on 03 November 2020
# -> Author: Weiguang Liu
# %% Importing Library
from sklearn import covariance as sk_cov
import nonlinshrink as nls
import numpy as np


# %%


class Covariance:
    """
    Several Estimators for Covariance Matrix Given a Sample of Size T*N or N*p
    """

    def __init__(self, DF_TxN=np.eye(2)):
        """
        Covariance estimation from the observations X:T*N
        """
        self.sample = DF_TxN
        self.T, self.N = DF_TxN.shape
        self.sample_cov_estimate = self.sample_cov()

    def sample_cov(self):
        sample_cov_estimate = np.cov(np.array(self.sample), rowvar=False)
        return sample_cov_estimate

    def linear_shrinkage(self):
        """
        Ledoit and Wolf 2004
        """
        S_lw = sk_cov.LedoitWolf().fit(self.sample).covariance_
        return S_lw

    def nonlinear_shrinkage(self):
        S_nlshrink = nls.shrink_cov(self.sample)
        return S_nlshrink

    def network_hard_threshold(self, G, cov_mat=None):
        """
        This calculates the hard-threshold estimator using a network matrix G.
        The original estimator is cov_mat
        """
        if cov_mat == None:
            _S = self.sample_cov_estimate
        else:
            _S = cov_mat
        _S[np.where((G + np.eye(self.N)) == 0)] = 0
        return _S

    def threshold_corr(
        self, cov_mat=None, thresholding_method="soft", regularization_constant=2
    ):
        if cov_mat == None:
            cov_mat = self.sample_cov_estimate
        else:
            cov_mat = cov_mat
        rate = np.sqrt((np.log(self.N)) / self.T)

        if thresholding_method == "soft":
            threshold_matrix = (
                np.ones((self.N, self.N)) * rate * regularization_constant
            )
            threshold_matrix = threshold_matrix - np.diag(np.diag(threshold_matrix))

            diag_vals = np.sqrt(np.diag(cov_mat))
            corr_mat = cov_mat / np.outer(diag_vals, diag_vals)

            return np.outer(diag_vals, diag_vals) * generalized_threshold(
                corr_mat, threshold_matrix, method="soft threshold"
            )
        else:
            raise NotImplementedError("Other methods are not implemented yet")
        
def generalized_threshold(S: np.ndarray, T: np.ndarray, method, **kwargs) -> np.ndarray:
    """
    H = [h_ij] where h_ij(s_ij, t_ij) computes the generalized shrinkaged estimates
    This function applies generalized thresholding to a matrix M \n
    M : input matrix \n
    method: specify the name of the generalized thresholding function\n
    """
    assert S.shape == T.shape
    if method == "hard threshold":
        H = S
        H[np.where(np.abs(H) < T)] = 0
    elif method == "old soft threshold":
        dim1, dim2 = S.shape
        vec_sgn = np.sign(S.flatten())
        vec_value = np.maximum(np.zeros(dim1 * dim2), np.abs(S.flatten()) - T.flatten())
        H = (vec_sgn * vec_value).reshape(dim1, dim2)
    elif method == "soft threshold":
        H = np.sign(S) * ((np.abs(S) - T).clip(0))
    elif method == "soft threshold with sign":
        H = kwargs["sign"] * ((np.abs(S) - T).clip(0))
    else:
        print("unfinished")
        H = None
    return H


# %%
def isPD(Hermitian_Matrix):
    """
    Judge if a **symmetric** matrix M is pd
    """
    try:
        np.linalg.cholesky(Hermitian_Matrix)
        # print('success')
        return True
    except np.linalg.LinAlgError:
        return False


# %%


def isPSD(A, tol=1e-8):
    E = np.linalg.eigvalsh(A)
    return np.all(E > -tol)


# %%
import scipy.linalg as LA


def make_pd(A, tol=-1e-8):
    E = LA.eigvalsh(A)
    if np.any(E < tol):
        print("There are very small eigenvalues, might not be a good idea to truncate.")
        return None
    else:
        Ec = E.clip(0) + 1e-10
        V = LA.eigh(A)[1]
        Ac = V @ np.diag(Ec) @ V.T
        return Ac

# %% 
def hac_cov(residuals: np.ndarray, lags: int = None, demean = True) -> np.ndarray:
    """Calculate the heteroskedasticity and autocorrelation consistent (HAC) covariance matrix of a given set of residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        The residuals to calculate the HAC covariance matrix for. Should be a 1D or 2D array of (nfeature, nsample). 
    lags : int, optional
        The number of lags to use in the calculation, by default None
    
    Returns
    -------
    np.ndarray
        The HAC covariance matrix.
    """
    if residuals.ndim == 1:
        residuals = residuals[np.newaxis,:]
    p,n = residuals.shape
    if demean:
        residuals = residuals - np.mean(residuals, axis=1, keepdims=True)
    covariance = residuals @ residuals.T
    if lags == None:
        lags = int(np.floor(n**(1/4)))
    weights = np.zeros((lags + 1, 1))
    for i in range(1,lags + 1):
        weights[i] = 1 - (i / (lags + 1))
        covariance += weights[i] * (residuals[:,i:] @ residuals[:,:-i].T)
    covariance /= n
    return covariance
# %%
