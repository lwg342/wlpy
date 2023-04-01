# -> Created on 03 November 2020
# -> Author: Weiguang Liu
# %% Importing Library
from sklearn import covariance as sk_cov
import nonlinshrink as nls
import numpy as np
from wlpy.gist import generalized_threshold
# %%


class Covariance():
    """
    Several Estimators for Covariance Matrix Given a Sample of Size T*N or N*p
    """

    def __init__(self, DF_TxN=np.eye(2)):
        """
        Covariance estimation from the observations X:T*N
        """
        self.X = DF_TxN
        self.T, self.N = DF_TxN.shape
        self.sample_cov_estimate = self.sample_cov()

    def sample_cov(self):
        sample_cov_estimate = np.cov(np.array(self.X), rowvar=False)
        return sample_cov_estimate

    def linear_shrinkage(self):
        """
        Ledoit and Wolf 2004
        """
        S_lw = sk_cov.LedoitWolf().fit(self.X).covariance_
        return S_lw

    def nonlinear_shrinkage(self):
        S_nlshrink = nls.shrink_cov(self.X)
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

    def threshold_corr(self, cov_mat=None, thresholding_method="soft", regularization_constant=2):
        if cov_mat == None:
            cov_mat = self.sample_cov_estimate
        else:
            cov_mat = cov_mat
        rate = np.sqrt((np.log(self.N))/self.T)

        if thresholding_method == "soft":
            threshold_matrix = np.ones(
                (self.N, self.N))*rate*regularization_constant
            threshold_matrix = threshold_matrix - \
                np.diag(np.diag(threshold_matrix))

            diag_vals = np.sqrt(np.diag(cov_mat))
            corr_mat = cov_mat / np.outer(diag_vals, diag_vals)

            return np.outer(diag_vals, diag_vals) * generalized_threshold(corr_mat, threshold_matrix, method="soft threshold")
        else:
            raise NotImplementedError("Other methods are not implemented yet")
