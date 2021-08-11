"""

Module implementing 1D sparse GP for covariates.
This follows equations detailed in MLHC VAE-GAM paper (2021) and notation/ideas
for sparse GP follow work of Hensman.

"""
import numpy as np
import torch
from torch.distributions import MultivariateNormal, kl
import os, sys

class GP():
    """1D Gaussian Process w/ X-values on a grid and a Gaussian kernel."""
    def __init__(self, Xu, k_var, ls, qu_m, qu_S):
        """
        Parameters
        ----------
        Xu : torch.Tensor
        X values of inducing points.
        k_var : float
        Vertical variance for Gaussian kernel. Trainable.
        ls : float
        Lengthscale for Gaussian kernel. Trainable.
        qu_m: torch.tensor
        posterior mean for 1D GP. Trainable.
        qu_S: torch.tensor
        posterior covariance matrix for 1D GP. Trainable.
        """
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.n = Xu.shape[0]
        assert len(Xu) > 1
        self.step = Xu[1] - Xu[0]
        self.Xu = Xu
        self.k_var = k_var
        self.ls = ls
        self.qu_m = qu_m
        self.qu_S = qu_S

    def compute_GP_kl(self, num_inducing_pts, i, xq, save_dir):
        """
        Computes KL divergence for (non-linear gain) GP term.
        """
        prior_dist = MultivariateNormal(torch.zeros(num_inducing_pts).to(self.device), \
        10*torch.eye(num_inducing_pts).to(self.device))
        #adding try/except block to catch/save GP params when instability occurs
        # i, xq and save_dir are ONLY used here. They will be taken out of this method
        # once stability resolves
        try:
            post_dist = MultivariateNormal(self.qu_m, self.qu_S)
        except:
            print('Oops, something went wrong with qu_S!!')
            vars_dict = {}
            fname = os.path.join(save_dir, 'qu_S_diagnostics.tar')
            vars_dict['qu_m'] = self.qu_m.detach()
            vars_dict['qu_S'] = self.qu_S.detach()
            vars_dict['ls'] = self.ls.detach()
            vars_dict['k_var'] = self.k_var.detach()
            vars_dict['Xu'] = self.Xu
            vars_dict['cov_id'] = i
            vars_dict['batch_vals'] = xq
            torch.save(vars_dict, fname)
        gp_kl = kl.kl_divergence(post_dist, prior_dist) #this will fail if previous try fails...
        return gp_kl

    def evaluate_posterior(self, X_q):
        """
        Computes posterior over data points -- q(f).

        Parameters
        ----------
        X_q : torch.tensor
        Query points.
        qu_m: torch.Tensor
        Mean for posterior over inducing points.
        qu_S: torch.Tensor
        Covariance matrix for posterior over inducing pts.

        Returns
        -----------
        f_bar : torch.tensor
        Mean for posterior distribution over data points.
        Sigma: torch.tensor
        Covariance Matrix for posterior distribution over data points
        """

        #get Knu --> kernel distances between inducing pts and data points
        n_q = X_q.shape[0]
        knu = torch.zeros((self.n, n_q)).to(self.device)
        diff = self.step * self.n
        for j in range(n_q):
            dist = float(self.Xu[0] - X_q[j])
            knu[:,j] = torch.arange(dist, dist + diff, self.step)[:self.n]
        knu = _distance_to_kernel(knu, self.k_var, self.ls)
        #get Knn --> mat formed by evaluating kernel at data/min-batch points
        knn = torch.zeros((n_q, n_q)).to(self.device)
        for i in range(n_q):
            item = X_q[i].expand(1, n_q)
            diff = X_q - item
            knn[i, :] = diff
        knn = _distance_to_kernel(knn, self.k_var, self.ls)
        #get Ku --> mat formed by evaluating kernel at each pair of inducing pts
        ku = _striped_matrix(self.n).to(self.device)
        ku = _distance_to_kernel(ku, self.k_var, self.ls, self.step)
        #now get Sigma and f_bar
        A = knu.T @ torch.inverse(ku)
        f_bar = A @ torch.squeeze(self.qu_m)
        Sigma = knn + (A @ (self.qu_S - ku) @ A.T)
        return f_bar, Sigma


def _striped_matrix(n):
    """Make and n-by-n matrix with entries given by l1 distance to diagonal."""
    mat = torch.zeros((n,n)).cuda()
    for i in range(1,n):
        mat[range(i,n),range(0,n-i)] = i
        mat[range(0,n-i),range(i,n)] = i
    return mat

def _distance_to_kernel(dist_mat, k_var, ls, scale_factor=1.0):
    """
    Map distance to Gaussian kernel similarity, elementwise.

    Parameters
    ----------
    dist_mat : torch.Tensor
    Distance matrix (possibly signed).
    k_var : float
    Vertical variance for Gaussian kernel.
    ls : float
    Lengthscale for Gaussian kernel.
    scale_factor : float, optional
    Scale distances by this value. Defaults to `1.0`.
    """
    return (k_var * torch.exp(-torch.pow(scale_factor / np.sqrt(2) / ls * dist_mat, 2)))
