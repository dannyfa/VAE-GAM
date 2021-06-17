"""

Module implementing 1D GP for regressors

This is largely based on Jack's code & on the notation for GP chapter in Kevin Murphy's textbook.
It also follows closely ideas the original Rassmussen & William's text.

"""

import numpy as np
import torch
from torch.distributions import MultivariateNormal, kl
import os, sys

class GP():
    """1D Gaussian Process w/ X-values on a grid and a Gaussian kernel."""
    def __init__(self, Xu, Yu, k_var, ls):
        """
        Parameters
        ----------
        Xu : torch.Tensor
        X values of inducing points.
        Yu : torch.Tensor
        Y values of inducing points. Trainable.
        k_var : float
        Vertical variance for Gaussian kernel. Trainable.
        ls : float
        Lengthscale for Gaussian kernel. Trainable.
        """
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.n = Yu.shape[0]
        assert len(Xu) > 1
        self.step = Xu[1] - Xu[0]
        self.Xu = Xu
        self.k_var = k_var
        self.ls = ls
        self.Yu = Yu #added this for debudding. Not truly needed as an attr
        k = _striped_matrix(self.n)
        k = _distance_to_kernel(k, self.k_var, self.ls, self.step)
        #unsure if fudge factor of 1e-4 is still needed
        #before observation noise was added in, this was needed to make model run stably
        self.ky = k + 1e-4*torch.eye(self.n).to(self.device)
        self.k_chol = torch.cholesky(self.ky)
        self.alpha = torch.inverse(self.k_chol.transpose(0,1)) @ torch.inverse(self.k_chol) @ Yu.unsqueeze(1)

    def evaluate_posterior(self, X_q):
        """
        Calculate the posterior at the given query points.
        Parameters
        ----------
        X_q : torch.Tensor
        Query points.

        Returns
        -------
        mean : torch.tensor
        Posterior means.
        covar : torch.tensor
        Posterior covariances.
        """
        n_q = X_q.shape[0]
        k_q = torch.zeros((self.n, n_q)).to(self.device)
        diff = self.step * self.n
        for j in range(n_q):
            dist = float(self.Xu[0] - X_q[j])
            k_q[:,j] = torch.arange(dist, dist + diff, self.step)[:self.n]
        k_q = _distance_to_kernel(k_q, self.k_var, self.ls)
        mean = k_q.transpose(0,1) @ self.alpha
        v = torch.inverse(self.k_chol) @ k_q
        k_qq = torch.zeros((n_q,n_q)).to(self.device)
        for i in range(n_q-1):
            for j in range(i+1, n_q):
                dist = X_q[i] - X_q[j]
                k_qq[i,j] = dist
                k_qq[j,i] = dist
        k_qq = _distance_to_kernel(k_qq, self.k_var, self.ls)
        covar = k_qq - v.transpose(0,1) @ v
        return mean.squeeze(1), covar

    def predict(self, X_q):
        """
        Predict the given query points.

        Parameters
        ----------
        X_q : torch.Tensor
        Query points.

        Returns
        -------
        mean : torch.tensor
        Prediction means.
        var : torch.tensor
        Prediction variances.
        """
        mean, covar = self.evaluate_posterior(X_q)
        return mean, torch.diag(covar) # diag necessary?


    def rsample(self, X_q, covar_id, save_dir, eps=1e-4):
        """
        Sample from the posterior at the given query points.

        Parameters
        ----------
        X_q : torch.Tensor
        Query points.
        n : int
        Number of samples.
        eps : float
        Conditioning number.
        Returns
        -------
        Y_qs : torch.Tensor
        A posterior samples.
        """
        mean, covar = self.evaluate_posterior(X_q)
        covar = covar + eps * torch.eye(covar.shape[0]).to(self.device)
        #adding block below to catch cases where posterior cov becomes singular
        #this happened once, but overall it is a very rare event
        m = MultivariateNormal(mean, covar)
        return m.rsample()

    #commented method and adopted a non-trainable (fixed) Sigma_0 instead
    #Sigma_0 (prior over query pts) was singular and causing issues later on
    #when computing kl for GP

    #def calc_prior_cov(self, X_q, nn_counts):
    #    """
    #    Computes prior covariance matrix for query data points (Sigma_0)
    #    Using covariance for prior over inducing points, Ku and Knu -- as defined on Appendix # B
    #    Parameters
    #    ----------
    #    X_q : torch.tensor
    #    Query points.

    #    Returns
    #    -----------
    #    Sigma_0: torch.tensor
    #    Covariance Matrix for prior distribution over data (query) points
    #    """
        #get Knu --> kernel distances between inducing pts and data points
    #    n_q = X_q.shape[0]
    #    k_q = torch.zeros((self.n, n_q)).to(self.device)
    #    diff = self.step * self.n
    #    for j in range(n_q):
    #        dist = float(self.Xu[0] - X_q[j])
    #        k_q[:,j] = torch.arange(dist, dist + diff, self.step)[:self.n]
    #    k_q = _distance_to_kernel(k_q, self.k_var, self.ls, self.sigma_y) #shape == 6x32, for 6 inducing pts and n=32 minibatch
    #    pu_cov = 10*torch.eye(6).to(self.device) #this is cov for prior over inducing pts. Chose s0^2 == 10.
        #get Ku --> mat formed by evaluating kernel at each pair of inducing pts
    #    ku = _striped_matrix(self.n)
    #    ku = ku * self.step
    #    ku = _distance_to_kernel(ku, self.k_var, self.ls, self.sigma_y)
    #    A = k_q.T @ torch.inverse(ku)
    #    Sigma_0 = A @ (pu_cov - ku) @ A.T
    #    return Sigma_0

    def calc_GP_kl(self, X_q):
        """
        Computes KL between GP prior and posterior GPs
        using closed-form for KL between 2 MV Gaussians.

        Parameters
        ------------
        X_q: torch.tensor
        Query points
        Assumes a non-trainable (fixed) prior over query pts.

        Returns
        -------------
        Kl divergence between prior and posterior GP distributions over data pts
        """

        fa, Sigma_a = self.evaluate_posterior(X_q)
        #using a pre-set prior here... As opposed to a trainable one.
        Sigma_0 = 1e-1*torch.eye(X_q.shape[0]).to(self.device)
        #again unsure if 1e-4 fudge factor below is truly needed.
        #in simulations I ran, it helps a ton in stabilizing things and allowing model to run stably
        q_f = MultivariateNormal(fa, (Sigma_a + 1e-4*torch.eye(X_q.shape[0]).to(self.device)))
        p_f = MultivariateNormal(torch.zeros(X_q.shape[0]).to(self.device), Sigma_0)
        gp_kl = kl.kl_divergence(p_f, q_f)
        return gp_kl

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
