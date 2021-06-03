"""
Module implementing 1D GP for regressors

This is largely based on Jack's code & on the GP chapter in Kevin Murphy's textbook.

- Added some extras to pass tensors to CUDA as needed
- Got rid of observation noise (y_var) term
- Added extra methods to calc covariance for prior over query points and to compute GP kl term --> used in joint objective
- mll method still here --> NOT currently used... Will likely take it out
"""
import numpy as np
import torch
from torch.distributions import MultivariateNormal, kl

class GP():
    """1D exact Gaussian Process w/ X-values on a grid and a Gaussian kernel."""
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
        #adding device attr
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        #init attrs
        self.n = Yu.shape[0]
        assert len(Xu) > 1
        self.step = Xu[1] - Xu[0]
        self.Xu = Xu
        self.k_var = k_var
        self.ls = ls
        # Calculate the Cholesky factor of the kernel matrix.
        k = _striped_matrix(self.n)
        k = _distance_to_kernel(k, self.k_var, self.ls, self.step)
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


    #am using this method on new GP implementation
    #i.e., am taking actual samples vs. mean plug-in estimator
    def rsample(self, X_q, eps=1e-6):
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
        return MultivariateNormal(mean, covar).rsample()

    def calc_prior_cov(self, X_q):
        """
        Computes prior covariance matrix for query data points (Sigma_0)
        Using covariance for prior over inducing points, Ku and Knu -- as defined on Appendix # B
        Parameters
        ----------
        X_q : torch.tensor
        Query points.

        Returns
        -----------
        Sigma_0: torch.tensor
        Covariance Matrix for prior distribution over data (query) points
        """
        #get Knu --> kernel distances between inducing pts and data points
        n_q = X_q.shape[0]
        k_q = torch.zeros((self.n, n_q)).to(self.device)
        diff = self.step * self.n
        for j in range(n_q):
            dist = float(self.Xu[0] - X_q[j])
            k_q[:,j] = torch.arange(dist, dist + diff, self.step)[:self.n]
        k_q = _distance_to_kernel(k_q, self.k_var, self.ls) #shape == 6x32
        pu_cov = 10*torch.eye(6).to(self.device) #this is cov for prior over inducing pts
        #get Ku --> mat formed by evaluating kernel at each pair of inducing pts
        ku = _striped_matrix(self.n)
        ku = ku * self.step #this is dist mat between each pair of inducing pts
        ku = _distance_to_kernel(ku, self.k_var, self.ls)
        A = k_q.T @ torch.inverse(ku) #transpose here needed to make Knu (32x6) -- originally (6x32), Ku shape = 6x6
        Sigma_0 = A @ (pu_cov - ku) @ A.T
        return Sigma_0

    def calc_GP_kl(self, X_q, Sigma_0, M):
        """
        Computes KL between GP prior and posterior
        using closed-form for KL between 2 MV Gaussians
        Parameters
        ------------
        X_q: torch.tensor
        Query points
        Sigma_0: cov matrix for prior distribution over data pts
        M: number of inducing pts

        Returns
        -------------
        Kl divergence between prior and posterior GP distributions over data pts
        Used Cholesky decomposition for better numerical stability when inverting/Using
        covariance matrices
        """
        #get posterior mean and covariance
        fa, Sigma_a = self.evaluate_posterior(X_q)
        #set up 2 MV Gaussians
        #am having issues with Chol decomposition here
        #these mats are SINGULAR :(
        #had to add some extra noise to diag of both Sigm_0 and Sigma_a for things to work...
        #this is even when I kick out similar points :(
        q_f = MultivariateNormal(fa, (Sigma_a + 1e-4*torch.eye(X_q.shape[0]).to(self.device)))
        p_f = MultivariateNormal(torch.zeros(X_q.shape[0]).to(self.device), \
        (Sigma_0 + 1e-4*torch.eye(X_q.shape[0]).to(self.device)))
        gp_kl = kl.kl_divergence(p_f, q_f)
        return gp_kl

        #version below is without using torch's module ...
        #La = torch.cholesky(Sigma_a)
        #L0 = torch.cholesky(Sigma_0)
        #fa = fa.unsqueeze(1) #put this back into 32x1 shape
        #now compute KL
        #this follows Eqn 13 on App. # B
        #each added line corresponds to a term on RHS of Eqn 13
        #kl = 0.5 * (fa.T @ (torch.inverse(L0.transpose(0,1)) @ torch.inverse(L0)) @ fa)
        #kl += 0.5 * (torch.trace((torch.inverse(L0) @ La) @ (torch.inverse(L0) @ La).T))
        #kl += 0.5 * (torch.logdet(L0 @ L0.T))
        #kl -= 0.5 * (torch.logdet(La @ La.T))
        #kl -= M
        #return kl[0][0]

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
    return k_var * torch.exp(-torch.pow(scale_factor / np.sqrt(2) / ls * dist_mat, 2))
