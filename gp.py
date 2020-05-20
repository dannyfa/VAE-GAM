"""
Module implementing 1D GP for regressors

This is largely based on Jack's code & on the GP chapter in Kevin Murphy's textbook.

- Added extra method to calculate marginal likelihood for join training w/ VAE
- Added some extras to pass tensors to CUDA as needed
- To Do's: add a plotting method to look into smoothness of GP as it gets trained 

"""

import numpy as np
import torch
from torch.distributions import MultivariateNormal

class GP():
    """1D exact Gaussian Process w/ X-values on a grid and a Gaussian kernel."""
    def __init__(self, Xu, Yu, k_var, ls, y_var=0.1):
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
        y_var : float
        Observation noise. Set to 0.1 for now.
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
        self.ky = k + y_var*torch.eye(self.n).to(self.device)
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

    #am not really using this method for now...
    #ask Jack if needed??
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

    def calc_mll(self, Yu):
        #based on Murphy's book pp. 231
        cte = torch.as_tensor((2*np.pi)).to(self.device)
        a = (-0.5*torch.matmul(Yu.unsqueeze(-1).transpose(0,1), self.alpha)).squeeze(-1)
        b = -0.5*torch.logdet(self.ky)
        c = -0.5*self.n*torch.log(cte)
        mll = a + b + c
        return mll

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
