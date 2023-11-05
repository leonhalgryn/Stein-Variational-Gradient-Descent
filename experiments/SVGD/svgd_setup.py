import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.optim as optim
import altair as alt
import pandas as pd
import functools
import torch.distributions as dist
import torch.nn.functional as F
import os
import tensorflow as tf
import tensorflow_probability as tfp

# SETUP FOR GMM CLASS

class GMM(torch.distributions.Distribution):
    """
    This class is used to construct Gaussian Mixture Models (GMM's)

    Attributes:
        - covariance_matrices (iterable): list of covariance matrices for respective components
        - means (iterable): list of means for the GMM components
        - weights (iterable): weights for the GMM components (if None assume equal weights)

    Methods:
        - log_prob(value): function to calculate the log probability of sample 'value' for the GMM
    """

    def __init__(self, covariance_matrices, means, weights=None):
        self.n_components = len(means)
        if weights is None:
            weights = torch.repeat_interleave(torch.tensor([1.0 / self.n_components]), self.n_components)
        self.means = means
        self.covariance_matrices = covariance_matrices
        self.weights = weights
        self.distributions = [
            torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
            for mu, cov in zip(means, covariance_matrices)
        ]

    def log_prob(self, value):
        log_probs = torch.cat(
            [p.log_prob(value).unsqueeze(-1) for p in self.distributions], dim=-1)
        weighted_log_probs = log_probs + torch.log(self.weights)
        return weighted_log_probs.logsumexp(dim=-1)
    

# TensorFlow version of GMM class (needed for NUTS sampler using tfp.mcmc)
class GMM_tf(tf.Module):
    def __init__(self, covariance_matrices, means, weights=None):
        self.n_components = len(means)
        if weights is None:
            weights = np.repeat(1.0 / self.n_components, self.n_components)
        self.means = means
        self.covariance_matrices = covariance_matrices
        self.weights = weights

    def log_prob(self, value):
        log_probs = tf.concat([
            tfp.distributions.MultivariateNormalTriL(
                loc=mu, scale_tril=tf.linalg.cholesky(cov)).log_prob(value)[:, tf.newaxis]
                for mu, cov in zip(self.means, self.covariance_matrices)], axis=-1)
        weighted_log_probs = log_probs + tf.math.log(self.weights)
        return tf.reduce_logsumexp(weighted_log_probs, axis=-1)
    

# PLOTTING CODE
"""
#NOTE: Plotting code taken from https://sanyamkapoor.com/kb/the-stein-gradient
Here I have used code for plotting from an SVGD tutorial by Sanyam Kapoor. 
I have done this since matplotlib does not allow multiple contourf calls on one plot
"""

alt.data_transformers.enable('default', max_rows=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_density_chart(P, d=7.0, step=0.1, save_path=None):
    xv, yv = torch.meshgrid([
        torch.arange(-d, d, step), 
        torch.arange(-d, d, step)
    ])
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
    p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()
    
    df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
    df = pd.DataFrame({
        'x': df[:, :, 0].ravel(),
        'y': df[:, :, 1].ravel(),
        'p': df[:, :, 2].ravel(),
    })
    
    chart = alt.Chart(df).mark_point().encode(
        x='x:Q',
        y='y:Q',
        color=alt.Color('p:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['x','y','p']
    )
    
    if save_path:
        chart.save(save_path)
    
    return chart

def get_particles_chart(X, save_path=None):
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
    })

    chart = alt.Chart(df).mark_circle(color='red').encode(
        x='x:Q',
        y='y:Q'
    )
    
    if save_path:
        chart.save(save_path)
    
    return chart


def k(x, y, sigma=None):
    #NOTE: here I use the RBF kernel implementation from Sanyam Kapoor https://sanyamkapoor.com/kb/the-stein-gradient
    # This implementation is much more efficient than mine from earlier, and automatically implements median heuristic for bandwidth
    dnorm2 = -2 * x.matmul(y.t()) + x.matmul(x.t()).diag().unsqueeze(1) + y.matmul(y.t()).diag().unsqueeze(0)
    if sigma is None:
        dnorm_np = dnorm2.detach().cpu().numpy()
        h = np.median(dnorm_np) / (2 * np.log(x.size(0) + 1))
        sigma = np.sqrt(h).item()
    
    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    return (-gamma * dnorm2).exp()

# SAMPLERS

class Metropolis_Hastings:
    """
    Class to implement random-walk Metropolis-Hastings sampler
    Attributes:
        - proposal_std (float): standard deviation for proposals
        - std_decay (float): (optional) to decay proposal_std over iterations
        - target (torch distribution): target distribution with a log_prob method
    Methods:
        - sample(initial_sample, n_samples, burn_in): function to sample from target distribtion
            using MH sampler. ARgs: initial sample (torch tensor), n_samples (int): number of samples desired
                                    burn_in (int): number of burn-in samples
    """
    def __init__(self, proposal_std, std_decay, target):
        self.proposal_std = proposal_std
        self.std_decay = std_decay
        self.target = target
        self.samples = []
        self.acceptance_ratio = None

    def sample(self, initial_sample, n_samples, burn_in):
        samples = [initial_sample]
        current = initial_sample
        n_accepts = 0
        for _ in tqdm(range(n_samples + burn_in)):
            proposal = current + torch.randn_like(current) * self.proposal_std
            mh_ratio = min(1, torch.exp(self.target.log_prob(proposal) - self.target.log_prob(current)))
            if torch.rand(1) < mh_ratio:
                current = proposal
                n_accepts += 1
            samples.append(current)
            self.proposal_std *= self.std_decay
        
        self.acceptance_ratio = n_accepts / (burn_in + n_samples)
        samples = samples[burn_in+1:]
        self.samples = samples
        return torch.stack(samples)


class Hamiltonian_MC:
    """
    Class to implement Hybrid/Hamiltonian Monte Carlo sampler.
    Attributes:
        - target (torch distribution): target distribution to sample from; must have log_prob method
        - leapfrog_steps (int): number of leapfrog steps to simulate trajectories
        - step_size (float): step size for each leapfrog step
    Methods:
        - _leapfrog(sample, momentum): implement one leapfrog step
        - sample(initial_sample, n_samples, burn_in): function to sample from target distribution
            Args: initial sample (torch tensor), n_samples (int) number of smaples desired,
                    burn_int (int): number of burn-in samples
    """
    def __init__(self, target, leapfrog_steps, step_size):
        self.target = target
        self.leapfrog_steps = leapfrog_steps
        self.step_size = step_size
        self.samples = []
        self.acceptance_ratio = None

    def _leapfrog(self, sample, momentum):
        sample.requires_grad_(True)
        grad_log_p = autograd.grad(self.target.log_prob(sample), sample)[0]
        new_momentum = momentum + 0.5 * self.step_size * grad_log_p
        sample.requires_grad_(False)
        new_sample = sample + self.step_size * new_momentum.detach().clone()
        final_momentum = new_momentum + 0.5 * self.step_size * grad_log_p
        # sample.requires_grad_(False)
        return new_sample, final_momentum

    def sample(self, initial_sample, n_samples, burn_in):
        samples = [initial_sample]
        x = initial_sample
        n_accepts = 0
        for _ in tqdm(range(n_samples + burn_in)):
            momentum_dist = torch.distributions.MultivariateNormal(loc=torch.zeros_like(x), covariance_matrix=torch.eye(x.shape[0]))
            m_0 = momentum_dist.sample()
            x_new = x
            energy = torch.exp(self.target.log_prob(x_new) - 0.5 * torch.square(m_0).sum())
            x_temp = x
            m = m_0
            for i in range(self.leapfrog_steps):
                x_temp, m = self._leapfrog(x_temp, m)
            new_energy = torch.exp(self.target.log_prob(x_temp) - 0.5 * torch.square(m).sum())
            alpha = min(1, new_energy / energy)
            if torch.rand(1) < alpha:
                x_new = x_temp
                m = m
                n_accepts += 1
            samples.append(x_new)
        self.acceptance_ratio = n_accepts / (n_samples + burn_in)
        samples = samples[burn_in+1:]
        return torch.stack(samples)

 
class NUTS:
    """
    Class to implement No U-Turn Sampler using tf.probability.
    Attributes:
        - target (torch distribution): target distribution to sample from; must have a log_prob method
        - leapfrog_step_size (float): step size for each leapgrog step
        - leapfrog_steps (int): number of leapfrog steps for simulating trajectories
    """
    def __init__(self, target, leapfrog_step_size=0.1, leapfrog_steps=1):
        self.target = target
        self.samples = None
        self.sampler = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn = self.target.log_prob,
            step_size = leapfrog_step_size,
            max_tree_depth=10,
            max_energy_diff=1000.0,
            unrolled_leapfrog_steps=leapfrog_steps,
            parallel_iterations=10,
            experimental_shard_axis_names=None,
            name=None
        )

    def sample(self, n_samples, burn_in):
        samples = []
        init_state = tf.zeros(2, dtype=tf.float64)[tf.newaxis, :]
        state = init_state[tf.newaxis, :]
        prev_results = self.sampler.bootstrap_results(state)
        for t in tqdm(range(n_samples + burn_in)):
            next_state, prev_results = self.sampler.one_step(state, prev_results)
            samples.append(next_state)
            state = next_state
        samples = tf.squeeze(tf.stack(samples)).numpy()
        samples = samples[burn_in:]
        self.samples = samples
        return samples
    

class SVGD:
    """
    This class is used to implement 'vanilla' Stein Variational Gradient Descent (SVGD)

    Attributes:
        - P (torch distribution): target distribution to approximate using SVGD
        - K (function): kernel function
        - optimizer (torch.optim optimizer): optimizer to use for step size selection

    Methods:
        - phi_fn(X, alpha): function to calculate optimal perturbation direction
        - step(X, alpha): function to implement a gradient descent step using gradient returned by phi_fn
    """

    def __init__(self, P, K, optimizer, alpha=1.0):
        self.P = P
        self.K = K
        self.alpha = alpha
        self.optimizer = optimizer
        self.repulse_norm_history = []
        self.driving_norm_history = []
    
    def phi_fn(self, X):
        X = X.detach().requires_grad_(True)
        log_prob = self.P.log_prob(X)
        score_fn = autograd.grad(log_prob.sum(), X)[0]

        k_xx = self.K(X, X.detach())
        dk = - autograd.grad(k_xx.sum(), X)[0]
        self.repulse_norm_history.append(torch.linalg.norm(dk))
        self.driving_norm_history.append(torch.linalg.norm(score_fn))
        phi = (k_xx.detach().matmul(score_fn) + self.alpha*dk) / X.size(0)
        return phi
    
    def step(self, X):
        self.optimizer.zero_grad()
        X.grad = -self.phi_fn(X)
        self.optimizer.step()

    def run(self, X, max_steps, tol=1e-6):
        # for t in tqdm(range(max_steps)):
        for t in range(max_steps):
            self.step(X)
            phi_norm = torch.linalg.norm(X.grad)
            if torch.abs(phi_norm) <= tol:
                print(f"Algorithm converged after {t + 1} iterations")
        return X
    

# KSD

class KSD:
    def __init__(self, target, kernel):
        self.target = target
        self.K = kernel
        self.sigma = None
        
    def median_heuristic(self, samples):
        dnorm2 = -2 * samples.matmul(samples.T) + samples.matmul(samples.T).diag().unsqueeze(1) + samples.matmul(samples.T).diag().unsqueeze(0)
        dnorm_np = dnorm2.detach().cpu().numpy()
        h = np.median(dnorm_np) / (2 * np.log(len(samples) + 1))
        sigma = np.sqrt(h).item()
        self.sigma = sigma


    def _stein_kernel(self, x, y):
        d = len(x)
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        score_x = autograd.grad(self.target.log_prob(x), x)[0]
        score_y = autograd.grad(self.target.log_prob(y), y)[0]

        k = self.K(x.unsqueeze(0), y.unsqueeze(0), sigma=self.sigma)

        dk_x = autograd.grad(k.sum(), x, create_graph=True)[0]
        dk_y = autograd.grad(k.sum(), y, create_graph=True)[0]
        dk_y = dk_y.requires_grad_(True)
        ddk = autograd.grad(dk_y.sum(), x)[0]
        
        u = k * score_x @ score_y + score_x @ dk_y + dk_x @ score_y + ddk.sum()

        return u.item()
    
    def estimate(self, samples):
        self.median_heuristic(samples=samples)
        n = len(samples)
        ksd = 0
        for i in tqdm(range(n)):
            for j in range(n):
                if i == j:
                    continue
                else:
                    ksd += self._stein_kernel(x=samples[i], y=samples[j])
        return ksd / (n * (n - 1))

if __name__ == "__main__":
    pass
