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
from svgd_setup import *


if __name__ == "__main__":
    print("Running sampling experiment to reproduce Figure 2.1 and Table 2.1 in report\n")
    alt.data_transformers.enable('default', max_rows=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temp = np.array([[0.6, 0.4], [0.4, 1.7]])
    cov = temp.T @ temp
    cov1 = np.linalg.inv(cov)
    covariance_matrices = [torch.from_numpy(cov), torch.from_numpy(cov1)]
    means = [torch.zeros(2), torch.zeros(2)]
    weights = np.array([0.5, 0.5])

    gmm = GMM(covariance_matrices=covariance_matrices, means=means, weights=torch.from_numpy(weights))
    gmm_tf = GMM_tf(covariance_matrices=[cov, cov1], means=[np.zeros(2), np.zeros(2)], weights=weights)

    torch.manual_seed(42)
    INITIAL_SAMPLE = torch.zeros(2)
    NUM_SAMPLES = 100
    BURN_IN = 100
    PROPOSAL_STD = 1.3 # For MH sampler
    STD_DECAY = 1.0 # to anneal proposal std in MH sampler
    LEAPFROG_STEPS = 7 # for HMC sampler
    STEP_SIZE = 2.2e-1 # leapfrog step size for HMC sampler

    initial_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=2*torch.eye(2))
    initial_particles = initial_dist.sample([NUM_SAMPLES])
    OPTIMIZER = optim.Adam([initial_particles], lr=1e-2)

    metropolis_sampler = Metropolis_Hastings(proposal_std=PROPOSAL_STD, std_decay=STD_DECAY, target=gmm)
    hamiltonian_sampler = Hamiltonian_MC(target=gmm, leapfrog_steps=LEAPFROG_STEPS, step_size=STEP_SIZE)
    svgd = SVGD(P=gmm, K=k, optimizer=OPTIMIZER)
    nuts = NUTS(target=gmm_tf, leapfrog_step_size=5e-2, leapfrog_steps=10)

    mh_samples = metropolis_sampler.sample(initial_sample=INITIAL_SAMPLE, n_samples=NUM_SAMPLES, burn_in=BURN_IN)
    print(f"MH acceptance ratio = {metropolis_sampler.acceptance_ratio}")
    hmc_samples = hamiltonian_sampler.sample(initial_sample=INITIAL_SAMPLE, n_samples=NUM_SAMPLES, burn_in=BURN_IN)
    print(f"HMC acceptance ratio = {hamiltonian_sampler.acceptance_ratio}")
    nuts_samples = nuts.sample(n_samples=NUM_SAMPLES, burn_in=BURN_IN)
    svgd_samples = svgd.run(initial_particles, max_steps=500)


    save_path = "./Figures"
    gmm_chart = get_density_chart(gmm)
    mh_chart = gmm_chart + get_particles_chart(mh_samples)
    mh_chart.save(os.path.join(save_path, "mh_sample.png"))

    hmc_chart = gmm_chart + get_particles_chart(hmc_samples)
    hmc_chart.save(os.path.join(save_path, "hmc_sample.png"))

    nuts_chart = gmm_chart + get_particles_chart(nuts_samples)
    nuts_chart.save(os.path.join(save_path, "nuts_sample.png"))

    svgd_chart = gmm_chart + get_particles_chart(svgd_samples)
    svgd_chart.save(os.path.join(save_path, "svgd_sample.png"))

    ksd_estimator = KSD(target=gmm, kernel=k)
    ksd_mh = ksd_estimator.estimate(samples=mh_samples)
    ksd_hmc = ksd_estimator.estimate(samples=hmc_samples)
    ksd_nuts = ksd_estimator.estimate(samples=torch.from_numpy(nuts_samples))
    ksd_svgd = ksd_estimator.estimate(samples=svgd_samples)

    print(f"Kernelised Stein Discrepancies:\n\tMH: {ksd_mh}\n\tHMC: {ksd_hmc}\n\tNUTS: {ksd_nuts}\n\tSVGD: {ksd_svgd}")