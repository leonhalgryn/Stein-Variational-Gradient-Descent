import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.optim as optim
import altair as alt
import pandas as pd
import functools
import torch
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.spatial.distance import squareform, pdist
from svgd_setup import *


if __name__ == "__main__":
    print("Running code to reproduce figures from Chapter 3 in report")
    save_path = "./Figures"
    # Mode collapse figure (Figure 3.1a)
    # Here we illustrate the mode collapse by scaling the repulsive force to be very small.
    gmm = GMM(covariance_matrices=[torch.eye(2), torch.eye(2)], means=[torch.tensor([2.0, 2.0]), torch.tensor([-2.0, -2.0])])
    torch.manual_seed(42)
    INITIAL_SAMPLE = torch.zeros(2)
    NUM_SAMPLES = 100
    initial_dist = torch.distributions.MultivariateNormal(loc=torch.tensor([2.0, 2.0]), covariance_matrix=torch.eye(2))
    initial_particles = initial_dist.sample([NUM_SAMPLES])
    OPTIMIZER = optim.Adam([initial_particles], lr=1e-2)
    ALPHA = 1e-9
    MAX_STEPS = 500
    svgd = SVGD(P=gmm, K=k, optimizer=OPTIMIZER, alpha=ALPHA)
    svgd_samples = svgd.run(X=initial_particles, max_steps=MAX_STEPS)
    gmm_chart = get_density_chart(gmm, step=0.05)
    mode_collapse_chart = gmm_chart + get_particles_chart(svgd_samples)
    mode_collapse_chart.save(os.path.join(save_path, "mode_collapse.png"))

    # Variance collapse and diminishing repusive force figures (Figures 3.1b and 3.1c)
    torch.manual_seed(42)
    n_particles = torch.tensor([50])
    max_steps = 1000
    d_grid = torch.linspace(1, 1000, 100).int()
    var_estimates = []
    var_estimates_nuts = []
    avg_repulsive_forces = []
    avg_driving_forces = []
    for i in tqdm(range(len(d_grid))):
        d = d_grid[i]
        init_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(d), covariance_matrix=10*torch.eye(d))
        x_init = init_dist.sample(n_particles)
        particles = x_init.clone()
        target = GMM(covariance_matrices=[torch.eye(d)], means=[torch.zeros(d)])
        target_tf = GMM_tf(covariance_matrices=[np.eye(d)], means=np.zeros(d))
        svgd = SVGD(P=target, K=k, optimizer=optim.Adam([particles], lr=1e-2))
        svgd.run(X=particles, max_steps=max_steps, tol=1e-6)
        avg_repulsive_forces.append(np.mean(svgd.repulse_norm_history))
        avg_driving_forces.append(np.mean(svgd.driving_norm_history))
        est_var = torch.var(particles)
        var_estimates.append(est_var)

    avg_norm = [avg_driving_forces[i] + avg_repulsive_forces[i] for i in range(len(avg_repulsive_forces))]

    plt.plot(d_grid, var_estimates, label="$\hat{\sigma}^2$")
    plt.plot(d_grid, np.repeat(1.0, len(d_grid)), label="true $\sigma^2$")
    plt.xlabel("Dimensionality ($d$)")
    plt.ylabel("Marginal variance ($\sigma^2$)")
    plt.legend()
    plt.savefig(os.path.join(save_path, "variance_collape.pdf"), dpi=300)
    plt.show()

    plt.plot(d_grid, avg_repulsive_forces, label="Repulsive force norms")
    plt.plot(d_grid, avg_driving_forces, label="Driving force norms")
    plt.plot(d_grid, avg_norm, label="Overall norm")
    plt.xlabel("Dimensionality ($d$)")
    plt.ylabel("Average magnitude")
    plt.legend()
    plt.savefig(os.path.join(save_path, "norms_collapse.pdf"), dpi=300)
    plt.show()
    