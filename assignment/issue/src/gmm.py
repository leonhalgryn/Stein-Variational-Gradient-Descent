import torch

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

if __name__ == "__main__":
    pass