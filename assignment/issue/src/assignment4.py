# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from torch import autograd
from torch import optim
from scipy.stats import multivariate_normal
import gmm
import altair as alt
import plotting_code
from matplotlib.animation import FuncAnimation
from ipywidgets import interact, IntSlider
from IPython.display import Image, display, HTML

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2



# # Stein Variational Gradient Descent

# <div class="alert alert-block alert-info">
#     
# **Student numbers:** 
#     
# </div>

# ### General Instructions
#
# Please provide the student numbers of your group members in the above cell. 
#
# Submitted work will be graded on technical content as well as writing and presentation quality. **Submit this notebook and all supporting files as a zip archive on Sunlearn. Filename is: group-nr\_assignment4.zip, e.g. 3\_assignment4.zip - incorrectly named files may not be marked.** The notebook assignment4.ipynb in your submission will be converted to PDF during the marking process, and your submission should be able to be marked **based on the PDF alone**.  Thus ensure you include the output of code cells; note that widgets in the notebook are not included during the export process, so leave appropriate comments on your findings where relevant. 
#
# Clearly mark any of your group's code using SOLUTION_START and SOLUTION_END comments. Answer all questions and exercises in their corresponding blue blocks, and add blue blocks describing what you have done anywhere else you modify or perform additional work - the blue block and comments help us differentiate your work from the issued assignment, so if you fail to do this, some of your work may not be marked.
#
# You may not import any additional packages not available in NARGA or on the learner.cs.sun.ac.za server without prior approval, i.e. ensure your code runs in one of these environments.  If you use learner.cs.sun.ac.za, specify in the blue block above the configuration w.r.t. image and GPU that was used. 
#
# Lastly, note that the assignment is open-ended - further investigation of any aspect documented in the submission will be considered during marking.

# # 0. Required Reading

# Before attempting this assignment, read the following papers and lecture notes in order:
#
# 1. Read the [Wikipedia entry](https://en.wikipedia.org/wiki/Stein%27s_method) on Stein's method.
# 2. Read the [University College London lecture notes](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf) on kernels and reproducing kernel Hilbert spaces by Arthur Gretton. Do not get bogged down by the details, and rather focus on an intuitive understanding. For more information on kernels and RKHSs, see the [Blog post by Xavier Bourret Sicotte](https://xavierbourretsicotte.github.io/Kernel_feature_map.html).
# 3. Read the paper on kernelised Stein discrepancy (KSD) \[[2](#References)\].
# 4. Read the Stein Variational Gradient Descent (SVGD) paper \[[1](#References)\]. You can also read \[[3](#References)\] for more information on SVGD.
#
# Furthermore, the [Depth First Learning (DFL) course on SVGD](https://www.depthfirstlearning.com/2020/SVGD) by Bhairav Mehta \[[9](#References)\] contains excellent resources on the development of SVGD. The content introduced in this assignment follows the same structure as this DFL course, and you can refer to this course for additional information on each section.

# ## 1. Background and Introduction

# Before working through this section, it may be helpful to first read the [pdf notes](https://www.depthfirstlearning.com/assets/svgd_notes/week01.pdf) in Part 1 of the DFL course \[[9](#References)\]. The optional reading section of Part 1 in the DFL course \[[9](#References)\] also contains further information on linear algebra, measure theory, and kernel functions. This assignment has been constructed in such a way that you do not require any background in measure theory, and we side-step measure theoretic concepts by focusing the development of SVGD in terms of distributions, rather than measures. Furthermore, we introduce relevant background on kernels and reproducing kernel Hilbert spaces, and provide references to further reading on these topics.
#

# Stein Variational Gradient Descent (SVGD) \[[1](#References)\] is a recently proposed variational inference algorithm with roots in Stein's method in theoretical statistics. In a nutshell, SVGD starts with an initial set of sample points (called particles) $\{x_i\}_{i=1}^n$ which are drawn from an arbitrary, simple initial distribution (e.g. the standard normal distribution), and iteratively transports/moves these particles to better approximate the target distribution $p(x)$. This is achieved by, in each iteration, updating the particles along a small perturbation of the identity map:
# $$x_i \leftarrow x_i + \epsilon \phi(x_i)$$
# where $\phi(x)$ determines the update direction of the particles and $\epsilon$ is a step size.
#
# Before delving into the SVGD algorithm, some background information is needed, which we will discuss here.

# ### Stein's lemma

# The starting point for Stein's method is Stein's lemma for characterising the standard normal distribution, which states that, if $Z \sim \mathcal{N}(0,1)$, then the following holds for all absolutely continuous functions:
# $$\mathbb{E}_{Z \sim \mathcal{N}(0,1)}\left[Z f(Z) \right] = \mathbb{E}_{Z \sim \mathcal{N}(0,1)} \left[f'(Z) \right] \enspace.$$
#
# Stein showed that the above holds for no other distribution of $Z$, i.e. the above holds if and only if $Z \sim \mathcal{N}(0,1)$.
#
# For a quick introduction to absolutely continuous functions, see the [University of Alberta lecture slides](https://sites.ualberta.ca/~rjia/Math418/Notes/chap3.pdf). You do not have to concern yourself with the mathematical details of absolute continuity, and it will suffice to think of absolute continuity as simply a stronger form of continuity, where the function is continuous **and** differentiable almost everywhere. That is, all absolutely continuous functions are continuous, but not all continuous functions are absolutely continuous.
#
#
#

# <div class="alert alert-block alert-info">
#
# **Question 1.1** Let $Z \sim \mathcal{N}(0,1)$ be standard normally distributed and let $\phi(z) = \frac{1}{\sqrt{2 \pi}} \exp \left(- \frac{z^2}{2} \right)$ be the probability density function (pdf) of $Z$. Derive the following property of the standard normal pdf:
#
# $$\phi'(z) = - z \phi (z)$$
#
# **Answer** 
#

# <div class="alert alert-block alert-info">
#
# **Question 1.2** Prove Stein's lemma (Hint: use integration by parts and the fact that $\phi'(z) = - z \phi(z)$).
#
# **Answer** 
#
#

# ### 1.1 Integral Probability Metrics

# A probability metric is a distance function between two distributions that satisfies the usual properties of a metric. The details of probability metrics and integral probability metrics (IPMs) are not important for this assignment, and it will suffice to think of a probability metrics and IPMs as types of distance functions that quantify the similarity/dissimilarity between two distributions. For a brief introduction to IPMs, read the [Wikipedia](https://en.wikipedia.org/wiki/Integral_probability_metric) entry on IPMs.
#
# Integral probability metrics (IPMs) are a special case of probability metrics that are defined by the difference between the integrals (or expectations) over two distributions. That is, let $Q$ and $P$ be two distributions with the same support $\mathcal{X}$, then an IPM for quantifying the similarity/dissimilarity between $Q$ and $P$ takes the form:
#
# $$
# \begin{align*}
# d_{\mathcal{H}}(Q, P) &= \underset{h \in \mathcal{H}}{\sup}\left|\int_{x \in \mathcal{X}} hdQ - \int_{x \in \mathcal{X}} hdP \right|\\
# &= \underset{h \in \mathcal{H}}{\sup}\left| \mathbb{E}_{X \sim Q} \left[h(X) \right] - \mathbb{E}_{Y \sim P} \left[h(Y) \right] \right|
# \end{align*}
# $$
# where $\mathcal{H}$ is a class of bounded, continuous test functions. That is, $\mathcal{H}$ is a class of functions that we use to "test" if the distributions are equal.
#
# In simple terms, an IPM measures the distance between two distributions by considering the function $h \in \mathcal{H}$ that yields the largest difference in expectations between the two distributions. This is based on the assumption that, if two distributions are equal ($P = Q$), then these distributions should yield the same expected value for all functions $h \in \mathcal{H}$, i.e. if $P = Q$ then $\mathbb{E}_{X \sim P}[h(X)] = \mathbb{E}_{X \sim Q}[h(X)]$ for all $h \in \mathcal{H}$. If the distributions are not equal, then there must exist at least one $h \in \mathcal{H}$ such that the difference in expectations is non-zero. 
#
# $\textbf{Remark}$: The class of functions $\mathcal{H}$ determines/induces the IPM. That is, different choices of $\mathcal{H}$ yield different IPMs. The class of functions $\mathcal{H}$ should be rich enough to detect differences between $P$ and $Q$, but should also be simple/small enough such that the IPM is tractable to compute.
#
# If the class of functions $\mathcal{H}$ is sufficiently rich, we would have that $d_{\mathcal{H}}(Q, P) = 0 \iff Q = P$.
#

# ### 1.2 Kernels and reproducing kernel Hilbert spaces

# #### 1.2.1 Motivation and intuition for kernels

# Suppose we wish to fit a simple linear regression model to a set of data $\mathcal{D} = \{(x_i, y_i): i=1,2,\dots,N\}$, where $x_i \in \mathbb{R}$ denotes the observations of the predictors, and $y_i \in \mathbb{R}$ denotes the observations of the response variable. Furthermore, suppose that we observe that the relationship between $x_i$ and $y_i$ is non-linear. In this case, a simple linear model $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$ would yield unsatisfactory predictive performance. We can improve the model performance by increasing the model complexity/capacity. One approach to do this is to consider enlarging the original input space using a feature map $\phi: \mathbb{R} \rightarrow \mathbb{R}^d$ defined by:
# $$x_i \mapsto \begin{bmatrix} x_i & x_i^2 & x_i^3 & \dots & x_i^d\end{bmatrix}^T$$
#  and then fitting a linear regression model in the enlarged feature space: $\hat{y}_i = \hat{\alpha}_0 + \hat{\pmb{\alpha}}^T \phi(x_i)$<br>
# In this example, the fitted line will be linear in the enlarged feature space, but will be non-linear in the original input space, allowing for a much better fit to the data.
# However, the challenge is that constructing the feature map $\phi: \mathbb{R} \rightarrow \mathbb{R}^d$ demands substantial domain knowledge. To address this issue, one can turn to a kernel function which facilitates the computation of inner products between features after they have been transformed into a higher-dimensional (often infinite-dimensional) space. This is achieved without requiring explicit specification or computation of the feature maps. That is, a kernel function $k(x, x')$ computes the inner product between transformed features $\phi(x)$ and $\phi(x')$, i.e. $k(x, x') = \langle \phi(x), \phi(x') \rangle$, without having to compute the features $\phi(x)$ or $\phi(x')$.
#
# It is helpful to think of a kernel function, $k(x, x')$, as providing a measure of similarity between the feature maps $\phi(x)$ and $\phi(x')$ corresponding to input points $x$ and $x'$. That is, you can think of a kernel function as a measure of similarity between two points $x$ and $x'$ after they have been transformed to some higher-dimensional space.

# #### 1.2.2 Technical details

# The definitions on kernels and reproducing kernel Hilbert spaces given here are taken from [University College London lecture notes](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf) by Arthur Gretton.

# ##### Hilbert spaces

# In this assignment, we will only work with $\mathbb{R}$-Hilbert spaces. For the purposes of this assignment, you can think of a $\mathbb{R}$-Hilbert space $\mathcal{H}$ as a space of real-valued functions that is equipped with an inner product, $\langle \cdot, \cdot \rangle_{\mathcal{H}}: \mathcal{H} \times \mathcal{H} \rightarrow \mathbb{R}$, which also induces a norm on $\mathcal{H}$ given by $\lVert f \rVert_{\mathcal{H}} = \sqrt{\langle f, f \rangle_{\mathcal{H}}}$. You do not have to worry about the technical condition of completeness, which means that every Cauchy sequence in $\mathcal{H}$ converges to an element in $\mathcal{H}$.

# ##### Kernels

# Let $\mathcal{X}$ be a non-empty set. A function $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ is called a kernel if there exists a $\mathbb{R}$-Hilbert space $\mathcal{H}$ and a map $\phi: \mathcal{X} \rightarrow \mathcal{H}$ such that the following holds for all $x, x' \in \mathcal{X}$:
#
# $$k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$$
#
# Note that kernel functions are positive definite, which means that the following holds for all $n \geq 1$, $\forall a_1, a_2, \dots, a_n \in \mathbb{R}$, $\forall x_1, x_2, \dots, x_n \in \mathcal{X}$:
#
# $$\sum_{i=1}^n \sum_{j=1}^n a_i a_j k(x_i, x_j) \geq 0$$
#
# This definition of positive definiteness coincides with that of matrices in linear algebra. Let $K: n \times n$ denote the matrix of kernel evaluations (which is called the Gram matrix) with $(i, j)^{\text{th}}$ element given by $K_{ij} = k(x_i, x_j)$. Then the positive definiteness of the kernel $k(x, x') $implies that:
#
# $$\pmb{a}^T K \pmb{a} \geq 0 \enspace \forall \pmb{a} \in \mathbb{R}^n \setminus \{\pmb{0}\}$$
#
# Furthermore, if the equality above holds if and only if $\pmb{a} = \pmb{0}$, then the kernel $k(x,  x')$ is called strictly positive definite.
#

# ##### Reproducing kernel Hilbert space

# Intuitively, a reproducing kernel Hilbert space (RKHS), $\mathcal{H}$, is a Hilbert space in which the functions are defined by the kernel $k(x, x')$, and which has the following two properties:
#
# 1. The feature map of every point is in the feature space. That is, 
#
# $$k(\cdot, x) \in \mathcal{H} \enspace \forall x \in \mathcal{X}$$
#
# 2. The reproducing property:
#
# $$f(x) = \langle f(\cdot), k(\cdot, x) \rangle_{\mathcal{H}} \enspace \forall f \in \mathcal{H} \forall x \in \mathcal{X}$$
#
# These properties together also imply that:
#
# $$k(x, y) = \langle k(\cdot, x), k(\cdot, y) \rangle_{\mathcal{H}}$$
#
# Here we think of $k(\cdot, x)$ for fixed $x \in \mathcal{X}$ as a function of one argument.
#
# Note that functions in the RKHS can be written as a linear combination of the feature maps, i.e. for all $f \in \mathcal{H}$ and for all $x \in \mathcal{X}$ we can write:
#
# $$f(\cdot) = \sum_{i=1}^m \alpha_i k(x_i, \cdot)$$
#
# See Section 4 of the [University College London lecture notes](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf) for more details on an RKHS.

# ## 2. Stein's method

# Before working through this section, read the [pdf notes](https://www.depthfirstlearning.com/assets/svgd_notes/week02.pdf) in Part 2 of the DFL course \[[9](#References)\]. You can also read the papers in the optional reading section in Part 2 of the DFL course for more information.

# Stein's method is a three-step procedure for bounding the distance, in the form of an IPM, between two distributions $P$ and $Q$. For the remainder of this section, we assume that we are interested in the distance between a true underlying distribution $Q$ with density $q(x)$, and a target distribution $P$ with density $p(x)$. For example, in the context of goodness-of-fit tests, the distribution $Q$ may represent the true, unknown distribution undelying observed data $\{x_i\}_{i=1}^n \sim q(x)$, and we would like to test whether the data could have plausibly arisen from some hypothesised/target distribution $p(x)$. If we had explicit access to both $p(x)$ and $q(x)$ (i.e. if we had closed form expressions for both $p(x)$ and $q(x)$), we could calculate the distance between the distributions using an IPM, $d_{\mathcal{H}}(Q, P)$; assuming the class of functions $\mathcal{H}$ is rich enough to distinguish the distributions $P$ and $Q$, we may use the distance $d_{\mathcal{H}}(Q, P)$ to determine whether $P = Q$. However, we typically do not have explicit access to the true distribution, $q(x)$, underlying the data. Furthermore, the target distribution $p(x)$ may be a complex distribution that we cannot evaluate in closed form, and hence expectations under $p(x)$ are intractable to compute. It is also typically the case in machine learning that we only have access to the unnormalised target density $\tilde{p}(x) = \frac{1}{Z}p(x)$, where $Z$ denotes the normalisation constant, in which case expectations under $p(x)$ are intractable to compute.
#
# We will now show how Stein's method can be used to bound reference IPMs of the form:
#
# $$\delta_1 \leq d_{\mathcal{H}}(Q, P) \leq \delta_2$$
#
# Although it is beyond the scope of this assignment, such bounds can be used to prove that two distributions are equal. Furthermore, such bounds can also be used to prove convergence in, for example, central limit theorems.
#
# In Section 3 of this notebook, we will show that Stein's method also gives rise to a new measure of distance between distributions that does not involve a reference IPM. The resulting measure is the so-called $\textit{Stein discrepancy}$, which directly quantifies the similarity/dissimilarity between two distributions.

# #### Step 1: Stein operator

# The first step in Stein's method involves specifying/constructing a $\textit{Stein operator}$ to characterise the distribution of interest. Here, we assume that we are interested in a $d$-dimensional random variable $X$ supported on a subset of $\mathbb{R}^{d}$, i.e. $\mathrm{supp}(X) \subseteq \mathbb{R}^d$.
#
# Given a target distribution $P$ with density function $p(x)$, a $\textit{Stein operator}$ is an operator, $\mathcal{T}_p: \mathcal{F} \rightarrow \mathbb{R}^d$, acting on functions $f \in \mathcal{F}$, which yields zero expectation under the distribution of interest. That is, the operator $\mathcal{T}_p$ maps all functions $f \in \mathcal{F}$ to zero-mean vectors under the target distribution:
# $$\mathbb{E}_{X \sim P} \left[\mathcal{T}_p f(X) \right] = 0 \forall  f \in \mathcal{F} \iff X \sim P \enspace.$$
# The class of functions for which the above holds is called a $\textit{Stein class}$ for the distribution $P$, and is denoted by $\mathcal{F}(\mathcal{T}_p)$.
#

#
# $\textbf{Example}$: 
#
# _Stein's lemma suggests that a Stein operator for the standard normal distribution, $\mathcal{N}(0,1)$, with probability density function $\phi(z)$ is given by: $\mathcal{T}_\phi f(z) = f'(z) - z f(z)$ since we would have $\mathbb{E}[f'(Z) - Z f(Z)] = 0 \hspace{0.1cm}$ for all absolutely continuous functions. Here, the Stein class is given by $\mathcal{F}(\mathcal{T}_\phi) = \mathcal{C}^1(\mathbb{R})$, the class of continuous, real-valued functions with a continuous first derivative._
#

# For the remainder of this assignment, we will only be working with a special class of Stein operators called $\textit{Langevin-Stein operators}$. For a vector-valued function $f: \mathcal{X} \rightarrow \mathbb{R}^{d'}$, the Langevin-Stein operator is defined by:
#
# $$
# \begin{align*}
# \mathcal{T}_p f(x) &= \nabla_x \log p(x) f(x)^T + \nabla_x f(x)\\
# &= s_p(x) f(x)^T + \nabla_x f(x)
# \end{align*}
# $$
#
# where $s_p(x) := \nabla_x \log p(x)$ is called the $\textit{Stein score function}$ for the target distribution $p(x)$. Recall that, we assume that the random variable $X$ is $d$-dimensional with support given by $\mathrm{supp}(X) \subseteq \mathbb{R}^{d}$, and hence we have that $s_p:\mathcal{X} \rightarrow \mathbb{R}^{d}$. Therefore, we have that $\mathcal{T}_p f$ is a $d \times d'$ matrix-valued function (since $\nabla f$ is a $d \times d'$ matrix-valued function and the outer product $s_p f^T$ is also a $d \times d'$ matrix-valued function).
#
# _In the case of a scalar-valued function $f: \mathcal{X} \rightarrow \mathbb{R}$, the Langevin-Stein operator is given by:_
# $$
# \begin{align*}
# \mathcal{T}_p f(x) &= \nabla_x \log p(x) f(x) + \nabla_x f(x)\\
# &= s_p(x) f(x) + \nabla_x f(x)
# \end{align*}
# $$
#
# _Here, $\nabla f$ and $\mathcal{T}_p f$ are $d \times 1$ vector-valued functions._
#
# The Stein class corresponding to a Langevin-Stein operator is called a Langevin-Stein class, and is given by the collection of functions $f: \mathcal{X} \rightarrow \mathbb{R}$ that satisfy:
#
# $$\int_{x \in \mathcal{X}} \nabla_x \left[p(x)f(x) \right]dx = 0 \enspace.$$
#
# In the case of a vector-valued function $f: \mathcal{X} \rightarrow \mathbb{R}^{d'}$ of the form $f(x) = \begin{bmatrix}f_1(x) & f_2(x) & \dots & f_{d'}(x) \end{bmatrix}^T$, the function $f$ is said to be in the Langevin-Stein class of a distribution $P$ if each component function $f_i$ is in the Langevin-Stein class of $P$ as defined above.

# <div class="alert alert-block alert-info">
#
# **Question 2.1** Show that the definition of a Langevin-Stein class coincides with the general definition of a Stein class when using the Langevin-Stein operator. That is, prove that:
#
# $$\mathbb{E}_{X \sim P}\left[\mathcal{T}_p f(X) \right] = 0 \iff \int_{x \in \mathcal{X}} \nabla_x \left[f(x)p(x) \right]dx = 0$$
#
# when $\mathcal{T}_p := s_p(x)f(x) + \nabla_x f(x)$ is the Langevin-Stein operator. Here, you need only consider scalar-valued functions $f: \mathcal{X} \rightarrow \mathbb{R}$.
#
# Hint: Start with the LHS and write it as an integral, then simplify the expression to obtain the RHS.
#
# **Answer** 
#

# <div class="alert alert-block alert-info">
#
# **Question 2.2** Show that the Stein operator for the standard normal distribution is a Langevin-Stein operator. Recall the Stein operator for the standard normal distribution is given by $\mathcal{T}_{\phi} f(z) = f'(z) - z f(z)$. Note that, in the case of scalar-valued random variables and functions, the gradient operator is replaced by the derivative operator (i.e. $\nabla_z f(z)$ becomes $f'(z)$ and $\nabla_z \log p(z)$ becomes $\frac{d}{d z} \log p(z)$).
#
# **Answer** 
#

# #### Step 2: Stein Equation

# The second step in Stein's method involves the Stein equation, which is a differential equation that arises in probability theory. There are a lot of techical details regarding the Stein equation which we do not discuss here as it is beyond the scope of this assignment. 
#
# Given a Stein operator, $\mathcal{T}_p$, and corresponding Stein class $\mathcal{F}(\mathcal{T}_p)$ for the target distribution $p(x)$, the Stein equation is given by:
#
# $$\mathcal{T}_p f_h(x) = h(x) - \mathbb{E}_{Y \sim P}[h(Y)]$$
#
# Here, $\mathcal{H}$ is a class of test functions as in the definition of an IPM, and $f_h$ is the solution we seek. 
#
# Given the Stein equation above, the second step in Stein's method involves proving that a solution $f_h \in \mathcal{F}(\mathcal{T}_p)$ exists for every bounded, continuous test function $h \in \mathcal{H}$. Proving that such a solution exists is beyond the scope of this assignment, and we will assume that a solution $f_h \in \mathcal{F}(\mathcal{T}_p)$ exists for every bounded, continuous test function $h \in \mathcal{H}$. Furthermore, let $\mathfrak{F} \subseteq \mathcal{F}(\mathcal{T}_p)$ denote the function class of solutions to Stein's equation, i.e. 
# $$\mathfrak{F} = \{f_h \in \mathcal{F}(\mathcal{T}_p) | f_h \text{ is a solution to Stein's equation}\}$$
#
# You do not have to concern yourself with the technical details of Stein's equation. For the purposes of this assignment, you can take Stein's equation as a given, and assume that we always have the class of solutions $\mathfrak{F}$. However, note that, since the IPM depends on the function class $\mathcal{H}$, bounding the reference IPM requires that a solution $f_h \in \mathcal{F}(\mathcal{T}_p)$ to Stein's equation exists for all $h \in \mathcal{H}$.

# #### Step 3: Obtaining Bounds

# The third step in Stein's method involves bounding the reference IPM, $d_{\mathcal{H}}(Q, P)$. We now assume that a solution $f_h \in \mathcal{F}(\mathcal{T}_p)$ exists for every test function $h \in \mathcal{H}$. If we replace the quantity $x$ in Stein's equation with the random variable $X$, and take expectations with respect to the true distribution, $Q$, we arrive at:
# $$\mathbb{E}_{X \sim Q}\left[\mathcal{T}_p f(X) \right] = \mathbb{E}_{X \sim Q}\left[h(X) \right] - \mathbb{E}_{Y \sim P}\left[h(Y) \right]$$
#
# Notice that, if we ignore the absolute value, the right hand side of the above equation is the same as the expression given in the supremum ($\sup$) of an IPM. Hence, this allows us to rewrite the IPM as:
#
# $$
# \begin{align*}
# d_{\mathcal{H}}(Q, P) &= \underset{h \in \mathcal{H}}{\sup}\left| \mathbb{E}_{X \sim Q} \left[h(X) \right] - \mathbb{E}_{Y \sim P} \left[h(Y) \right] \right|\\
# &= \underset{f_h \in \mathfrak{F}}{\sup} \left|\mathbb{E}_{X \sim Q}\left[\mathcal{T}_p f_h(X) \right] \right|\\
# & \leq \underset{f_h \in \mathcal{F}(\mathcal{T}_p)}{\sup} \left|\mathbb{E}_{X \sim Q}\left[\mathcal{T}_p f_h(X) \right] \right| \quad \text{(since $\mathfrak{F} \subseteq \mathcal{F}(\mathcal{T}_p)$)}
# \end{align*}
# $$
#
# Furthermore, if we assume that $\mathfrak{F}$ is closed under negation, i.e. $f \in \mathfrak{F} \implies -f \in \mathfrak{F}$, then we can ignore the absolute value above (this is also true for $\mathcal{H}$ and $\mathcal{F}(\mathcal{T}_p)$).

# $\textbf{Example}$:
#
#  _Suppose we have a set of observed data $\{x_i\}_{i=1}^n \sim q(x)$, and we want to determine whether the data could have plausibly arisen from the standard normal distribution, $\mathcal{N}(0,1)$. That is, we would like to determine whether $\{x_i\}_{i=1}^n \sim \mathcal{N}(0,1)$. Recall that the Stein operator for the standard normal distribution is given by $\mathcal{T}_\phi f(z) = f'(z) - z f(z)$. In this case, Stein's method can be used to simplify the IPM as follows:_
#
# $$
# \begin{align*}
# d_{\mathcal{H}}(Q, \mathcal{N}(0,1)) &= \underset{h \in \mathcal{H}}{\sup} \left|\mathbb{E}_{X \sim Q}\left[h(X) \right] - \mathbb{E}_{Y \sim \mathcal{N}(0, 1)}\left[h(Y) \right] \right| \quad \text{(by defn of an IPM)}\\
# &= \underset{f_h \in \mathfrak{F}}{\sup}\left|\mathbb{E}_{X \sim Q}\left[\mathcal{T}_{\phi}f_h(X) \right] \right| \quad \text{(by Stein's equation)} \\
# &= \underset{f_h \in \mathfrak{F}}{\sup}\left|\mathbb{E}_{X \sim Q}\left[f_h'(X) - X f_h(X) \right] \right| \quad \text{(by defn of the Stein operator for $\mathcal{N}(0,1)$ )}\\
# & \leq \underset{f \in \mathcal{C}^1(\mathbb{R})}{\sup}\left|\mathbb{E}_{X \sim Q} \left[f'(X) - X f(X) \right] \right|
# \end{align*}
# $$

# <div class="alert alert-block alert-info">
#
# **Question 2.3** Explain in your own words how Stein's method simplifies the task of bounding the distance (in the form of an IPM) between two distributions $P$ and $Q$. That is, why do you think it would be easier to bound the expression:
#
# $$\underset{f_h \in \mathfrak{F}}{\sup} \left|\mathbb{E}_{X \sim Q}\left[\mathcal{T}_p f_h(X) \right] \right|$$
#
# than the IPM itself?
#
# **Answer** 
#

# ## 3. Stein Discrepancy

# The previous section showed how Stein's method simplifies the task of bounding reference IPMs, $d_{\mathcal{H}}(Q, P)$. However, Stein's method also gives rise to a new measure of distance between two distributions that does not involve a reference IPM. The resulting measure of distance is the so-called $\textit{Stein discrepancy}$.
#
# Before we present the Stein discrepancy, we require the notion of $\textit{Stein's identity}$, which can be viewed as a generalisation of Stein's lemma to distributions beyond the standard normal. 
#
# Given a distribution $P$ with continuous, differentiable density function $p(x)$, and the associated Langevin-Stein operator $\mathcal{T}_p$ and Langevin-Stein class $\mathcal{F}(\mathcal{T}_p)$, $\textit{Stein's identity}$ for vector-valued functions $f: \mathcal{X} \rightarrow \mathbb{R}^{d'}$ states that:
#
# $$\mathbb{E}_{X \sim P} \left[\mathcal{T}_p f(X) \right] := \mathbb{E}_{X \sim P} \left[s_p(X)f(X)^T + \nabla_X f(X) \right] = 0 \enspace \forall f \in \mathcal{F}(\mathcal{T}_p) \enspace.$$
#
# In the case of scalar-valued functions $f: \mathcal{X} \rightarrow \mathbb{R}$, Stein's identity states that:
#
# $$\mathbb{E}_{X \sim P} \left[\mathcal{T}_p f(X) \right] := \mathbb{E}_{X \sim P} \left[s_p(X)f(X) + \nabla_X f(X) \right] = 0 \enspace \forall f \in \mathcal{F}(\mathcal{T}_p) \enspace.$$
#

# <div class="alert alert-block alert-info">
#
# **Question 3.1** Prove Stein's identity for scalar-valued functions $f: \mathcal{X} \rightarrow \mathbb{R}$ given by:
#
# $$\mathbb{E}_{X \sim P} \left[\mathcal{T}_p f(X) \right] = 0 \enspace \forall f \in \mathcal{F}(\mathcal{T}_p)$$
#
# (Hint: First write the expectation as an integral and simplify, and then use the definition of a Langevin-Stein class.)
#
# **Answer** 
#

# If we consider Stein's identity when taking expectations with respect to another distribution $Q \not= P$, with the same support as $P$, then there must exist a function $f$ for which Stein's identity fails to hold when taking expectations with respect to $Q$. That is, for vector-valued functions $f: \mathcal{X} \rightarrow \mathbb{R}^{d'}$, we have that:
#
# $$Q \not= P \implies \exists f \in \mathcal{F}(\mathcal{T}_p) \enspace s.t. \enspace \mathbb{E}_{X \sim Q}\left[s_p(X)f(X)^T + \nabla_X f(X) \right] \not= 0 \enspace.$$
#
# This suggests that we may be able to define a notion of distance between $Q$ and $P$ by considering the extent to which Stein's identity is violated. However, since this quantity depends on the choice of function $f$, we consider the function $f$ that yields the "maximum violation of Stein's identity" \[[1](#References)\], which then gives rise to a $\textit{Stein discrepancy}$ defined below.
#
# Let $P$ and $Q$ be distributions supported on $\mathcal{X}$ with continuous, differentiable densities $p(x)$ and $q(x)$ respectively. Let $\mathcal{T}_p$ be the Langevin-Stein operator for $P$ and let $\mathcal{F}_q$ be a class of continuously differentiable vector-valued functions that satisfy Stein's identity. The Stein discrepancy from $q$ to $p$ with respect to $\mathcal{F}_q$ is defined as:
#
# $$
# \begin{align*}
# \mathbb{D}_{\mathrm{Stein}}(q, p; \mathcal{F}_q) :&= \underset{f \in \mathcal{F}_q}{\sup} \mathbb{E}_{X \sim Q} \left[ \mathrm{trace}\left(\mathcal{T}_p f(X) \right) \right]^2\\
# &= \underset{f \in \mathcal{F}_q}{\sup} \mathbb{E}_{X \sim Q} \left[\mathrm{trace}\left\{\left(s_p(X) - s_q(X) \right)f(X)^T \right\} \right]^2\\
# &= \underset{f \in \mathcal{F}_q}{\sup} \mathbb{E}_{X \sim Q} \left[\mathrm{trace}\left\{\left( s_p(X) - s_q(X) \right)^T f(X)\right \} \right]^2 \quad \text{(since matrix trace is invariant to cyclical permutations)}\\
# &= \underset{f \in \mathcal{F}_q}{\sup} \mathbb{E}_{X \sim Q} \left[\left( s_p(X) - s_q(X) \right)^T f(X) \right]^2 
# \end{align*}
# $$
#
# $\textbf{Remark}$: The definition of Stein discrepancy above uses the trace to map the vector-valued function $\mathcal{T}_p f(x)$ to a scalar-valued function.

# <div class="alert alert-block alert-info">
#
# **Question 3.2** The definition of Stein discrepancy given above relies on an important result which you will prove here. Let $P$ and $Q$ be distributions with continuous, differentiable densities $p$ and $q$ respectively. Suppose that Stein's identity holds for distribution $Q$ with Langevin-Stein operator $\mathcal{T}_q$, i.e. $\mathbb{E}_{X \sim Q}\left[\mathcal{T}_q f(X) \right] = 0$ for all functions $f \in \mathcal{F}(\mathcal{T}_q)$ in the Langevin-Stein class of $Q$. Suppose that $f: \mathcal{X} \rightarrow \mathbb{R}^{d'}$ is a vector-valued function in the Langevin-Stein class of $P$. Prove that the following relationship holds when taking expectations with respect to $Q$ in Stein's identity:
#
# $$\mathbb{E}_{X \sim Q}\left[\mathcal{T}_p f(X) \right] = \mathbb{E}_{X \sim Q} \left[\left(s_p(X) - s_q(X) \right) f(X)^T \right]$$
#
# where $s_q(x) := \nabla_x \log q(x)$ is the Stein score function for $Q$. 
#
# (Hint: use Stein's identity for $Q$.)
#
# **Answer** 
#

# The class of functions $\mathcal{F}_q$ in the definition of Stein discrepancy should be chosen to be (i) rich enough to contain functions that can discriminate $Q$ from $P$ (i.e. rich enough to ensure that $\mathbb{D}_{\mathrm{Stein}}(q, p; \mathcal{F}_q) > 0$ whenever $p \not= q$), but should also be chosen to be (ii) simple/small enough such that the optimisation problem is computationally tractable.

# <div class="alert alert-block alert-info">
#
# **Question 3.3**  Explain in your own words how the requiremens for $\mathcal{F}_q$ in Stein discrepancy are related to the requirements for the class of distributions, $\mathcal{Q}$, in variational inference.
#
# **Answer** 
#

# ## 4. Kernelised Stein Discrepancy

# The form of Stein discrepancy presented in the previous section is problematic for two reasons: firstly, we have to explicitly specify the class of functions $\mathcal{F}_q$ over which the optimisation is performed, and secondly, computing the Stein discrepancy may still be intractable (unless we carefully design the class of functions $\mathcal{F}_q$ to allow tractable computations; unfortunately, this is a very difficult task and may not always be possible). This section introduces a tractable form of Stein discrepancy known as $\textit{kernelised Stein discrepancy}$ (KSD). For a quick introduction to KSD, read [[Short intro to KSD - Qiang Liu]](https://www.cs.utexas.edu/~lqiang/PDF/ksd_short.pdf).
#
# To overcome these problems, Liu et al. \[[2](#References)\] takes $\mathcal{F}_q$ to be the (closed) unit ball in a reproducing kernel Hilbert space (RKHS) $\mathcal{H}$, corresponding to a positive definite kernel $k(x,x')$. I.e. Liu et al. \[[2](#References)\] takes $\mathcal{F}_q$ to be $\mathcal{B}^1 = \{f \in \mathcal{H}: \lVert f \rVert_{\mathcal{H}}^2 \le 1 \}$. This then gives rise to the $\textit{kernelised Stein discrepancy}$ (KSD) defined as:
#
# $$\mathbb{S}(q || p) := \mathbb{E}_{X, X' \sim Q} \left[\left(s_p(X) - s_q(X) \right)^T k(X, X') \left(s_p(X') - s_q(X') \right) \right] \enspace.$$
#
# Under certain mild assumptions on the kernel $k(x, x')$ - the technical details of these assumptions are not important for this assignment; however, note that the assumptions are always valid for the RBF kernel - Liu et al. \[[2](#References)\] derive a closed form expression for KSD given by:
#
# $$\mathbb{S}(q || p) = \mathbb{E}_{X, X' \sim Q}\left[\kappa_p(X, X') \right]$$
#
# where $\kappa_p (x, x')$ is called a "Steinalised kernel" \[[10](#References)\] and is given by:
#
# $$\kappa_p(x, x') := s_p(x)^T k(x, x') s_p(x') + s_p(x)^T \nabla_{x'}k(x, x') + \nabla_x k(x, x')^Ts_p(x') + \mathrm{trace}(\nabla_x \nabla_{x'}k(x, x')) \enspace.$$
#
# This form of KSD is very convenient for several reasons: firstly, we no longer have to explicitly specify the class of functions $\mathcal{F}_q$, and we need only specify a positive definite kernel function $k(x, x')$; secondly, the KSD can be computed in closed form using the "Steinalised kernel"; thirdly, this form of KSD does not require calculation of the score function $s_q$ of the distribution $Q$ (this is useful since we generally do not know the distribution $Q$ and only have access to the distribution via a sample $\{x_i\}_{i=1}^n \sim q(x)$, and hence the score function cannot be computed exactly).
#
# $\textbf{Remark}$: The "Steinalised kernel" above allows us to get rid of the score function of the distribution $Q$, $s_q$, in the calculation of KSD and can be obtained by applying the Langevin-Stein operator to the kernel $k(x, x')$ twice, once for each argument. That is, $\kappa_p(x, x') = \mathcal{T}_p^{x}\left(\mathcal{T}_p^{x'}k(x, x') \right)$, where $\mathcal{T}_p^x$ denotes the Langevin-Stein operator with respect to $x$.
#
# In practice, we can use a $\textit{U-statistic}$ - see the [Wikipedia entry](https://en.wikipedia.org/wiki/U-statistic#:~:text=In%20statistical%20theory%2C%20a%20U,producing%20minimum%2Dvariance%20unbiased%20estimators.) on U-statistics -  to estimate the expectation with respect to the unknown distribution $Q$, which then yields the following unbiased estimate of KSD:
#
# $$
# \begin{align*}
# \mathbb{\hat{S}}(q || p) &= \mathbb{\hat{E}}_{X, X' \sim Q}\left[\kappa_p(X, X') \right]\\
# &= \frac{1}{n(n-1)} \sum_{i=1}^n \sum_{j \not= i} \kappa_p(x_i, x_j)
# \end{align*}
# $$
#
# This allows us to compare an unknown distribution $Q$ to arbitrary target distributions $P$ using only a sample $\{x_i\}_{i=1}^n \sim q(x)$ from the distribution $Q$ and the score function of the target distribution, $s_p$.
#

# ## 5. Stein Variational Gradient Descent

# ### 5.1 Algorithm definition

# In the previous sections, we discussed how Stein discrepancy and its kernelised variant, KSD, can be used to compare two distributions $P$ and $Q$. We now discuss the Stein variational gradient descent (SVGD) algorithm, which is used to incrementally transform a distribution $Q$ into the target distribution $P$.
#
# SVGD is a non-parametric, deterministic, particle-based variational inference algorithm that allows approximation of (or sampling from) intractable target distributions. SVGD works by iteratively transporting a set of particles to approximate the target distribution. In each iteration, the update directions of the particles are chosen from a suitable class of functions to maximise the reduction in reverse KL divergence between the particle and target distributions. Hence, SVGD performs "a type of functional gradient descent on the KL divergence" \[[5](#References)\].
#
# SVGD starts with an initial set of particles $\{x_i\}_{i=1}^n$ drawn from an arbitrary initial distribution $q_0$ (e.g. the standard normal distribution), and iteratively updates each of the particles via a perturbed identity map given by:
# $$T(x_i) = x_i + \epsilon \phi(x_i)$$
# where $\epsilon$ is a step size and $\phi(\cdot)$ is a $\textit{velocity field}$ that determines the update direction - don't worry too much about what a velocity field is, and simply think of $\phi$ as a 'function' that determines the update direction for the purposes of this assignment. In each iteration, the velocity field $\phi$ is chosen from a suitable class of continuously differentiable functions $\mathcal{F}$ to maximise the rate of decay in the reverse KL divergence. Specifically, 
# $$\phi(x) = \underset{\phi \in \mathcal{F}}{\arg \max} \left\{ - \frac{d}{d \epsilon}\mathbb{D}_{\mathrm{KL}}(q_{[\epsilon \phi]} || p) \big|_{\epsilon=0}\right\}$$
# where $q_{[\epsilon \phi]}$ denotes the density represented by the updated particles $x' = x + \epsilon \phi(x)$ when $x \sim q(x)$, which can be obtained via the change-of-variables formula (assuming $\epsilon$ is small enough such that the perturbed identity map is invertible):
# $$q_{[\epsilon \phi]}(x') = q(T^{-1}(x))\cdot |\det J_{T^{-1}}(x)|$$
#
# As was done in KSD, we again take $\mathcal{F}$ to be a closed ball in a reproducing kernel Hilbert sapce (RKHS) $\mathcal{H}$ corresponding to a positive definite kernel $k(x, x')$, but with a different radius given by: $\mathcal{B} = \{\phi \in \mathcal{H}: \lVert \phi \rVert_{\mathcal{H}}^2 \le \mathbb{S}(q||p)\}$. In this case, the update direction has a closed form expression given by:
# $$
# \begin{align*}
# \phi^*(\cdot) &= \underset{\phi \in \mathcal{B}}{\arg \max} \left\{- \frac{d}{d \epsilon}\mathbb{D}_{\mathrm{KL}}(q_{[\epsilon \phi]} || p) \big|_{\epsilon=0} \right\}\\
# &= \mathbb{E}_{X \sim Q} \left[ \mathcal{T}_p k(X, \cdot)\right]\\
# &= \mathbb{E}_{X \sim Q}\left[k(X, \cdot) \nabla_X \log p(X)  + \nabla_X k(X, \cdot) \right]
# \end{align*}
# $$

# <div class="alert alert-block alert-info">
#
# **Question 5.1.1** Derive an expression for the rate at which the KL divergence decreases in a single update step in SVGD. That is, derive an expression for the following optimisation problem:
#
# $$\underset{\phi \in \mathcal{B}}{\max}\left\{- \frac{d}{d \epsilon} \mathbb{D}_{\mathrm{KL}}(q_{[\epsilon \phi]} || p) \big|_{\epsilon=0} \right\}$$
#
# Hint: Your final expression should be related to the KSD. Furthermore, the following decomposition of the KL divergence will come in handy:
# $$\mathbb{D}_{\mathrm{KL}}(q_{[\epsilon \phi]}||p) = \mathbb{D}_{\mathrm{KL}}(q||p) - \epsilon \mathbb{E}_{X \sim Q} \left[ \mathrm{trace}(\mathcal{T}_p \phi(X))\right] + \mathcal{O}(\epsilon^2)$$
#
# If you are unfamiliar with big-O notation (the $\mathcal{O}$ above), read the [Wikipedia entry](https://en.wikipedia.org/wiki/Big_O_notation).
#
# **Answer** 
#

# As before, we can use MC integration to estimate the expectation under the current distribution of the particles, $Q$ with density function $q(x)$, which yields the following unbiased estimate of the optimal update direction:
# $$
# \begin{align*}
# \hat{\phi}^*(\cdot) &= \mathbb{\hat{E}}_{X \sim Q}\left[k(X, \cdot) s_p(X) + \nabla_X \log k(X, \cdot) \right]\\
# &= \frac{1}{n}\sum_{j=1}^n \left[k(x_j, \cdot) \nabla_{x_j}\log p(x_j) + \nabla_{x_j}k(x_j, \cdot) \right]
# \end{align*}
# $$
# where $x_i \overset{i.i.d}{\sim} q(x), i=1,2,\dots,n$.
#
# Given this optimal update direction, SVGD iteratively updates the positions of the particles using the perturbed identity map with optimal perturbation direction $\phi^*$ given by: $T^*(x_i) = x_i + \epsilon \phi^*(x_i)$. That is, SVGD iteratively updates the positions of the particles by:
# $$
# x_i \leftarrow T^*(x_i) = x_i + \epsilon \hat{\phi}^*(x_i)
# $$
#
# Hence, the SVGD update equation for each of the particles is given by:
#
# $$x_i \leftarrow x_i + \frac{\epsilon}{n}\sum_{j=1}^n \underbrace{k(x_j, x_i) \nabla_{x_j} \log p(x_j)}_{\text{driving force}} + \underbrace{\nabla_{x_j}k(x_j, x_i)}_{\text{repulsive force}}, \enspace i=1,2,\dots, n.$$
#
# $\textbf{Remark}$: The update rule in Equation above contains two opposing terms: the first term is a kernel-weighted gradient of the log density of the target distribution, which serves as a driving force \[[1](#References)\] that pushes the particles towards high-density regions of the target distribution; the second term is the gradient of the kernel function, which serves as a repulsive force \[[1, 6](#References)\] that pushes the particles away from each other, thereby encouraging diversity in the particle positions to prevent the particles from collapsing into a single mode of the target density.

# <div class="alert alert-block alert-info">
#
# **Question 5.1.2** Explain the effect of the kernel function on the driving force in the SVGD update equation.
#
# **Answer** 
#

# A defining feature of SVGD is that it provides a spectrum of inference algorithms depending on the number of particles used. When using only a single particle ($n = 1$), and a kernel that satisfies $\nabla_x k(x, x') = 0$ whenever $x = x'$, then SVGD reduces to gradient ascent for maximum a posteriori (MAP) estimation \[[1](#References)\] (assuming $p(x)$ is a posterior, otherwise SVGD reduces to gradient ascent for maximum likelihood estimation). Conversely, in the limit of infinitely many particles ($n \rightarrow \infty$), SVGD becomes a full Bayesian inference algorithm \[[1](#References)\].

# <div class="alert alert-block alert-info">
#
# **Question 5.1.3** Is SVGD with a single particle ($n = 1$) guaranteed to find the global mode of the target distribution?
#
# **Answer** 
#

# <div class="alert alert-block alert-info">
#
# **Question 5.1.4** What are the major differences between SVGD and Markov chain Monte Carlo (MCMC) methods?
#
# **Answer** 
#

# <div class="alert alert-block alert-info">
#
# **Question 5.1.5** What are the major differences between SVGD and traditional (parametric) variational inference (VI) methods?
#
# **Answer** 
#

# The SVGD algorithm is summarised below.
#
# ![](./figures/svgd_algorithm.png)

# ##### Animation of SVGD for sampling from a bivariate Gaussian

# +
HTML("""
<video width="640" height="480" controls>
  <source src="./figures/animation.mp4" type="video/mp4">
</video>
""")


# -

# ### 5.2 SVGD Implementation

# #### Implementing SVGD to sample from a bivariate Gaussian

# We can now implement SVGD to sample from a bivariate Gaussian given by:
# $$P = \mathcal{N}_2(\mu, \Sigma)$$
#
# with probability density function given by:
#
# $$p(x) = \frac{1}{2 \pi |\Sigma|^{\frac{1}{2}}} \exp \left(\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu) \right)$$
#
# where $|\Sigma|$ denotes the determinant of the covariance matrix.
#
# We will use an RBF kernel in the implementation given by:
#
# $$
# \begin{align*}
# k(x, x') &= \exp \left(- \frac{||x - x'||_2^2}{2 \sigma^2} \right)\\
# &= \exp \left(- \frac{(x - x')^T(x - x')}{2 \sigma^2} \right)
# \end{align*}
# $$
#
# where $\sigma$ is called the bandwidth. In the implementation below, the `median_heuristic` function can be used to calculate the median heuristic bandwidth given by:
#
# $$\sigma_{\mathrm{med}} = \frac{med^2}{2 \log( n + 1)}$$
#
# Using the median heuristic bandwidth allows the kernel bandwidth to adapt to the spread of the particles in each iteration.
#
#

# <div class="alert alert-block alert-info">
#
# **Question 5.2.1** Here you will derive the update equation for the above bivariate Gaussian sampling problem by hand. Derive expressions for the following terms of the SVGD update equation:
#
# a.) Score function:
#
# $$s_p(x) = \nabla_x \log p(x)$$
#
# b.) Gradient of kernel function:
#
# $$\nabla_x k(x, x')$$
#
#
# **Answer** 
#

# <div class="alert alert-block alert-info">
#
# Now use the update equation derived above to complete the following functions to implement SVGD. In each iteration of SVGD, use the `median_heuristic` function to calculate the median heuristic bandwidth. You do not have to worry about vectorized operations, and you may use multiple nested for loops for this first 'vanilla' implementation.

# +
def median_heuristic(x, y):
    """
    This function is used to calculate the median heuristic bandwidth
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    dnorm2 = -2 * x.matmul(y.t()) + x.matmul(x.t()).diag().unsqueeze(1) + y.matmul(y.t()).diag().unsqueeze(0)
    dnorm2_np = dnorm2.detach().cpu().numpy()
    h = np.median(dnorm2_np) / (2 * np.log(x.size(0) + 1))
    return np.sqrt(h).item()

def k(x, y, sigma=1):
    #TODO: implement the RBF kernel
    return k_xy

def score_fn(x, mu, cov):
    """ 
    Score function for the target distribution
    """
    #TODO: implement function to calculate the score function of the target distribution
    #HINT: do not use np.linalg.inv to compute the inverse covariance matrix, rather use np.linalg.solve
    return sp_x

def dk_x(x, y, sigma=1):
    #TODO: implement function to calculate derivative of RBF kernel w.r.t first argument: dk(x, y)/dx
    return grad_k


def svgd(particles, mu, cov, max_steps, eps):
    """
    This function implements the SVGD algorithm.
    Args:
        - particles ()
    """
    #TODO: implement SVGD algorithm
    #NOTE: in each iteration, use the median_heuristic function to compute the bandwidth, sigma. 
    #      I.e. use sigma=median_heuristic(particles, particles) in each iteration, and pass this sigma to the kernel.
    return particles


# +
true_mean = np.array([-1, 3])
temp = np.array([[3, 2], [2, 3]])
true_cov = temp.T @ temp

np.random.seed(42)
temp = np.random.normal(0, 5, size=4).reshape([2, 2])
test_cov = temp.T @ temp
test_mean = np.random.normal(0, 5, size=2)
x = np.random.multivariate_normal(mean=test_mean, cov=test_cov, size=20)
initial_particles = x.copy()

z = svgd(particles=x.copy(), mu=true_mean, cov=true_cov, max_steps=10000, eps=0.1)

# +
# Run this chunk to plot the results
x_range = np.linspace(-15, 15, 400)
y_range = np.linspace(-15, 15, 400)
X, Y = np.meshgrid(x_range, y_range)
pos = np.dstack((X, Y))

rv = multivariate_normal(true_mean, true_cov)
Z = rv.pdf(pos)

plt.contourf(X, Y, Z, levels=20, cmap='viridis')
cbar = plt.colorbar()
plt.scatter(initial_particles[:, 0], initial_particles[:, 1], label="initial")
plt.scatter(z[:, 0], z[:, 1], label="final")
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bivariate Normal Contour Plot')
plt.show()

# -

# The resulting figure should look something like the following:

Image("./figures/gaussian.png")


# #### SVGD to sample from a GMM using PyTorch

# <div class="alert alert-block alert-info">
#
# Now that you have completed a basic, 'vanilla' implementation of SVGD from scratch, we now use PyTorch and its automatic differentiation capabilities to implement SVGD to sample from a bivariate Gaussian mixture model (GMM). You will need to complete the methods marked with TODO's in the code below.

# +
def k(x, y, sigma=None):
    """
    This function implements the RBF kernel and automatically calculates median heuristic bandwidth
    """
    dnorm2 = -2 * x.matmul(y.t()) + x.matmul(x.t()).diag().unsqueeze(1) + y.matmul(y.t()).diag().unsqueeze(0)
    if sigma is None:
        dnorm_np = dnorm2.detach().cpu().numpy()
        h = np.median(dnorm_np) / (2 * np.log(x.size(0) + 1))
        sigma = np.sqrt(h).item()
    
    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    return (-gamma * dnorm2).exp()

class SVGD:
    """
    This class is used to implement Stein Variational Gradient Descent (SVGD)

    Attributes:
        - P (torch distribution): target distribution to approximate using SVGD
        - K (function): kernel function
        - optimizer (torch.optim optimizer): optimizer to use for step size selection

    Methods:
        - phi_fn(X, alpha): function to calculate optimal perturbation direction
        - step(X, alpha): function to implement a gradient descent step using gradient returned by phi_fn
    #TODO: complete the phi_fn method below. Your solution should not contain any for loops.
    """

    def __init__(self, P, K, optimizer):
        self.P = P
        self.K = K
        self.optimizer = optimizer
    
    def phi_fn(self, X):
        X = X.detach().requires_grad_(True)
        return phi
    
    def step(self, X):
        self.optimizer.zero_grad()
        X.grad = -self.phi_fn(X)
        self.optimizer.step()

    def run(self, X, max_steps, tol=1e-2):
        for t in tqdm(range(max_steps)):
            self.step(X)
            phi_norm = torch.linalg.norm(X.grad)
            if torch.abs(phi_norm) <= tol:
                print(f"Algorithm converged after {t + 1} iterations")
                break
        return X
    


# +
temp = np.array([[0.6, 0.4], [0.4, 1.7]])
cov = temp.T @ temp
cov1 = np.linalg.inv(cov)
covariance_matrices = [torch.from_numpy(cov), torch.from_numpy(cov1)]
means = [torch.zeros(2), torch.zeros(2)]
weights = np.array([0.5, 0.5])

gmm = gmm.GMM(covariance_matrices=covariance_matrices, means=means, weights=torch.from_numpy(weights))

torch.manual_seed(42)
NUM_SAMPLES = 100

initial_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(2), covariance_matrix=2*torch.eye(2))
initial_particles = initial_dist.sample([NUM_SAMPLES])
OPTIMIZER = optim.Adam([initial_particles], lr=1e-2)

svgd = SVGD(P=gmm, K=k, optimizer=OPTIMIZER)
svgd_samples = svgd.run(initial_particles, max_steps=500)
# -

# run this code to plot the results
gmm_chart = plotting_code.get_density_chart(gmm)
gmm_chart + plotting_code.get_particles_chart(svgd_samples)

# Your figure should look something like the figure below (in fact, the figures should be (at least almost) identical since SVGD is deterministic.):

Image("./figures/svgd_gmm.png")

# ### 5.3 Major limitations of SVGD

# #### Mode/Variance Collapse

# The major limitation of SVGD is the so-called mode/variance collapse phenomenon \[[6, 7](#References)\], which refers to the
# situation in which the SVGD particles collapse onto a single mode of the target distribution. When the particles experience mode collapse, the variance of the particles drastically underestimates the variance of the target distribution \[[6](#References)\], in which case the particles fail to explain the uncertainty in the target distribution. Read the papers "Understanding the variance collapse of SVGD in high dimensions" \[[6](#References)\] and "Message passing Stein variational gradient descent" \[[8](#References)\] to understand the mode/variance collapse phenomenon of SVGD.

# <div class="alert alert-block alert-info">
#
# **Question 5.3.1** Explain in your own words why the mode/variance collapse phenomenon is problematic.
#
# **Answer** 
#

# <div class="alert alert-block alert-info">
#
# **Question 5.3.2** Explain in your own words the possible cause(s) of the mode/variance collapse phenomenon in high dimensions.
#
# **Answer** 
#

# <div class="alert alert-block alert-info">
#
# **Question 5.3.3** Think of or research some approaches that help to alleviate the mode/variance collapse phenomenon of SVGD, and give a **brief** overview of one or two such approaches.
#
# **Answer** 
#

# #### Other limitations

# Some other limitations of SVGD include:
#
# 1. Sensitivity to choice of kernel function. The performance of SVGD depends heavily on the choice of kernel function, where the optimal kernel function cannot be determined $\textit{a priori}$.
# 2. Only works for continuous and differentiable target distributions. However, there are extensions of vanilla SVGD that works for non-differentiable and discrete target distributions.
#

# <div class="alert alert-block alert-info">
#
# **Question 5.3.4** Think of or research some approaches that help to alleviate the sensitivity of SVGD to the choice of kernel function.
#
# **Answer** 
#

# # References

# \[1\] Liu, Q. & Wang, D. 2016. Stein variational gradient descent: A general purpose bayesian inference
# algorithm. _In Proceedings of the 30th Conference on Neural Information Processing Systems (NIPS 2016)_. Barcelona, Spain.
# Available at: https://arxiv.org/abs/1608.04471
#
# \[2\] Liu, Q., Lee, J. & Jordan, M. 2016. A Kernelized Stein Discrepancy for Goodness-of-fit Tests. In
# Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings
# of Machine Learning Research, pages 276–284. New York, New York, USA: PMLR.
# Available at: https://proceedings.mlr.press/v48/liub16.html
#
# \[3\] Liu, Q. 2016. Stein variational gradient descent: Theory and applications.
# Available at: https://www.cs.utexas.edu/~lqiang/PDF/svgd_aabi2016.pdf
#
# \[4\] Liu, Y., Ramachandran, P., Liu, Q. & Peng, J. 2017. Stein Variational Policy Gradient. ArXiv,
# abs/1704.02399.
# Available at: https://api.semanticscholar.org/CorpusID:4410100
#
# \[5\] Han, J. & Liu, Q. 2018. Stein variational gradient descent without gradient. In Proceedings of
# the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine
# Learning Research, pages 1900–1908. PMLR.
# Available at: https://proceedings.mlr.press/v80/han18b.html
#
# \[6\] Ba, J., Erdogdu, M.A., Ghassemi, M., Sun, S., Suzuki, T., Wu, D. & Zhang, T. 2022. Understanding
# the variance collapse of SVGD in high dimensions. In International Conference on Learning
# Representations.
# Available at: https://openreview.net/forum?id=Qycd9j5Qp9J
#
# \[7\] D’Angelo, F. & Fortuin, V. 2021. Annealed Stein Variational Gradient Descent. Proceedings of the
# 3rd Symposium on Advances in Approximate Bayesian Inference, pages 1–12.
# Available at: https://arxiv.org/abs/2101.09815
#
# \[8\] Zhuo, J., Liu, C., Shi, J., Zhu, J., Chen, N. & Zhang, B. 2017. Message Passing Stein Variational
# Gradient Descent. In International Conference on Machine Learning.
# Available at: https://arxiv.org/abs/1711.04425
#
# \[9\] B. Mehta. 2020. Stein Variational Gradient Descent. Depth First Learning. AVailable at: https://www.depthfirstlearning.com/2020/SVGD
#
# \[10\] Liu, Q. 2017. Stein Variational Gradient Descent as Gradient Flow. In Advances in Neural
# Information Processing Systems, volume 30. Curran Associates, Inc.
# Available at: https://proceedings.neurips.cc/paper_files/paper/2017/file/17ed8abedc255908be746d245e50263a-Paper.pdf
#
