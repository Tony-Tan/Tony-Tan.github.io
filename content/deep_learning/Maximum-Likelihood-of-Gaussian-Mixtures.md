---
title: 'Maximum Likelihood of Gaussian Mixtures'
tags:
  - machine learning
  - mixture models
  - Gaussian mixtures
  - maximum likelihood
keywords:
  - machine learning
  - mixture models
  - Gaussian mixtures
  - maximum likelihood
date: 2020-03-05 18:54:20
markup: pdc
draft: false
images: ""
url: "/Maximum-Likelihood-of-Gaussian-Mixtures"
---

## Preliminaries
1. Probability Theory
  - multiplication principle
  - joint distribution
  - the Bayesian theory
  - Gaussian distribution
  - log-likelihood function
2. ['Maximum Likelihood Estimation'](https://anthony-tan.com/Maximum-Likelihood-Estimation/)

## Maximum Likelihood[^1]

Gaussian mixtures had been discussed in ['Mixtures of Gaussians'](https://anthony-tan.com/Mixtures-of-Gaussians/). And once we have a training data set and a certain hypothesis, what we should do next is estimate the parameters of the model. Both kinds of parameters from a mixture of Gaussians $\Pr(\mathbf{x})= \sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)$:
- the parameters of Gaussian: $\mathbf{\mu}_k,\Sigma_k$ 
- and latent variables: $\mathbf{z}$

need to be estimated. When we investigated the linear regression problems, ['Maximum Likelihood Estimation'](https://anthony-tan.com/Maximum-Likelihood-Estimation/) method assumed the output of the linear model also has a Gaussian distribution. So, we could try the maximum likelihood again in the Gaussian mixture task, and find whether it could solve the problem.

Notations: 
- input data: $\{\mathbf{x}_1,\cdots,\mathbf{x}_N\}$ for $\mathbf{x}_i\in \mathbb{R}^D$ and $i=\{1,2,\cdots,N\}$ and assuming they are i.i.d. Rearranging them in a matrix:
$$
X = \begin{bmatrix}
  -&\mathbf{x}_1^T&-\\
  -&\mathbf{x}_2^T&-\\
  &\vdots&\\
  -&\mathbf{x}_N^T&-\\
\end{bmatrix}\tag{1}
$$

- Latent varibales $\mathbf{z}_i$, for $i\in\{1,\cdots,N\}$. And similar to matrix (3), the matrix of latent varibales is
$$
Z = \begin{bmatrix}
  -&\mathbf{z}_1^T&-\\
  -&\mathbf{z}_2^T&-\\
  &\vdots&\\
  -&\mathbf{z}_N^T&-\\
\end{bmatrix}\tag{2}
$$


Once we got these two matrices, based on the definition of ['Mixtures of Gaussians'](https://anthony-tan.com/Mixtures-of-Gaussians/):

$$
\Pr(\mathbf{x})= \sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)\tag{3}
$$

the log-likelihood function is given by:

$$
\begin{aligned}
\ln \Pr(\mathbf{x}|\mathbf{\pi},\mathbf{\mu},\Sigma)&=\ln (\Pi_{n=1}^N\sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k))\\
&=\sum_{n=1}^{N}\ln \sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}_n|\mathbf{\mu}_k,\Sigma_k)\\
\end{aligned}\tag{4}
$$

This looks different from the single Gaussian model where the logarithm operates directly on $\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)$ who is an exponential function. The existence of summation in the logarithm makes the problem hard to solve.

Because the combination order of the Gaussian mixture could be arbitrary, we could have $K!$ equivalent solutions of $\mathbf{z}$. So which one we get did not affect our model.

## Maximum Likelihood Could Fail


### Covariance Matrix $\Sigma$ is not Singular
In the Gaussian distribution, the covariance matrix must be able to be inverted. So in our following discussion, we assume all the covariance matrice are invisible. For simplicity we take $\Sigma_k=\delta_k^2 I$ where $I$ is identity matrix.

### When $\mathbf{x}_n=\mathbf{\mu}_j$

When a point in the sample accidentally equals the mean $\mu_j$, the Gaussian distribution of the random variable $\mathbf{x}_n$ is:

$$
\mathcal{N}(\mathbf{x}_n|\mathbf{\mu}_k,\delta_j^2I)=\frac{1}{2\pi^{\frac{1}{2}}}\frac{1}{\delta_j}\tag{5}
$$

When the variance $\delta_j\to 0$, this part goes to infinity and the whole algorithm failed.

This problem does not exist in a single Gaussian model, for the $\mathbf{x}_n-\mathbf{\mu}_j=0$ is not an exponent in its log-likelihood.



## Summary
The maximum likelihood method is not suited for a Gaussian mixture model. Then we introduce the EM algorithm in the next post.

## References
[^1]: Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.