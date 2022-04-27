---
title: 'Mixtures of Gaussians'
tags:
  - Machine Learning
  - Mixture Models
  - latent varible
  - reponsibility
  - expectation-maximization
  - EM
  - responsibility
keywords:
  - Machine Learning
  - Mixture Models
  - latent varible
  - reponsibility
  - expectation-maximization
  - EM
  - responsibility
date: 2020-03-05 16:05:50
markup: pdc
draft: false
images: ""
url: "/Mixtures-of-Gaussians"
---

## Preliminaries
1. Probability Theory
  - multiplication principle
  - joint distribution
  - the Bayesian theory
  - Gaussian distribution
2. Calculus 1,2


## A Formal Introduction to Mixtures of Gaussians[^1]

We have introduced a mixture distribution in the post ['An Introduction to Mixture Models'](). And the example in that post was just two components Gaussian Mixture. However, in this post, we would like to talk about Gaussian mixtures formally. And it severs to motivate the development of the expectation-maximization(EM) algorithm.

Gaussian mixture distribution can be written as:

$$
\Pr(\mathbf{x})= \sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)\tag{1}
$$

where $\sum_{k=1}^K \pi_k =1$ and $0\leq \pi_k\leq 1$.

And then we introduce a random variable(vector) called latent variable(vector) $\mathbf{z}$, that each component of $\mathbf{z}$:

$$
z_k\in\{0,1\}\tag{2}
$$

and $\mathbf{z}$ is a $1$-of-$K$ representation, which means there is one and only one component is $1$ and others are $0$. 

To build a joint distribution $\Pr(\mathbf{x},\mathbf{z})$, we should build $\Pr(\mathbf{x}|\mathbf{z})$ and $\Pr(\mathbf{z})$ firstly. We define the distribution of $\mathbf{z}$:

$$
\Pr(z_k=1)=\pi_k\tag{3}
$$

for $\{\pi_k\}$ for $k=1,\cdots,K$. And equation (3) is now written as:

$$
\Pr(\mathbf{z}) = \Pi_{k=1}^K \pi_k^{z_k}\tag{4}
$$

And according to the definition of $\Pr(\mathbf{z})$ we can get the condition distribution of $\mathbf{x}$ given $\mathbf{z}$. Under the condition $z_k=1$, we have:

$$
\Pr(\mathbf{x}|z_k=1)=\mathcal{N}(\mathbf{x}|\mu_k,\Sigma_k)\tag{5}
$$

and then we can derive the vector form of conditional distribution:

$$
\Pr(\mathbf{x}|\mathbf{z})=\Pi_{k=1}^{K}\mathcal{N}(\mathbf{x}|\mathbf{\mu}_k,\Sigma_k)^{z_k}\tag{6}
$$

Once we have both the probability distribution of $\mathbf{z}$, $\Pr(\mathbf{z})$, and conditional distribution, $\Pr(\mathbf{x}|\mathbf{z})$, we can build joint distribution by multiplication principle:

$$
\Pr(x,z) = \Pr(\mathbf{z})\cdot \Pr(\mathbf{x}|\mathbf{z})\tag{7}
$$

However, what we are concerning is still the distribution of $\mathbf{x}$. We can calculate the probability of $\mathbf{x}$ by:

$$
\Pr(\mathbf{x}) = \sum_{j}\Pr(x,\mathbf{z}_j) = \sum_{j}\Pr(\mathbf{z}_j)\cdot \Pr(\mathbf{x}|\mathbf{z_j})\tag{8}
$$

where $\mathbf{z_j}$ is every possible value of random variable $\mathbf{z}$.

This is how latent variables construct mixture Gaussians. And this form is easy for us to analyze the distribution of a mixture model.

## 'Responsibility' of Gaussian Mixtures

The Bayesian formula can help us produce posterior. And the posterior probability of latent variable $\mathbf{z}$ by equation (7) can be calculated:

$$
\Pr(z_k=1|\mathbf{x})=\frac{\Pr(z_k=1)\Pr(\mathbf{x}|z_k=1)}{\sum_j^K \Pr(z_j=1)\Pr(\mathbf{x}|z_j=1)}\tag{9}
$$

and substitute equation (3),(5) into equation (9) and we get:

$$
\Pr(z_k=1|\mathbf{x})=\frac{\pi_k\mathcal{N}(\mathbf{x}|\mu_k,\Sigma_k)}{\sum^K_j \pi_j\mathcal{N}(\mathbf{x}|\mu_j,\Sigma_j)}\tag{10}
$$

And $\Pr(z_k=1|\mathbf{x})$ is also called reponsibility, and denoted as:

$$
\gamma(z_k)=\Pr(z_k=1|\mathbf{x})\tag{11}
$$


## References

[^1]: Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.