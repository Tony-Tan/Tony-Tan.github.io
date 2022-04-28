---
title: 'Committees'
tags:
  - Machine Learning
  - Mixture Models
  - committees
keywords:
  - Machine Learning
  - Mixture Models
  - committees
date: 2020-03-07 13:55:21
markup: pdc
draft: false
images: ""
url: "/Committees"
---

## Preliminaries
1. Basic machine learning concepts
2. Probability Theory concepts
  - expectation
  - correlated random variable 


## Analysis of Committees[^1]

The committee is a native inspiration for how to combine several models(or we can say how to combine the outputs of several models). For example, we can combine all the models by:

$$
y_{COM}(X)=\frac{1}{M}\sum_{m=1}^My_m(X)\tag{1}
$$

Then we want to find out whether this average prediction of models is better than every one of them.

To compare the committee and a single model, we need first to build a criterion depending on which we can distinguish which result is better. Assuming that the true generator of the training data $x$ is:

$$
h(x)\tag{2}
$$

So our prediction of the $m$th model for $m=1,2\cdots,M$ can be represented as:

$$
y_m(x) = h(x) +\epsilon_m(x)\tag{3}
$$
where $\epsilon_m(x)$ is the error of $m$ th model.Then the average sum-of-squares of error can be a nice criterion. 

The criterion of a single model is:

$$
\mathbb{E}_x[(y_m(x)-h(x))^2] = \mathbb{E}_x[\epsilon_m(x)^2] \tag{4}
$$

where the $\mathbb{E}[\cdot]$ is the frequentist expectation(or average for usual saying). 

Now we consider the average of the error over $M$ models:

$$
E_{AV} = \frac{1}{M}\sum_{m=1}^M\mathbb{E}_x[\epsilon_m(x)^2]\tag{5}
$$

And on the other hand, the committees have the error given by equations (1), (3), and (4):

$$
\begin{aligned}
  E_{COM}&=\mathbb{E}_x[(\frac{1}{M}\sum_{m=1}^My_m(x)-h(x))^2] \\
  &=\mathbb{E}_x[\{\frac{1}{M}\sum_{m=1}^M\epsilon_m(x)\}^2]
\end{aligned} \tag{6}
$$

Now we assume that the random variables $\epsilon_i(x)$ for $i=1,2,\cdots,M$ have **mean 0** and **uncorrelated**, so that:

$$
\begin{aligned}
  \mathbb{E}_x[\epsilon_m(x)]&=0 &\\
  \mathbb{E}_x[\epsilon_m(x)\epsilon_l(x)]&=0,&m\neq l
\end{aligned} \tag{7}
$$

Then substitute equation (7) into equation (6), we can get:

$$
E_{COM}=\frac{1}{M^2}\mathbb{E}_x[\epsilon_m(x)]\tag{8}
$$

According to the equation (5) and (8):

$$
E_{AV}=\frac{1}{M}E_{COM}\tag{9}
$$

**All the mathematics above is based on the assumption that the error of each model is uncorrelated**. However, most time they are highly correlated and the reduction of error is generally small. But the relation:

$$
E_{COM}\leq E_{AV}\tag{10}
$$

exists definitely which means committees can produce better predictions than a single model.

## References
[^1]: Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.