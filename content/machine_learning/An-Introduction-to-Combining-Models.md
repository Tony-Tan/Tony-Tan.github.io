---
title: 'An Introduction to Combining Models'
tags:
  - Machine Learning
  - Combining Models
  - AdaBoost
  - boosting
  - bagging
  - EM algorithm
keywords:
  - Machine Learning
  - Combining Models
  - AdaBoost
  - boosting
  - bagging
  - EM algorithm
date: 2020-03-07 12:04:00
markup: pdc
draft: false
images: ""
url: "/An-Introduction-to-Combining-Models"
---
## Preliminaries
1. ['Mixtures of Gaussians'](https://anthony-tan.com/Mixtures-of-Gaussians/)
2. Basic machine learning concepts

## Combining Models[^1]

The mixture of Gaussians had been discussed in the post ['Mixtures of Gaussians'](https://anthony-tan.com/Mixtures-of-Gaussians/). It was used to introduce the 'EM algorithm' but it gave us the inspiration of improving model performance. 


All models we have studied, besides neural networks, are all single-distribution models. That is just like that, to solve a problem we invite an expert who is very good at this kind of problem, then we just do whatever the expert said. However, if our problem is too hard that no expert can solve it completely by himself, inviting more experts is a good choice. This inspiration gives a new way to improve performance by combining multiple models but not just by improving the performance of a single model.

## Organising Models

A naive idea is voting by several models equally, which means averaging the prediction of all models. However, different models may have different abilities, and voting equally is not a good idea. Then boosting and other methods were introduced.

In some combining methods, such as AdaBoost(boosting), bootstrap, bagging, and e.t.c, the input data has an identical distribution with the training set. However, in some methods, the training set is cut into several subsets with different distributions from the original training set. The decision tree is such a method. A decision tree is a sequence of binary selection and it worked well in both regression and classification tasks. 

We will briefly discuss:
- [committees](https://anthony-tan.com/Committees/)
- [boosting](https://anthony-tan.com/Boosting-and-AdaBoost/)
- [decision tree]()

in the following posts.

## References
[^1]: Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.