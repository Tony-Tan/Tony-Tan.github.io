---
title: 'From Linear Regression to Linear Classification'
tags:
  - Machine Learning
  - Linear Classification
  - Algorithm
  - decision regions
  - binary code scheme
  - decision boundaries
  - decision surfaces
  - generative model
  - discriminate model
  - active function
keywords:
  - machine learning
  - linear classification
  - decision regions
  - binary code scheme
  - decision boundaries
  - decision surfaces
  - generative model
  - discriminate model
  - active function
date: 2020-02-17 11:20:11
markup: pdc
draft: false
images: ""
url: "/From-Linear-Regression-to-Linear-Classification"
---



## Preliminaries
1. [An Introduction to Linear Regression](https://anthony-tan.com/An-Introduction-to-Linear-Regression/)
2. [A Simple Linear Regression](https://anthony-tan.com/A-Simple-Linear-Regression/)
3. Bayesian theorem
4. Feature extraction


## Recall Linear Regression

The goal of a regression problem is to find out a function or hypothesis that given an input $\mathbf{x}$, it can make a prediction $\hat{y}$ to estimate the target. Both the target $y$ and prediction $\hat{y}$ here are continuous. They have the properties of numbers[^1]:

> Consider 3 inputs $\mathbf{x}_1$, $\mathbf{x}_2$ and $\mathbf{x}_3$ and their coresponding targets are $y_1=0$, $y_2=1$ and $y_3=2$. Then a good predictor should give the predictions $\hat{y}_1$, $\hat{y}_2$ and $\hat{y}_3$ where the distance between  $\hat{y}_1$ and $\hat{y}_2$ is larger than the one between  $\hat{y}_1$ and $\hat{y}_3$ 

Some properties of regression tasks we should pay attention to are:

1. The goal of regression is to produce a hypothesis that can give a prediction as close to the target as possible
2. The output of the hypothesis and target are continuous numbers and have numerical meanings, like distance, velocity, weights, and so on.

## General Classification

On the other side, we met more classification tasks in our life than regression. Such as in the supermarket we can tell the apple and the orange apart easily. And we can even verify whether this apple is tasty or not.

Then the goal of classification is clear:

> Assign input $\mathbf{x}$ to a certain class of $K$ available classes. And $\mathbf{x}$ must belong to one and only one class.

The input $\mathbf{x}$, like the input of regression, can be a feature or basis function and can be continuous or discrete. However, its output is discrete. Let's go back to the example that we can tell apple, orange, and pineapple apart. The difference between apple and orange and the difference between apple and pineapple can not be compared, because the distance(it is the mathematical name of difference) itself had no means.

### A Binary Code Scheme
we can not calculate apple and orange directly. So a usual first step in the classification task is mapping the target or labels of an example into a number, like $1$ for the apple and $2$ for the orange.

A **binary code scheme** is another way to code targets. 

For a two classes mission, the numerical labels can be:

$$
\mathcal{C}_1=0 \text{ and }\mathcal{C}_2=1\tag{1}
$$

It's equal to:

$$
\mathcal{C}_1=1 \text{ and }\mathcal{C}_2=0\tag{2}
$$

And to a $K$ classes target, the binary code scheme is:

$$
\begin{aligned}
  \mathcal{C}_1 &= \{1,0,\cdots,0\}\\
  \mathcal{C}_2 &= \{0,1,\cdots,0\}\\
  \vdots & \\
  \mathcal{C}_K &= \{0,0,\cdots,1\}\\
\end{aligned}\tag{3}
$$

The $n$-dimensional input $\mathbf{x}\in \mathbb{R}^n$ and $\mathbb{R}^n$ is called the input space. In the classification task, the input points can be separated by the targets, and these parts of space are called **decision regions** and the boundaries between decision regions are called **decision boundaries** or **decision surfaces**. When the decision boundary is linear, the task is called linear classification.

There are roughly two kinds of procedures for classification:

1. Discriminant Function: assign input $\mathbf{x}$ to a certain class directly.
2. We infer $\Pr(\mathcal{C}_k|\mathbf{x})$ firstly and then make a decision based on the posterior probability.
   1. Inference of $\Pr(\mathcal{C}_k|\mathbf{x})$ was calculated firstly
   2. $\Pr(\mathcal{C}_k|\mathbf{x})$ can also be calculated by Bayesian Theorem $\Pr(\mathcal{C}_k|\mathbf{x})=\frac{\Pr(\mathbf{x}|\mathcal{C}_k)\Pr(\mathcal{C}_k)}{\Pr(\mathbf{x})}=\frac{\Pr(\mathbf{x}|\mathcal{C}_k)\Pr(\mathcal{C}_k)}{\sum_k \Pr(\mathbf{x}|\mathcal{C}_k)\Pr(\mathcal{C}_k)}$

They are the discriminate model and generative model, respectively.

## Linear Classification

In the regression problem, the output of the linear function:

$$
\mathbf{w}^T\mathbf{x}+b\tag{4}
$$

is approximate of the target. But in the classification task, we want the output to be the class to which the input $\mathbf{x}$ belongs. However, the output of the linear function is always continuous. This output is more like the posterior probability, say $\Pr({\mathcal{C}_i|\mathbf{x}})$ rather than the discrete class label. To generate a class label output, function $f(\cdot)$ which is called 'action function' in machine learning was employed. For example, we can choose a threshold function as the active function:

$$
y(\mathbf{x})=f(\mathbf{w}^T\mathbf{x}+b)\tag{5}
$$

where $f(\cdot)$ is the threshold function:

$$
f(x) = \begin{cases}1&x\geq c\\0&\text{otherwise}\end{cases}\tag{6}
$$
where $c$ is a constant.

In this case, the boundary is $\mathbf{w}^T\mathbf{x}+b = c$, and it is a line. So we call this kind of model 'linear classification'. The input $\mathbf{x}$ can be replaced by a basis function $\phi(\mathbf{x})$ as mentioned in [the polynomial regression](https://anthony-tan.com/Polynomial-Regression-and-Features-Extension-of-Linear-Regression/).


## References
[^1]: Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.