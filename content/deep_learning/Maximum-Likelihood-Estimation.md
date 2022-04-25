---
title: 'Maximum Likelihood Estimation'
tags:
  - Machine Learning
  - linear regression
  - maximum likelihood estimation
  - regression function
  - least square estimation
keywords:
  - Machine Learning
  - linear regression
  - maximum likelihood estimation
  - regression function
  - least square estimation
date: 2020-02-15 00:41:25
markup: pdc
draft: false
mages: ""
url: "/Maximum-Likelihood-Estimation"
---
## Priliminaries
1. [A Simple Linear Regression](https://anthony-tan.com/A-Simple-Linear-Regression/)
2. [Least Squares Estimation](https://anthony-tan.com/Least-Squares-Estimation/)
3. linear algebra

## Square Loss Function for Regression[^1]

For any input $\mathbf{x}$, our goal in a regression task is to give a prediction $\hat{y}=f(\mathbf{x})$ to approximate target $t$ where the function $f(\cdot)$ is the chosen hypothesis or model as mentioned in the post [https://anthony-tan.com/A-Simple-Linear-Regression/](https://anthony-tan.com/A-Simple-Linear-Regression/). 

The difference between $t$ and $\hat{y}$ can be called 'error' or more precisely 'loss'. Because in an approximation task, 'error' occurs by chance and always exists, and 'loss' is a good word to represent the difference. The loss can be written generally as function $\ell(f(\mathbf{x}),t)$. Intuitively, the smaller the loss, the better the approximation. 

So the expectation of loss:

$$
\mathbb E[\ell]=\int\int \ell(f(\mathbf{x}),t)p(\mathbf{x},t)d \mathbf{x}dt\tag{1}
$$

should be as small as possible.

In probability viewpoint, the input vector $\mathbf{x}$, target $t$ and parameters in function(model) $f(\cdot)$ are all random variables. Then the expectation of loss function may exist.

Considering the square error loss function $e=(f(\mathbf{x})-t)^2$, it is a usual measure of the difference between the prediction and the target. And substitute the loss function into equation (1), we have:

$$
\mathbb E[\ell]=\int\int (f(\mathbf{x})-t)^2p(\mathbf{x},t)d \mathbf{x}dt\tag{2}
$$

To minimize this function, we could use [Euler-Lagrange equation](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation), [Fundamental theorem of calculus](https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus) and [Fubini's theorem](https://en.wikipedia.org/wiki/Fubini%27s_theorem):


Fubini's theorem told us that we can change the order of integration:
$$
\begin{aligned}
\mathbb E[\ell]&=\int\int (f(\mathbf{x})-t)^2p(\mathbf{x},t)d \mathbf{x}dt\\
&=\int\int (f(\mathbf{x})-t)^2p(\mathbf{x},t)dtd \mathbf{x}
\end{aligned}\tag{3}
$$


According to the Euler-Lagrange equation, we first create a new function $G(x,f,f')$:
$$
G(x,f,f')= \int (f(\mathbf{x})-t)^2p(\mathbf{x},t)dt\tag{4}
$$


The Euler-Lagrange equation is used to minimize the equation (2):
$$
\frac{\partial G}{\partial f}-\frac{d}{dx}\frac{\partial G}{\partial f'}=0\tag{5}
$$


Because there is no $y'$ component in function $G()$. Then the equation:
$$
\frac{\partial G}{\partial f}=0\tag{6}
$$ 
becomes the necessary condition to minimize the equation (2):

$$
2\int (f(\mathbf{x})-t)p(\mathbf{x},t)dt=0 \tag{7}
$$

Rearrange the equation (7), and we get a good predictor that can minimize the square loss function :

$$
\begin{aligned}
  \int (f(\mathbf{x})-t)p(\mathbf{x},t)dt&=0\\
  \int f(\mathbf{x})p(\mathbf{x},t)dt-\int tp(\mathbf{x},t)dt&=0\\
  f(\mathbf{x})\int p(\mathbf{x},t)dt&=\int tp(\mathbf{x},t)dt\\
  f(\mathbf{x})&=\frac{\int tp(\mathbf{x},t)dt}{\int p(\mathbf{x},t)dt}\\
  f(\mathbf{x})&=\frac{\int tp(\mathbf{x},t)dt}{p(\mathbf{x})}\\
  f(\mathbf{x})&=\int tp(t|\mathbf{x})dt\\
  f(\mathbf{x})&= \mathbb{E}_t[t|\mathbf{x}]
\end{aligned}\tag{8}
$$

We finally find the expectation of $t$ given $\mathbf{x}$ is the optimum solution. The expectation of $t$ given $\mathbf{x}$ is also called the **regression function**.

A small summary: $\mathbb{E}[t| \mathbf{x}]$ is a good estimate of $f(\mathbf{x})$


## Maximum Likelihood Estimation

Generally, we assume that there is a generator behind the data:

$$
t=g(\mathbf{x},\mathbf{w})+\epsilon\tag{9}
$$

where the function $g(\mathbf{x},\mathbf{w})$ is a deterministic function, $t$ is the target variable and $\epsilon$ is zero mean Gaussian random variable with percision $\beta$ which is the inverse variance. Because of the property of Gaussian distribution, $t$ has a Gaussian distribution, with mean(expectation) $g(\mathbf{x},\mathbf{w})$ and percesion $\beta$. And recalling the standard form of Gaussian distribution:

$$
\begin{aligned}
\Pr(t|\mathbf{x},\mathbf{w},\beta)&=\mathcal{N}(t|g(\mathbf{x},\mathbf{w}),\beta^{-1})\\
&=\frac{\beta}{\sqrt{2\pi}}\mathrm{e}^{-\frac{1}{2}(\beta(x-\mu)^2)}
\end{aligned}\tag{10}
$$

Our task here is to approximate the generator in equation (9) with a linear function. Somehow, when we use the square loss function, the optimum solution for this task is $\mathbb{E}[t|\mathbf{x}]$ to equation (8). 

the solution to equation (10) is: 

$$
\mathbb{E}[t|\mathbf{x}]=g(\mathbf{x},\mathbf{w})\tag{11}
$$

We set the linear model as:
$$
f(x)=\mathbf{w}^T\mathbf{x}+b\tag{12}
$$

and this can be converted to:

$$
f(x)=
\begin{bmatrix}
b&\mathbf{w}^T
\end{bmatrix}
\begin{bmatrix}
1\\
\mathbf{x}
\end{bmatrix}=\mathbf{w}_a^T\mathbf{x}_a
\tag{13}
$$

for short, we just write the $\mathbf{w}_a$ and $\mathbf{x}_a$ as $\mathbf{w}$ and $\mathbf{x}$. Then the linear model becomes:

$$
f(x)=\mathbf{w}^T\mathbf{x}\tag{14}
$$

As we mentioned above we consider all the parameter as a random variable, then the conditioned distribution of $\mathbf{w}$ is $\Pr(\mathbf{w}|\mathbf{t},\beta)$. $X$ or $\mathbf{x}$ was omitted in the condition because it does not affect the result at all. And the Bayesian theorem told us:

$$
\Pr(\mathbf{w}|\mathbf{t},\beta)=\frac{\Pr(
  \mathbf{t}|\mathbf{w},\beta)
  \Pr(\mathbf{w})}
  {\Pr(\mathbf{t})}=\frac{\text{Likelihood}\times \text{Prior}}{\text{Evidence}}\tag{15}
$$

We want to find the $\mathbf{w}^{\star}$ that maximise the posterior probability $\Pr(\mathbf{w}|\mathbf{t},\beta)$. Because $\Pr(\mathbf{t})$ and  $\Pr(\mathbf{w})$ are constant. Then the maximum of likelihood $\Pr(\mathbf{t}|\mathbf{w},\beta)$ maximise the posterior probability. 

$$
\begin{aligned}
\Pr(\mathbf{t}|\mathbf{w},\beta)&=\Pi_{i=0}^{N}\mathcal{N}(t_i|\mathbf{w}^T\mathbf{x}_i,\beta^{-1})\\
\ln \Pr(\mathbf{t}|\mathbf{w},\beta)&=\sum_{i=0}^{N}\ln \mathcal{N}(t_i|\mathbf{w}^T\mathbf{x}_i,\beta^{-1})\\
&=\sum_{i=0}^{N}\ln \frac{\beta}{\sqrt{2\pi}}\mathrm{e}^{-\frac{1}{2}(\beta(t_i-\mathbf{w}^T\mathbf{x}_i)^2)}\\
&=\sum_{i=0}^{N} \ln \beta - \sum_{i=0}^{N} \ln \sqrt{2\pi} - \frac{1}{2}\beta\sum_{i=0}^{N}(t_i-\mathbf{w}^T\mathbf{x}_i)^2
\end{aligned}\tag{16}
$$

This gives us a wonderful result. 

We can only control the component $\frac{1}{2}\beta\sum_{i=0}^{N}(t_i-\mathbf{w}^T\mathbf{x}_i)^2$ of the last line of equation(16), because $\sum_{i=0}^{N} \ln \beta$ and $- \sum_{i=0}^{N} \ln \sqrt{2\pi}$ were decided by the assumptions. In other words, to maximise the likelihood, we just need to minimise:

$$
\sum_{i=0}^{N}(t_i-\mathbf{w}^T\mathbf{x}_i)^2\tag{17}
$$

This was just to minimize the sum of squares. Then this optimization problem went back to the [least square problem](https://anthony-tan.com/Least-Squares-Estimation/).


## Least Square Estimation and Maximum Likelihood Estimation

When we assume there is a generator:

$$
t=g(\mathbf{x},\mathbf{w})+\epsilon\tag{18}
$$

behind the data, and $\epsilon$ has a zero-mean Gaussian distribution with any precision $\beta$, the maximum likelihood estimation finally converts to the least square estimation. This is not only worked for linear regression because we did not assume what $g(\mathbf{x},\mathbf{w})$ is.

However, when the $\epsilon$ has a different distribution but not Gaussian distribution, the least square estimation will not be the optimum solution for maximum likelihood estimation.

## References
[^1]: Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.