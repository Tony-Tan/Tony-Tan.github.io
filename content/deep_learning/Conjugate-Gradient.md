---
title: 'Conjugate Gradient'
tags:
 - Artificial Neural Networks
 - Artificial Intelligence
 - Conjugate Gradient
 - steepest descent method
keywords:
 - Artificial Neural Networks
 - Artificial Intelligence
 - Conjugate Gradient
 - steepest descent method
categories:
 - Artificial Neural Networks
date: 2019-12-21 13:40:24
markup: pdc
draft: false
images: ""
url: "/Conjugate-Gradient"
---

## Preliminaries
1. ['steepest descent method'](https://anthony-tan.com/Steepest-Descent-Method/)
2. ["Newton's method"](https://anthony-tan.com/Newton_s-Method/)

## Conjugate Gradient[^1]

We have learned ['steepest descent method'](https://anthony-tan.com/Steepest-Descent-Method/) and ["Newton's method"](https://anthony-tan.com/Newton_s-Method/). The main advantage of Newton's method is the speed, it converges quickly. And the main advantage of the steepest descent method guarantees to converge to a local minimum. But the limit of Newton's method is that it needs too many resources for both computation and storage when the number of parameters is large. And the speed of the steepest descent method is too slow. What we are going to research is to mix them up into a new algorithm that needs fewer resources but convergents quickly. In other words, we use first-order derivatives but still have quadratic efficiency.

In the last section of ['steepest descent method'](https://anthony-tan.com/Steepest-Descent-Method/), we found that when we minimize along the line at every iteration of the steepest descent method the directions of descent steps are orthogonal for a quadratic function.  

Our target here was simplified to a quadratic function to have an insight into the process of the new method:

$$
F(\mathbf{x})=\frac{1}{2}\mathbf{x}^TA\mathbf{x}+\mathbf{d}\mathbf{x}+c\tag{1}
$$

Before we go to the method, we need to recall the linear algebra concept "conjugate" firstly:

> A set of vectors $\{\mathbf{p}_k\}$ for $k=1,2,\cdots,n$ are mutually conjugate with respect to a positive definite matrix $A$ if and only if:
> $$\mathbf{p}^T_kA\mathbf{p}_j=0\tag{2}$$
> where $j\neq k$

the vectors in a conjugate, set are linear independent, so if there are $n$ vectors in a conjugate set, they can span a $n$-dimansion space. And to a certain space, there are infinite numbers of conjugate sets. For instance, eigenvectors of $A$ in equation(1) are $\{\mathbf{z}_1,\cdots,\mathbf{z}_n\}$ and eigenvalues are $\{\lambda_1,\cdots,\lambda_n\}$ and we have:

$$
\mathbf{z}_k^TA\mathbf{z}_j=\lambda_j\mathbf{z}_k^T\mathbf{z}_j\tag{3}
$$

when the matrix $A$ is positive definite, eigenvectors are mutually orthogonal. And then equation(3) equal to $0$ when $k\neq j$

['Quadratic function'](https://anthony-tan.com/Quadratic-Functions/) told us the eigenvectors are principal axes, and searching along these eigenvectors can finally minimize the quadratic function.

The calculation of a Hessian matrix should be avoided in our new method because of the computational resources. So searching along the directions of eigenvectors would not be used. 

Then an idea comes to us whether other conjugate sets could give a trajectory to the minimum. And mathematicians did prove this guess is correct. Searching along a conjugate set, $\{\mathbf{p_1},\dots,\mathbf{p}_n\}$, of a  matrix $A$ can converge to the minimum in at most $n$ searches.


Based on this theory, what we should do next is find a conjugate set of a matrix without the use of the Hessian matrix. We restate that:

$$
\nabla F(\mathbf{x})=\mathbf{g}=A\mathbf{x}+\mathbf{d}\tag{4}
$$

and 

$$
\nabla^2 F(\mathbf{x})=A\tag{5}
$$

in the steepest descent method, we search along the negative direction of gradient $\mathbf{g}$. 

Then the change between the consecutive steps is:
$$
\Delta \mathbf{g}_k = \mathbf{g}_{k+1}-\mathbf{g}_k\tag{6}
$$

Insert equation(4) into equation(6):

$$
\begin{aligned}
   \Delta \mathbf{g}_k &= \mathbf{g}_{k+1}-\mathbf{g}_k\\
   &=A\mathbf{x}_{k+1}+\mathbf{d} -A\mathbf{x}_{k}+\mathbf{d}\\
   &=A\Delta\mathbf{x} 
\end{aligned}
\tag{7}
$$

equation(7) is the key point in this new algorithm because this equation replace $A\Delta\mathbf{x}$ with $\Delta \mathbf{g}_k$ to avoid the calculation about matrix $A$. And for in the post[]() we update the variables in each step by $\mathbf{x}_{k+1} =\mathbf{x}_k +\alpha\mathbf{p}_k$ then we have:

$$
\Delta\mathbf{x} =\mathbf{x}_{k+1} -\mathbf{x}_k =\alpha\mathbf{p}_k\tag{8}
$$

the assumption in equation(2) makes sure that:

$$
\alpha\mathbf{p}^T_kA\mathbf{p}_j=0\tag{9}
$$

and take equation(8) into equation(9):

$$
\alpha\mathbf{p}^T_kA\mathbf{p}_j=\Delta\mathbf{x}^TA\mathbf{p}_j=0\tag{10}
$$

Then the key point equation(7) would be used to replace the $\Delta\mathbf{x}^TA$ in equation(10):

$$
\Delta\mathbf{x}^TA\mathbf{p}_j=\mathbf{g}^T\mathbf{p}_j=0\tag{11}
$$

where $j\neq k$

Till now, we have all of what we need to build a method acting like a second-order method with just the amount of calculation of the first-order method. And the procedure of method is:

1. initialize $\mathbf{p}_0=-\mathbf{g}_0$
2. construct a vector, the next direction, $\mathbf{p}_k$ orthogonal to $\{\Delta\mathbf{g}_0,\Delta\mathbf{g}_1,\cdots,\Delta\mathbf{g}_{k-1}\}$(somehow it looks like Gram-Schimidt method)
3. $\mathbf{p}_k=-\mathbf{g}_k+\beta_k\mathbf{p}_{k-1}$
4. $\beta_k$ is a scalar and can be calculated in three kinds of ways:
   1. $\beta_k=\frac{\Delta\mathbf{g}_{k-1}^T\mathbf{g}_k}{\Delta\mathbf{g}_{k-1}^T\mathbf{p}_{k-1}}$
   2. $\beta_k=\frac{\mathbf{g}_{k}^T\mathbf{g}_k}{\mathbf{g}_{k-1}^T\mathbf{p}_{k-1}}$
   3. $\beta_k=\frac{\Delta\mathbf{g}_{k-1}^T\mathbf{g}_k}{\mathbf{g}_{k-1}^T\mathbf{g}_{k-1}}$




## References
[^1]: Demuth, H.B., Beale, M.H., De Jess, O. and Hagan, M.T., 2014. Neural network design. Martin Hagan.