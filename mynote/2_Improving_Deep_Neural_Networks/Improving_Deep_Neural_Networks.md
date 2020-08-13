# Improving Deep Neural Networks

## 1 Setting Up your Machine Learning application

### 1.1 train / dev/ test sets

<img src="Improving_Deep_Neural_Networks.assets/image-20200809142458616.png" alt="image-20200809142458616" style="zoom: 33%;" />

- Previously: **70%** train + **30%** test // **60%** train + **20%** dev + **20%** test
- Now (big data): 1,000,000 = 980,000(**98%**) train + 10,000(**1%**) dev + 10,000(**1%**) test
for data sets with over 1 million examples, the training set can take up 99.5%

A test set is not always necessary (depend on whether you need an unbiased estimate)

#### Mismatched train/test distribution

for example:
- training set: cat pictures from webpages
- dev/test sets: cat pictures from users using the app

One Rule: make sure that ==dev and test sets come from the same distribution==

### 1.2 Bias & Variance

<img src="Improving_Deep_Neural_Networks.assets/image-20200809144638951.png" alt="image-20200809144638951" style="zoom: 33%;" />

| *               | High Variance | High Bias | High Bias & High Variance | Low Bias & Low Variance |
| --------------- | ------------- | --------- | ------------------------- | ----------------------- |
| Train set error | 1%            | 15%       | 15%                       | 0.5%                    |
| Dev set error   | 11%           | 16%       | 30%                       | 1%                      |
\* based on the assumption that Bayesion Optimal error is close to 0%

### 1.3 Basic "Recipe" for ML

<img src="Improving_Deep_Neural_Networks.assets/image-20200809231254321.png" alt="image-20200809231254321" style="zoom: 33%;" />

Bias-Variance trade-off

### 1.4 Regularization

#### in logistic regression:

$$
J(w,b)=\frac{1}{m}\sum\limits_{i=1}^m\mathcal{L}(\hat{y}^{(i)},y^{(i)})+\boxed{\frac{ \lambda }{2m}||w||^2_2} \quad (+ \frac{\lambda}{2m}b^2)\\
L_2 \ regularization: \ ||w||^2_2=\sum\limits_{j=1}^{n_x}w_j^2=w^Tw\\
\text{square Euclidean norm of the parameter vector w}\\
(L_1 \ regularization: \ \frac{\lambda}{2m}|w|_1=\frac{\lambda}{2m}\sum\limits_{j=1}^{n_x}|w_j|)
$$

- omitting the regularization of **b** makes little difference (w is a high dimensional parameter vector, while b is just a single number)
- if we use L~1~ regularization, then **w** will end up being sparse (have a lot of zeros)

**notice**: "lambda" is a reserved keyword in Python, use "lambd" to represent the lambda regularization parameter

#### in neural network:

$$
J(w^{[1]},b^{[1]}, ..., w^{[l]}, b^{[l]})=\frac{1}{m}\sum\limits_{i=1}^m\mathcal{L}(\hat{y}^{(i)},y^{(i)})+\boxed{\frac{ \lambda }{2m} \sum\limits_{l=1}^l ||w^{[l]}||^2_2} \\
||w^{[l]}||^2_F=\sum\limits_{i=1}^{n^{[l-1]}} \sum\limits_{j=1}^{n^{[l]}}(w_{ij}^{[l]})^2 \quad \quad w:(n^{[l-1]},n^{[l]})\\
\text{Frobenius norm of the parameter vector w}\\
dw^{[l]}=\text{(from backprop)}+\boxed{\frac{\lambda}{m}w^{[l]}}\\
\begin{align}
\text{"weight decay"} \quad w^{[l]}&:=w^{[l]}-\alpha dw^{[l]}\\
&:= w^{[l]}-\alpha [\text{(from backprop)}+\frac{\lambda}{m}w^{[l]}]\\
&:= (1-\frac{\alpha \lambda}{m})w^{[l]}-\alpha \text{(from backprop)}
\end{align}
$$

#### why regularization reduces overfitting?

big λ causes W close to zero 
→ zero out or at least reduce the impact of many hidden units
→ a simple, but deep NN (close to logistic regression)

big λ and small W causes Z relatively small
→ the activation function (on a small range) will be relatively linear
→ the neural network will be calculating something close to a simple linear function
→ less likely to overfit

### 1.5 Dropout Regularization

<img src="Improving_Deep_Neural_Networks.assets/image-20200810002202744.png" alt="image-20200810002202744" style="zoom: 33%;" />

for **each** example:
1. go through each of the layers
2. set some probability of eliminating a node in neural network
3. eliminate the selected nodes (remove all the ingoing & outgoing things from the selected nodes)
4. end up with a much smaller, diminished network

#### implementing dropout (inverted dropout)

L=3, keep-prob=0.8
d^3^=np.random.rand(a^3^.shape[0], a^3^.shape[1]) < keep-prob
a^3^=np.multiply(a^3^, d^3^)
a^3^ /= keep-prob  #ensure the expected value of a^3^ remains the same

NO dropout at **test** time
  the output shouldn't be random, it will add noise to your prediction

#### intuitions about dropout

##### why it works
- using a smaller network is like using a regularized network

- can't rely on any one feature, so have to spread out weights (onto differet units)

	tend to have an effect of shrinking the squared norm of the weights

##### usage:

1. lower the value of keep-prob for layers that are more likely to overfit (drawback: need to search more hyper parameters for using cross-validation)
2. apply dropout to some layers and don't apply to others

frequently used in computer vision (lack of data causes overfitting)
only use it when the function is overfitting

**downside**: while using dropout, the cost function **J** is not well-defined, so it becomes harder to check wheter the cost funtion **J** is declining. 

### 1.6 Other Regularization Methods

#### Data Augmentation

when having trouble obtaining new training data, augment the current training set

images: flip, distort, rotate, clip...
characters: rotate, distort...

#### Early Stopping

<img src="Improving_Deep_Neural_Networks.assets/image-20200812221424464.png" alt="image-20200812221424464" style="zoom:33%;" />

plot [dev set error] and [training error / cost function J] in the same graph
stop training **halfway**

using 1 method to solve 2 problems:

1. optimize the cost function J  -- gradient descent, ...
2. not overfit -- regularization, ...

### 1.7 Normalizing Input

1. substract out or zero out the mean:

$$
\mu = \frac{1}{m}\sum\limits_{i=1}^m x^{(i)}\\
x=x-\mu
$$

2. normalize variance

$$
\sigma ^2 = \frac{1}{m}\sum\limits_{i=1}^{m}x^{(i)}**2\\
x /= \sigma ^2 \\
**: \text{element y squaring ???}
$$

<img src="Improving_Deep_Neural_Networks.assets/image-20200812231929491.png" alt="image-20200812231929491" style="zoom:33%;" />

- result: the variance of x1 and x2 are both equal to 1 

- tip: use the **same** μ and σ to normalize the training set and the test set 

#### why normalize?

<img src="Improving_Deep_Neural_Networks.assets/image-20200812232112596.png" alt="image-20200812232112596" style="zoom: 33%;" />

ensure all the features on a similar scale, more symmetric, can use larger steps ...
  → easier to optimize

### 1.8 Vanishing/Exploding Gradients

if the weight parameter **w** is more\less than 1,
then in a very **deep** network, the activations increasing/decreasing **exponentially**

#### partial solution

set a reasonable **variance** of the initialization of the weight matrices
- doesn't solve but help reduce the vanishing/exploding gradients problem

for ReLU:
$$
z=w_1x_1+w_2x_2+...+w_nx_n \\
\text{larger} \ n \rightarrow \text{smaller} \ w_1 \\
\begin{align}
&\text{in order to make} \quad Var(w_i)=\frac{2}{n} \\
&\text{let} \quad W^{[l]}=\text{np.random.randn(shape)}*\boxed{\text{np.sqrt}(\frac{2}{n^{[l-1]}})} \\
\end{align}
$$
for other activations:
  tanh: $\sqrt{\frac{1}{n^{[l-1]}}}$ (Xavier Initialization),  or  $\sqrt{\frac{2}{n^{[l-1]}+n^{[l]}}}$

- set the variance as a hyperparameter
- tuning this parameter is not so effective

### 1.9 Gradient Checking

#### Numerical Approximation of Gradients

$$
\begin{align}
\text{two-sided:} \quad f'(\theta)&=\lim\limits_{\epsilon \rightarrow0} \frac{f(\theta + \epsilon)-f(\theta - \epsilon)}{2\epsilon} \quad error:O(\epsilon^2)\\
\text{one-sided:} \quad f'(\theta)&=\lim\limits_{\epsilon \rightarrow0} \frac{f(\theta + \epsilon)-f(\theta)}{\epsilon} \quad error:O(\epsilon)\\
\end{align}
$$

use **two-sided** difference (more accurate) to check the derivatives(backprop)

#### Gradient Checking for a NN

- concatenate:
  take $W^{[1]}, b^{[1]}, ..., W^{[L]}, b^{[L]}$ and reshape into a big vector $\theta$
  take $dW^{[1]}, db^{[1]}, ..., dW^{[L]}, db^{[L]}$ and reshape into a big vector $d\theta$

for each i:
$$
\begin{align}
d\theta_{approx}^{[i]}&=\frac{J(\theta_1, \theta_2,...,\theta_i+\epsilon,  ...)-J(\theta_1, \theta_2,...,\theta_i-\epsilon,  ...)}{2\epsilon} \\
d\theta^{[i]}&=\frac{\partial J}{\partial \theta_i}\\
if \ \epsilon=10^{-7}&\\
&
\left.
\text{check} \quad \frac{||d\theta_{approx}-d\theta||_2}{||d\theta_{approx}||_2+||d\theta||_2} \approx
\right \{
\begin{aligned}
&10^{-7} \ \text{great}\\
&10^{-5} \ \text{okay but double check}\\
&10^{-3} \ \text{not good}
\end{aligned}
\end{align} \\
$$

#### tips

- Don't use in trainin, only to debug
- If algorithm fails grad check, look at components to try to identify bug
- Remember regularization
- Doesn't work with dropout
- Run at random initialization; perhaps again after some training


## 2 Optimization Algorithms

### 2.1 Mini-batch Gradient Descent

 split the training set into smaller ones → mini-batches

example: (m=5,000,000)
$$
\mathop{X}\limits_{(n_x,m)}=[\underbrace{x^{(1)}x^{(2)} ... x^{(1000)}}_{X^{\{1\}}} |\underbrace{x^{(1001)}x^{(1002)} ... x^{(2000)}}_{X^{\{2\}}}|...|\underbrace{... x^{(m)}}_{X^{\{5000\}}}]\\
\mathop{Y}\limits_{(1,m)}=[\underbrace{y^{(1)}y^{(2)} ... y^{(1000)}}_{Y^{\{1\}}} |\underbrace{y^{(1001)}y^{(1002)} ... y^{(2000)}}_{Y^{\{2\}}}|...|\underbrace{... y^{(m)}}_{Y^{\{5000\}}}]\\
$$
**{i}**: different mini-batches

use $X^{\{t\}},Y^{\{t\}}$ to substitute $X,Y$ in gradient descent, use a for-loop to go through

**epoch**: a single pass through the entire training set

<img src="Improving_Deep_Neural_Networks.assets/image-20200813152855653.png" alt="image-20200813152855653" style="zoom: 25%;" />

more noise

#### choosing the mini-batch size

- if **size = m** : Batch gradient descent
	(takes too much time per iteration, especially when the training set is large)
- if **size =1** : Stochastic gradient descent 
  (extremely noisy)
  (won't converge, oscillate around the region of the minium)
  (lose speedup from vectorization)

- choose a size **in between** (not too big or too small)

instruction:
1. small training set: use batch gradient descent
2. big training set: typical mini-batch size: $2^n$ (usually n=6~9, 10)

- make sure mini-batch $X^{\{t\}},Y^{\{t\}}$ fits in CPU/GPU memory

### 1.2 Exponentially weighted averages

