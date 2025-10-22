---
title: Reparameterization and discrete coupling
subtitle: For training discrete latent-variable models
layout: default
date: 2025-10-22
keywords: computer science, machine learning
published: true
---

# title?

## Training Discrete Latent-Variable Models

### Estimating gradients via reparameterization and antithetic sampling

Let’s start simple. Consider a **stochastic binary latent variable**

$$
b \in \{0, 1\}, \quad b \sim \text{Bernoulli}(\sigma(\phi)),
$$

where the **logit** $\phi$ parameterizes the Bernoulli probability

$$
\sigma(\phi) = \frac{1}{1 + e^{-\phi}}.
$$

The logit $\phi$ can be the output of some probabilistic model whose parameters we wish to optimize.

Given a loss (or reward) function $f(b)$ , the optimization objective is

$$
\mathcal{E}(\phi) = \mathbb{E}_{b \sim \text{Bernoulli}(\sigma(\phi))}[f(b)].
$$



##### The score-function gradient

Using the standard *log-prob trick* — the identity

$$
\frac{\partial p(x)}{\partial x} = p(x) \frac{\partial \log p(x)}{\partial x},
$$

we can write the exact gradient of $\mathcal{E}(\phi)$ as

$$
\nabla_\phi \mathcal{E}(\phi)
= \mathbb{E}_{b}\!\left[f(b)\, \nabla_\phi \log p_\phi(b)\right].
$$

This expression holds for any distribution from which we can compute $\log p_\phi(b)$ , and it forms the basis of most gradient estimators for discrete latent variables.



##### The exact gradient for a single binary variable

For one binary random variable, we can compute this expectation analytically:

$$
\begin{aligned}
\nabla_\phi \mathcal{E}(\phi)
&= 
\sigma(\phi) f(1) \frac{1}{\sigma(\phi)} \frac{\partial \sigma (\phi)}{\partial_\phi} 
+ (1 - \sigma(\phi)) f(0) \frac{-1}{1 - \sigma(\phi)} \frac{\partial \sigma (\phi)}{\partial_\phi} \\
&= (f(1) - f(0)) \frac{e^{-\phi}}{(1 + e^{-\phi})^2}.
\end{aligned}
$$

This simple expression makes sense: the gradient depends only on the *difference in rewards* between the two binary outcomes, scaled by the sensitivity of the sigmoid.



##### The curse of dimensionality

If we have $k$ binary latent variables, the exact gradient would require evaluating $f(\mathbf{b})$ for all possible configurations $\mathbf{b} = (b_1, b_2, \ldots, b_k)^T \in \{0, 1\}^k$. That’s $2^k$ function evaluations — exponential in the number of variables — which quickly becomes intractable.

To avoid this combinatorial explosion, we resort to *sampling-based* gradient estimates. This is the fundamental challenge that all the methods discussed below will try to solve.



##### The REINFORCE estimator

The most direct approach is the **REINFORCE** estimator (Williams, 1992). Since

$$
\nabla_\phi \mathcal{E}(\phi)
= \mathbb{E}_{b}\!\left[f(b)\, \nabla_\phi \log p_\phi(b)\right],
$$

we can approximate this expectation via Monte Carlo samples of $b$ :

$$
g_{\text{REINFORCE}} = f(b)\, \nabla_\phi \log p_\phi(b).
$$

For the Bernoulli case,

$$
g_{\text{REINFORCE}} = (-1)^b\, f(b)\, \sigma_\phi (1 - \sigma_\phi).
$$

This estimator is **unbiased**, but it typically suffers from **high variance**, especially when $f(b)$ varies sharply between outcomes. 
This motivates the search for **lower-variance estimators**, which can often be derived through reparameterization, antithetic sampling, or control variates — directions we’ll explore next.



#### Reparameterizing the Binary Random Variable

The binary random variable $b \in \{0, 1\}$ can be expressed in terms of various **continuous latent variables**, depending on how we choose to represent the Bernoulli sampling process.  Below we explore two such formulations.



##### 1\. Exponential random variables

We begin with a simple property of exponential distributions: 
if $t_1 \sim \text{Exp}(\lambda_1)$ and $t_2 \sim \text{Exp}(\lambda_2)$ , then
$$
P(t_1 > t_2) = \frac{\lambda_2}{\lambda_1 + \lambda_2}.
$$

Also, since $t_i \overset{d}{=} \frac{1}{\lambda_i} \epsilon_i$ with $\epsilon_i \sim \text{Exp}(1)$ , we can reparameterize the Bernoulli variable $b$ as:
$$
b = \mathbf{1}\{t_1 > \epsilon_2\}, 
\quad \text{where } t_1 \sim \text{Exp}(e^{-\phi}), \ \epsilon_2 \sim \text{Exp}(1).
$$

Equivalently,

$$
b = \mathbf{1}\{ e^{\phi} \epsilon_1 > \epsilon_2 \},
\quad \epsilon_1, \epsilon_2 \sim \text{Exp}(1).
$$

The objective gradient can then be written as:

$$
\nabla_\phi \mathcal{E}(\phi)
= \nabla_\phi \mathbb{E}_{b}[f(b)]
= \nabla_\phi \, \mathbb{E}_{t_1, \epsilon_2}[f(t_1, \epsilon_2)]
= \mathbb{E}_{t_1, \epsilon_2}\!\left[
    f(t_1, \epsilon_2) \, \nabla_\phi \log p_\phi(t_1, \epsilon_2)
\right],
\tag{1}\label{eq:exp_grad}
$$

where $\nabla_\phi \log p_\phi(t_1, \epsilon_2) = \nabla_\phi \log p_\phi(t_1)$ ,  
since $\epsilon_2$ is independent of $\phi$ .

A Monte Carlo estimator for this gradient is:

$$
g_{\text{exp}} = f(\mathbf{1}\{t_1 > \epsilon_2\}) \, \nabla_\phi \log p_\phi(t_1),
$$

and for $p_\phi(t_1) = e^{-\phi} e^{-e^{-\phi} t_1}$ , we have:

$$
g_{\text{exp}} = f(\mathbf{1}\{t_1 > \epsilon_2\}) \, (-1 + e^{-\phi} t_1).
$$



##### 2\. Arbitrary continuous variable on $\mathbb{R}$

A more general view is to define the binary variable as a thresholded comparison:

$$
b = \mathbf{1}\{\epsilon < F^{-1}(\sigma(\phi))\},
$$

where $F$ is the CDF of some continuous random variable $\epsilon$ with corresponding PDF $p(\epsilon)$ .

Define the shifted variable $t = \epsilon - F^{-1}(\sigma(\phi))$ .  Then, applying the same score-function form gives:
$$
g_{\text{eps}} = f(\mathbf{1}\{t < 0\}) \, \nabla_\phi \log p_\phi(t).
$$

Any CDF–PDF pair for which the density of $t$​ can be written explicitly is a potential candidate for this reparameterization. In the previous section, just for the purpose of reparameterizing the gradients, we could have avoided introducing $\epsilon_2$​ all together. Choosing $\epsilon \sim - \exp(\lambda)$​ with $\lambda = - \log \sigma (\phi)$​ and $c = 1$​, 
$$
b = \mathbf{1}\{\epsilon > c \},
$$
fits all the conditions! The second random variable is later used to lower the variance via antithetic sampling.

##### Note 1 — Why uniform variables fail

Uniform random variables cannot be used in this formulation. The uniform distribution has *bounded support*, which causes the integration boundaries of the expectation to depend on $\sigma(\phi)$. This violates the key assumption that the parameterized density $p_\phi$ integrates over a fixed domain independent of $\phi$ .

For example, while one could write

$$
b = \mathbf{1}\{u < a\}, \quad u \sim U\!\left(0, \frac{a}{\sigma(\phi)}\right),
$$

it might seem tempting to use the simpler equivalent form

$$
b = \mathbf{1}\{u < \sigma(\phi)\}, \quad u \sim U(0, 1),
$$

but this destroys the explicit dependence of the likelihood on $\phi$, and thus prevents us from applying the log-probability trick correctly.

Nevertheless, for completeness, if we keep the parameterized density $p_\phi(u) = \frac{\sigma(\phi)}{a}$, then
$$
\nabla_\phi \mathcal{E}(\phi)
\neq \mathbb{E}_{u}\!\left[
    f(u)\, \nabla_\phi \log p_\phi(u)
\right],
$$

Because
$$
\begin{multline}
\nabla_\phi \mathcal{E}(\phi)
= \nabla_\phi \mathbb{E}_{u}\!\left[ f(u)\, \nabla_\phi \log p_\phi(u) \right] = \nabla_\phi \int_0^{a/\sigma(\phi)} p_\phi(u) f(\mathbf{1}\{u < a\}) \\
= \nabla_\phi(a / \sigma(\phi)) f(\mathbf{1}\{a / \sigma(\phi) < a\}) p_\phi(a / \sigma(\phi)) + \int_0^{a/\sigma(\phi)} f(\mathbf{1}\{u < a\}) \nabla_\phi \log p_\phi(u) \\
= -f(0) (1 - \sigma(\phi)) + \int_0^{a/\sigma(\phi)} f(\mathbf{1}\{u < a\}) \nabla_\phi \log \frac{\sigma(\phi)}{a}
\end{multline}
$$

and substituting gives:
$$
\begin{multline}
g_{\text{uniform}} = -f(0) (1 - \sigma(\phi)) + 
f(\mathbf{1}\{u > a\}) \, \nabla_\phi \log p_\phi(u) \\
= -f(0) (1 - \sigma(\phi))  + f(\mathbf{1}\{u > a\}) \, \frac{\sigma'(\phi)}{\sigma(\phi)} \\
= -f(0) (1 - \sigma(\phi))+  f(\mathbf{1}\{u > a\}) \, (1 - \sigma(\phi)).
\end{multline}
$$



##### Note 2 — Independence of $f$ from $\phi$

In equations such as $\eqref{eq:exp\_grad}$, any dependence of $f$ on $\phi$ **must be excluded**. If $f$ implicitly depends on $\phi$ (for example, through a transformation involving $\sigma(\phi)$ ), then the gradient will include an additional term involving $\nabla_\phi f$, which may not exist or may bias the estimate.

This is a subtle point: although the same function can often be expressed as a reparameterized expectation,

$$
\mathbb{E}_{t_1, \epsilon_2}[f(t_1, \epsilon_2)]
= \mathbb{E}_{\epsilon_1, \epsilon_2}[f(\mathbf{1}\{e^{\phi}\epsilon_1 > \epsilon_2\})],
$$

the second expression cannot be directly differentiated via the log-prob trick, since the underlying sampling distribution of $\epsilon_1, \epsilon_2$ no longer depends on $\phi$ .



##### Note 3 — No variance reduction (yet)

It is important to emphasize that both the exponential and general-variable formulations recover *exactly the same expected gradient* as the original Bernoulli parameterization. In fact, in preliminary experiments their variance is often *higher*. 
Thus, at this stage, the reformulations do not offer a computational advantage. They simply reveal new ways of expressing the same discrete stochastic process — a foundation we can later build upon to construct **lower-variance or antithetic estimators**.



### Future Muse — From Discrete Gradients to Thought-Like Search

The mathematical journey so far has focused on a single stochastic binary variable and its gradient estimation problem.  
But the same principles apply to far richer systems — for instance, a **decoder model** conditioned on a continuous latent variable.

---

### Continuous latent spaces as search domains

Consider a generative model $p_\theta(x \mid z)$, where $z \in \mathbb{R}^d$ is a continuous latent variable and $x$ may be a sequence, image, or text output. A trained language model, such as a GPT-like decoder, can be viewed in this way: the latent variable $z$ implicitly defines a *mode of reasoning or narrative trajectory*.

Now imagine we explore this latent space actively — by introducing stochastic perturbations, sampling correlated latent variables, and decoding each to a sequence $x(z)$. This procedure effectively generates *structured random trajectories* through the model’s output space.

In the discrete world, stochasticity arises from Bernoulli variables or categorical decisions (e.g., token sampling). In the continuous world, we can inject correlated noise into $z$ and observe how it propagates through the model’s decoding dynamics. Each latent sample defines a path through the model’s decision landscape.

---

### Correlated exploration and variance reduction

If the random variables used to perturb $z$ are **correlated**, then different trajectories in latent space can serve as **antithetic samples** — pairs of explorations that, when combined, reduce variance in the resulting gradient estimates.

This connects directly back to the discrete estimators studied earlier: variance reduction in gradient estimation is equivalent to variance reduction in *search-based reasoning*. Antithetic or symmetric sampling ensures that exploration does not wander chaotically but rather *balances* around the current latent trajectory.

Mathematically, one can view this as replacing the independent samples $\epsilon_i \sim p(\epsilon)$ by structured draws:

$$
\epsilon_1 \sim p(\epsilon), \quad \epsilon_2 = T(\epsilon_1),
$$

where $T$ is an *antithetic transform* (e.g., negation, inversion, or reflection through a symmetry of $p$ ). The resulting pair $(\epsilon_1, \epsilon_2)$ can then define two complementary latent variables $z_1, z_2$ that explore opposite directions in the latent space, yielding more informative updates or contrasts.

---

### Toward reasoning through latent gradients

If we view reasoning itself as a stochastic process — a sequence of discrete or continuous latent decisions optimized for coherence, reward, or accuracy — then learning such reasoning dynamics becomes an instance of the same gradient estimation challenge.

A “thinking” model could thus be seen as one that:

1. **Samples** latent trajectories through a continuous stochastic process;
2. **Evaluates** them via a reward or coherence measure $f(\cdot)$ ;
3. **Updates** its latent distribution parameters (analogous to $\phi$ ) using reparameterized, possibly antithetic gradient estimates.

In this view, inference and reasoning collapse into a unified optimization loop: exploration in latent space guided by structured randomness and variance-minimized gradient feedback.

---

### Closing remark

These notes only scratch the surface. The technical machinery — REINFORCE, reparameterization, antithetic sampling — is not merely a toolkit for optimizing stochastic neural networks. It may also describe a *mechanistic substrate* for exploratory, reasoning-like computation: a bridge between discrete logical inference and continuous gradient-based learning.

