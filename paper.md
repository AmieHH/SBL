# 论文全文（LaTeX格式）

## 标题和作者

**A Sparse Bayesian Learning Algorithm With Dictionary Parameter Estimation**

**Authors:**
- Thomas L. Hansen*, Mihai A. Badiu*, Bernard H. Fleury* and Bhaskar D. Rao†
- *Dept. of Electronic Systems, Aalborg University, Fr. Bajers Vej 7, DK-9220, Aalborg, Denmark
- †Dept. of Electrical and Computer Engineering, University of California at San Diego, 9500 Gilman Drive, La Jolla, CA 92093, USA

---

## ABSTRACT

This paper concerns sparse decomposition of a noisy signal into atoms which are specified by unknown continuous-valued parameters. An example could be estimation of the model order, frequencies and amplitudes of a superposition of complex sinusoids. The common approach is to reduce the continuous parameter space to a fixed grid of points, thus restricting the solution space. In this work, we avoid discretization by working directly with the signal model containing parameterized atoms. Inspired by the "fast inference scheme" by Tipping and Faul we develop a novel sparse Bayesian learning (SBL) algorithm, which estimates the atom parameters along with the model order and weighting coefficients. Numerical experiments for spectral estimation with closely-spaced frequency components, show that the proposed SBL algorithm outperforms state-of-the-art subspace and compressed sensing methods.

---

## I. INTRODUCTION

Suppose we have a length-$N$ signal $\mathbf{x}$ given by a weighted sum of $K \ll N$ elementary functions, so-called signal atoms:

$$\mathbf{x} = \sum_{i=1}^{K} \psi(\theta_i)\alpha_i \tag{1}$$

where $\boldsymbol{\theta} = [\theta_1, \cdots, \theta_K]^T$ is a vector of atom (dictionary) parameters and $\boldsymbol{\alpha} = [\alpha_1, \cdots, \alpha_K]^T$ is a vector of weighting coefficients. We specify the atoms by the vector-valued function $\psi : [0, 1) \to \mathbb{C}^N$. We take noisy measurements $\mathbf{y} \in \mathbb{C}^N$ as $\mathbf{y} = \mathbf{x} + \mathbf{w}$, where $\mathbf{w} \in \mathbb{C}^N$ is a zero-mean complex Gaussian noise vector with independent and identical distributed (i.i.d.) entries of variance $\lambda^{-1}$.

The problem of estimating $K$, $\boldsymbol{\theta}$ and $\boldsymbol{\alpha}$ given the atom specification $\psi(\cdot)$ and observation $\mathbf{y}$ is ubiquitous in signal processing. When the Fourier atom is considered, i.e. $\psi(\theta_i) = [e^{j2\pi\theta_i \cdot 0}, \cdots, e^{j2\pi\theta_i(N-1)}]^T$, the problem reduces to the line spectral estimation problem, which has a wide range of applications, e.g. direction of arrival estimation. The problem also arises in the context of compressed sensing (CS) reconstruction: When a sensing matrix $\boldsymbol{\Phi}$ is employed, the atoms in our model become $\psi(\cdot) = \boldsymbol{\Phi}\tilde{\psi}(\cdot)$, where $\tilde{\psi}(\cdot)$ is the atom specified without a sensing matrix.

A common approach, particularly in CS [1], is to discretize the parameter space into a (uniform) grid of $M \geq N$ values on $[0, 1)$. Then, a sparse representation of $\mathbf{x}$ is sought in the dictionary obtained by evaluating $\psi(\cdot)$ at the $M$ grid points. However, the mismatch between the atoms in the dictionary and the true atoms limits the accuracy of the fixed dictionary approach. To combat the severity of the atom mismatch a finer grid can be employed, leading to two undesired effects: a) the dictionary becomes increasingly coherent, rendering the estimation problem ill-posed [1] and b) the larger size of the dictionary results in higher computational cost of estimation.

Model-based CS can be applied to mitigate the coherence issue occurring with fixed dictionaries, see for example [2], [3]. If the objective is to reconstruct the signal vector $\mathbf{x}$, and not to find a decomposition into atoms, "analysis sparsity" [4], [5] alleviates the need for dictionary incoherence. All these coherence-controlling methods suffer from high computational complexity when a fine grid is used.

Recent works [6] formulate our estimation problem as a total variation norm minimization problem. The strongly related parallel development [7] uses a similar atomic norm minimization. Minimizing the $\ell_1$-norm promotes sparse estimates. The idea in [6], [7] is to generalize the $\ell_1$-norm from vectors to the real line, such that minimizing the norm promotes a sparse signal on the real line, i.e. a sum of spikes. For Fourier atoms, [6], [7] rewrite the norm minimization as a semi-definite program. The theoretical analysis of these works requires the frequencies in $\boldsymbol{\theta}$ to be well-separated, e.g. by at least $2/N$ in [6].

Another recently-proposed approach is to introduce a complementary dictionary which characterizes the basis mismatch, e.g. [8], [9].

For Fourier atoms (1) reduces to the line spectral estimation problem and many methods have been proposed; see [10] for a list of references. The most prominent of these methods are the so-called subspace methods [11], e.g. ESPRIT [12].

In this paper, we devise a sparse Bayesian learning (SBL) algorithm for estimating $K$, $\boldsymbol{\theta}$ and $\boldsymbol{\alpha}$. Since most SBL methods (e.g. [13]–[16]) are developed for discrete dictionaries, they suffer from the mentioned drawbacks when applied to our problem. Therefore, we instead use the parameterized model (1) to devise our algorithm. Specifically, we extend the sparse prior model proposed in [13] and devise an inference scheme, inspired by [14], which estimates $K$, $\boldsymbol{\theta}$ and $\boldsymbol{\alpha}$. A parallel development is found in [17], which uses a variational Bayesian method for estimation.

---

## II. PROBABILISTIC MODELLING

Our probabilistic model is an extension of that in [13] to include modelling of the atom parameter vector $\boldsymbol{\theta}$. The Bayesian network representation of our model is shown in Fig. 1. We note that the model in [13] is itself obtained by modifying the model in [15] such that $\boldsymbol{\alpha}$ depends on $\lambda$. This modification is employed since it gives certain benefits in computational complexity as discussed in [13].

The following treatment is valid for both the real and complex cases. The observation $\mathbf{y}$ is taken in white Gaussian noise with variance $\lambda^{-1}$:

$$p(\mathbf{y}|\boldsymbol{\theta}, \boldsymbol{\alpha}, \lambda) = \mathcal{N}(\mathbf{y}|\boldsymbol{\Psi}(\boldsymbol{\theta})\boldsymbol{\alpha}, \lambda^{-1}\mathbf{I}) \tag{2}$$

where the dictionary matrix $\boldsymbol{\Psi}(\boldsymbol{\theta}) = [\psi(\theta_1), \cdots, \psi(\theta_K)]$ contains the atoms as columns. The noise precision $\lambda$ is modelled as gamma distributed with shape $a$ and rate $b$: $p(\lambda) = \text{Ga}(\lambda|a, b)$. The atom parameter vector $\boldsymbol{\theta} \in [0, 1)^K$ has i.i.d. uniformly distributed entries: $p(\boldsymbol{\theta}) = \prod_{i=1}^{K} \text{unif}(\theta_i|0, 1)$. The coefficients $\boldsymbol{\alpha}$ are modelled through a two-layer hierarchical specification. The first layer is a zero-mean Gaussian distribution, i.e. $p(\boldsymbol{\alpha}|\boldsymbol{\gamma}, \lambda) = \mathcal{N}(\boldsymbol{\alpha}|\mathbf{0}, \lambda^{-1}\boldsymbol{\Gamma})$, where $\boldsymbol{\Gamma} = \text{diag}(\boldsymbol{\gamma})$. The vector $\boldsymbol{\gamma}$ constitute the second layer. Its entries are modelled as i.i.d. gamma distributed: $p(\boldsymbol{\gamma}) = \prod_{i=1}^{K} \text{Ga}(\gamma_i|\varepsilon, \eta)$.

**Note on notation:** The multivariate normal density is parameterized to encompass both the real ($\rho = 1/2$) and complex ($\rho = 1$) cases:

$$\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \left(\frac{\rho}{\pi}\right)^{\rho \dim(\mathbf{x})} |\boldsymbol{\Sigma}|^{-\rho} \exp\left(-\rho(\mathbf{x}-\boldsymbol{\mu})^H\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

---

## III. BAYESIAN INFERENCE

To infer on $(K, \boldsymbol{\theta}, \boldsymbol{\alpha})$, we proceed by applying Type-II estimation [18], i.e. we use Bayesian inference to find point estimates $(\hat{K}, \hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\gamma}}, \hat{\lambda})$ of the parameters $(K, \boldsymbol{\theta}, \boldsymbol{\gamma}, \lambda)$ and find the estimate $\hat{\boldsymbol{\alpha}}$ of $\boldsymbol{\alpha}$ as the mode of $p(\boldsymbol{\alpha}|\mathbf{y}, \hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\gamma}}, \hat{\lambda})$. From Bayes rule, $p(\boldsymbol{\alpha}|\mathbf{y}, \boldsymbol{\theta}, \boldsymbol{\gamma}, \lambda)$ turns out to be a normal density:

$$p(\boldsymbol{\alpha}|\mathbf{y}, \boldsymbol{\theta}, \boldsymbol{\gamma}, \lambda) = \mathcal{N}(\boldsymbol{\alpha}|\boldsymbol{\mu}, \lambda^{-1}\boldsymbol{\Sigma}) \tag{3}$$

where

$$\boldsymbol{\mu} = \boldsymbol{\Sigma}\boldsymbol{\Psi}^H(\boldsymbol{\theta})\mathbf{y} \tag{4}$$

$$\boldsymbol{\Sigma} = \left[\boldsymbol{\Psi}^H(\boldsymbol{\theta})\boldsymbol{\Psi}(\boldsymbol{\theta}) + \boldsymbol{\Gamma}^{-1}\right]^{-1} \tag{5}$$

We find $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\gamma}}, \hat{\lambda})$ as an approximation of the maximum a posteriori (MAP) estimate

$$(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\gamma}}, \hat{\lambda})_{\text{MAP}} = \arg\max_{(\boldsymbol{\theta},\boldsymbol{\gamma},\lambda)} \ln p(\boldsymbol{\theta}, \boldsymbol{\gamma}, \lambda|\mathbf{y}) \tag{6}$$

Following the steps of [13], we proceed by writing the log-posterior by its factors:

$$\ln p(\boldsymbol{\theta}, \boldsymbol{\gamma}, \lambda|\mathbf{y}) \propto \ln p(\mathbf{y}|\boldsymbol{\theta}, \boldsymbol{\gamma}, \lambda)p(\boldsymbol{\theta})p(\boldsymbol{\gamma})p(\lambda) \tag{7}$$

where $x \propto y$ denotes $x = y + \text{const}$. The marginal likelihood can be found by marginalizing the coefficient vector $\boldsymbol{\alpha}$ out:

$$p(\mathbf{y}|\boldsymbol{\theta}, \boldsymbol{\gamma}, \lambda) = \mathcal{N}(\mathbf{y}|\mathbf{0}, \lambda^{-1}\mathbf{B}) \tag{8}$$

where $\mathbf{B} = \left[\mathbf{I} - \boldsymbol{\Psi}(\boldsymbol{\theta})\boldsymbol{\Sigma}\boldsymbol{\Psi}^H(\boldsymbol{\theta})\right]^{-1} = \mathbf{I} + \boldsymbol{\Psi}(\boldsymbol{\theta})\boldsymbol{\Gamma}\boldsymbol{\Psi}^H(\boldsymbol{\theta})$.

Using this result, we can rewrite the log-posterior (7) such that one set of parameters $(\theta_i, \gamma_i, \lambda)$ appear explicitly:

$$\begin{align}
\ln p(\boldsymbol{\theta}, \boldsymbol{\gamma}, \lambda|\mathbf{y}) \propto &\rho N \ln \lambda - \rho \ln |\mathbf{B}_{-i}| - \rho \ln(1 + \gamma_i s_i) - \rho\lambda \mathbf{y}^H\mathbf{B}_{-i}^{-1}\mathbf{y} \\
&+ \frac{\rho\lambda|q_i|^2}{\gamma_i^{-1} + s_i} + \sum_{k=1}^{K}\{(\varepsilon - 1)\ln\gamma_k - \eta\gamma_k\} \\
&+ (a - 1)\ln\lambda - b\lambda
\end{align} \tag{9}$$

where $\mathbf{B}_{-i} = \mathbf{I} + \boldsymbol{\Psi}(\boldsymbol{\theta}_{-i})\boldsymbol{\Gamma}_{-i}\boldsymbol{\Psi}^H(\boldsymbol{\theta}_{-i})$ and $\boldsymbol{\Gamma}_{-i} = \text{diag}(\boldsymbol{\gamma}_{-i})$. The notation $\mathbf{a}_{-i}$ denotes a vector $\mathbf{a}$ with the $i$th component removed. We have further defined the quantities

$$s_i \triangleq \psi^H(\theta_i)\mathbf{B}_{-i}^{-1}\psi(\theta_i) \quad \text{and} \quad q_i \triangleq \psi^H(\theta_i)\mathbf{B}_{-i}^{-1}\mathbf{y} \tag{10}$$

To estimate the parameters $\{\theta_i, \gamma_i\}_{i=1,\ldots,\hat{K}}$ and $\lambda$ we iteratively maximize (9) with respect to one parameter, while keeping all other parameters fixed at their current estimate. To do so, we take partial derivatives of (9) w.r.t $\lambda$ and $\gamma_i$ and solve for the roots. Following a procedure similar to that in [15], to analyse the stationary points, we get the updates

$$\hat{\lambda} = \frac{\rho N + a - 1}{\rho \mathbf{y}^H\hat{\mathbf{B}}^{-1}\mathbf{y} - b} \tag{11}$$

$$\hat{\gamma}_i = \begin{cases}
\frac{-(2\varepsilon-2-\rho)\hat{s}_i - \rho\hat{\lambda}|\hat{q}_i|^2 - \sqrt{\Delta}}{2(\varepsilon-1-\rho)\hat{s}_i^2} & \rho\hat{\lambda}|\hat{q}_i|^2 > \delta \\
0 & \text{otherwise}
\end{cases} \tag{12}$$

where $\Delta = \left[(2\varepsilon - 2 - \rho)\hat{s}_i + \rho\hat{\lambda}|\hat{q}_i|^2\right]^2 - 4(\varepsilon-1)(\varepsilon-1-\rho)\hat{s}_i^2$ and $\delta = \left[2 + \rho - 2\varepsilon + 2\sqrt{(1-\varepsilon)(1+\rho-\varepsilon)}\right]\hat{s}_i$.

It is not tractable to obtain similar closed-form expression for updating $\hat{\theta}_i$. We instead use Newton's method for unconstrained optimization, which iteratively updates $\hat{\theta}_i$ as

$$\hat{\theta}_i^{\text{new}} = \hat{\theta}_i^{\text{old}} - l'(\hat{\theta}_i^{\text{old}}) / l''(\hat{\theta}_i^{\text{old}}) \tag{13}$$

where $l'(\theta_i)$ and $l''(\theta_i)$ are the first and second partial derivatives of (9) with respect to $\theta_i$ as given in the Appendix.

---

## ALGORITHM 1: SBL with Dictionary Parameter Estimation

**Input:** Signal measurement $\mathbf{y}$.

**Output:** Estimates of the model order $\hat{K}$, atom parameters in $\hat{\boldsymbol{\theta}} \in [0, 1)^{\hat{K}}$ and coefficients in $\hat{\boldsymbol{\alpha}} \in \mathbb{C}^{\hat{K}}$.

**Parameters:** Prior parameters $\varepsilon, a, b$ ($\eta = 0$ assumed.)

1. $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\gamma}}) \leftarrow$ Empty vectors; $\hat{\lambda} \leftarrow 100$
2. **while** Stopping criterion not met **do**
3.     **if** Iteration number $\in \{0, 5, 10, 20, 30, \ldots\}$ **then**
4.         **for** $i = 0, \ldots, 3N - 1$ **do**
5.             $\hat{\theta}_{\text{candidate}} \leftarrow$ Calc. from (13), with $\hat{\theta}^{\text{old}} = \frac{i}{3N}$
6.             $\hat{\gamma}_{\text{candidate}} \leftarrow$ Calculate from (12).
7.             **if** $\gamma_{\text{candidate}} > 0$ **then**
8.                 Append $(\theta_{\text{candidate}}, \gamma_{\text{candidate}})$ to $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\gamma}})$.
9.             **end**
10.        **end**
11.    **end**
12.    **for** $i = 1, \ldots, ||\hat{\boldsymbol{\theta}}||_0$ **do**
13.        $\hat{\theta}_i \leftarrow$ Update from (13).
14.        $\hat{\gamma}_i \leftarrow$ Update from (12).
15.        **if** $\hat{\gamma}_i == 0$ **then**
16.            Remove component $i$ from $\hat{\boldsymbol{\theta}}$ and $\hat{\boldsymbol{\gamma}}$.
17.        **end**
18.    **end**
19.    $\hat{\lambda} \leftarrow$ Update from (11).
20. **end**
21. $\hat{\boldsymbol{\alpha}} \leftarrow \hat{\boldsymbol{\mu}}$ where $\hat{\boldsymbol{\mu}}$ calculated from (4) based on $(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\gamma}})$.

**Stopping criterion:** In the $k$th iteration use $||\hat{\boldsymbol{\theta}}_k - \hat{\boldsymbol{\theta}}_{k-1}||_{\infty} < 10^{-6}/N$.

**Computational complexity:**
- With iterative updates of $\hat{\boldsymbol{\Sigma}}$: $O(N^2\hat{K})$ when searching for new atoms, $O(N\hat{K}^2)$ otherwise.

---

## IV. NUMERICAL EXPERIMENTS

To investigate the performance of our algorithm, we conduct a numerical experiment with Fourier atoms. We use $N = 100$ measurements and $K = 10$ signal components. We generate $\boldsymbol{\theta}$ consisting of 5 pairs of frequencies. The distance $d(\theta_i, \theta_{i+1})$ between two paired frequencies is i.i.d. uniform random on the interval $[0.7/N, 1/N]$. The distance metric $d(\cdot, \cdot)$ is the wrap-around distance on the interval $[0, 1)$. The pairs are located randomly such that any set of frequencies $(\theta_i, \theta_j)$ which are not paired are separated by at least $d(\theta_i, \theta_j) \geq 1.5/N$. The complex amplitudes of the signal components in $\boldsymbol{\alpha}$ are generated i.i.d. with uniform random phase on $[0, 2\pi)$ and amplitudes drawn from a normal density of mean 1 and variance 0.1.

We measure performance in terms of normalized mean-squared error (MSE) of the reconstructed signal $\hat{\mathbf{x}} = \boldsymbol{\Psi}(\hat{\boldsymbol{\theta}})\hat{\boldsymbol{\alpha}}$ and the following performance metric for $\hat{\boldsymbol{\theta}}$:

$$\beta(\boldsymbol{\theta}, \hat{\boldsymbol{\theta}}) \triangleq \frac{1}{K}\sum_{i=1}^{K}\left(\min_{\hat{\theta}\in\hat{\boldsymbol{\theta}}} d(\hat{\theta}, \theta_i)\right)^2 \tag{14}$$

where the notation $\hat{\theta} \in \hat{\boldsymbol{\theta}}$ signifies that $\hat{\theta}$ is given as one of the entries in $\hat{\boldsymbol{\theta}}$. Further, we show histograms of the model order estimation error at 15 dB SNR for some of the algorithms. All reported values are averaged over 100 trials.

**Comparison algorithms:**
- **Fixed dictionary SBL** [13]: Dictionary size $M = 10N$, Jeffreys priors ($\varepsilon = \eta = a = b = 0$)
- **SDP** [6]: Total variation norm minimization via semi-definite programming
- **CCBP** [9]: Complex continuous basis pursuit, dictionary size $M = 3N$
- **SIHT IP** [2]: Spectral iterative hard thresholding, dictionary $M = 10N$, $\hat{K} = 3K$
- **Analysis BPDN** [5]: Dictionary size $M = 10N$
- **ESPRIT** [12]: Given true model order $\hat{K} = K$
- **Oracle estimator**: Least-squares with true frequencies
- **CRB**: Cramér-Rao bound calculated as in [19]

**Results (Fig. 2):**

1. **Reconstruction MSE:** Proposed SBL superior at SNR $\geq 5$ dB, matches Oracle at high SNR
2. **Parameter estimation:** Proposed SBL attains CRB at SNR $\geq 5$ dB
3. **Model order:** Small estimation error across all SNR, unlike fixed dictionary methods
4. **Saturation effect:** Fixed dictionary methods saturate due to grid resolution limits
5. **Comparison to ESPRIT:** Outperforms ESPRIT even when ESPRIT is given true $K$

---

## V. CONCLUSION

In this paper we addressed the so-called off-grid sparse decomposition problem which consists in decomposing a noisy signal into atoms specified by unknown continuous-valued parameters. We found that a convenient way to avoid the undesired effects of a discretized atom parameter space is to consider the atom parameters as unknown variables to be estimated along with the model order and atom coefficients. Thus, we proposed a novel SBL algorithm based on an extension of the probabilistic model in [13]. Inspired by the constructive scheme of [14], we devised update expressions which provide a simple criterion for inclusion of an atom into the estimated model; candidate atoms for inclusion are identified via a grid search. Unlike for the rest of the variables, the updates of the atom parameters cannot be computed in closed-form, so we resorted to Newton's method to update these estimates.

The numerical results show that our algorithm is superior to the reference algorithms for spectral estimation with closely-spaced frequency components. This is remarkable since our algorithm estimates the model order, while many of the reference algorithms are given the true value. An interesting aspect for further research is to reduce the computational demands of the algorithm, in particular that connected with the search for new atoms to include in the model estimate.

---

## APPENDIX: PARTIAL DERIVATIVES OF (9) W.R.T. $\theta_i$

$$l'(\theta_i) = \frac{\rho\hat{\lambda}\hat{\gamma}_i}{1 + \hat{\gamma}_i\hat{s}_i}\frac{\partial|\hat{q}_i|^2}{\partial\theta_i} - \left(\frac{\rho\hat{\gamma}_i}{1 + \hat{\gamma}_i\hat{s}_i} + \frac{\rho\hat{\lambda}\hat{\gamma}_i^2|\hat{q}_i|^2}{(1 + \hat{\gamma}_i\hat{s}_i)^2}\right)\frac{\partial\hat{s}_i}{\partial\theta_i}$$

$$\begin{align}
l''(\theta_i) = &\left(\hat{\lambda}\frac{\partial^2|\hat{q}_i|^2}{\partial\theta_i^2} - \frac{\partial^2\hat{s}_i}{\partial\theta_i^2}\right)\frac{\rho\hat{\gamma}_i}{1 + \hat{\gamma}_i\hat{s}_i} \\
&+ \left(\frac{\partial\hat{s}_i}{\partial\theta_i}\right)^2 \frac{2\rho\hat{\lambda}\hat{\gamma}_i^3|\hat{q}_i|^2}{(1 + \hat{\gamma}_i\hat{s}_i)^3} \\
&+ \left[\left(\frac{\partial\hat{s}_i}{\partial\theta_i}\right)^2 - 2\hat{\lambda}\frac{\partial\hat{s}_i}{\partial\theta_i}\frac{\partial|\hat{q}_i|^2}{\partial\theta_i} - \hat{\lambda}|\hat{q}_i|^2\frac{\partial^2\hat{s}_i}{\partial\theta_i^2}\right]\frac{\rho\hat{\gamma}_i^2}{(1 + \hat{\gamma}_i\hat{s}_i)^2}
\end{align}$$

$$\frac{\partial\hat{s}_i}{\partial\theta_i} = 2\text{Re}\left\{\psi^H(\theta_i)\hat{\mathbf{B}}_{-i}^{-1}\frac{\partial\psi(\theta_i)}{\partial\theta_i}\right\}$$

$$\frac{\partial|\hat{q}_i|^2}{\partial\theta_i} = 2\text{Re}\left\{\hat{q}_i^* \mathbf{y}^H\hat{\mathbf{B}}_{-i}^{-1}\frac{\partial\psi(\theta_i)}{\partial\theta_i}\right\}$$

$$\frac{\partial^2\hat{s}_i}{\partial\theta_i^2} = 2\text{Re}\left(\psi^H(\theta_i)\hat{\mathbf{B}}_{-i}^{-1}\frac{\partial^2\psi(\theta_i)}{\partial\theta_i^2} + \frac{\partial\psi^H(\theta_i)}{\partial\theta_i}\hat{\mathbf{B}}_{-i}^{-1}\frac{\partial\psi(\theta_i)}{\partial\theta_i}\right)$$

$$\frac{\partial^2|\hat{q}_i|^2}{\partial\theta_i^2} = 2\text{Re}\left\{\mathbf{y}^H\hat{\mathbf{B}}_{-i}^{-1}\left[\frac{\partial\psi(\theta_i)}{\partial\theta_i}\frac{\partial\psi^H(\theta_i)}{\partial\theta_i} + \frac{\partial^2\psi(\theta_i)}{\partial\theta_i^2}\psi^H(\theta_i)\right]\hat{\mathbf{B}}_{-i}^{-1}\mathbf{y}\right\}$$

**For Fourier atoms:** $\frac{\partial\psi(\theta_i)}{\partial\theta_i} = \mathbf{D}\psi(\theta_i)$ and $\frac{\partial^2\psi(\theta_i)}{\partial\theta_i^2} = \mathbf{D}^2\psi(\theta_i)$, where

$$\mathbf{D} = \text{diag}([0, j2\pi, \cdots, j2\pi(N-1)])$$

---

## REFERENCES

[References omitted for brevity - 19 references total]
