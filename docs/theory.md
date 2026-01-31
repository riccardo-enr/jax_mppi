# Theoretical Background

This section provides the mathematical foundations for the Model Predictive Path Integral (MPPI) control algorithm and its variants implemented in `jax_mppi`.

## Standard MPPI

Model Predictive Path Integral (MPPI) control is a sampling-based model predictive control algorithm derived from information-theoretic principles. It solves the stochastic optimal control problem by simulating multiple trajectories and updating the control policy based on their costs.

### Stochastic Optimal Control Problem

We consider a discrete-time dynamical system with dynamics:

\[
\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{v}_t
\]

where $\mathbf{x}_t \in \mathbb{R}^{n_x}$ is the state, $\mathbf{u}_t \in \mathbb{R}^{n_u}$ is the control input, and $\mathbf{v}_t \sim \mathcal{N}(0, \Sigma)$ is Gaussian noise.

The objective is to find the control sequence $U = \{\mathbf{u}_0, \dots, \mathbf{u}_{T-1}\}$ that minimizes the expected cost:

\[
J(U) = \mathbb{E} \left[ \phi(\mathbf{x}_T) + \sum_{t=0}^{T-1} \left( q(\mathbf{x}_t) + \frac{1}{2} \mathbf{u}_t^T \Sigma^{-1} \mathbf{u}_t \right) \right]
\]

where $\phi(\mathbf{x}_T)$ is the terminal cost and $q(\mathbf{x}_t)$ is the state-dependent running cost. The term $\frac{1}{2} \mathbf{u}_t^T \Sigma^{-1} \mathbf{u}_t$ represents the control effort cost.

### Information Theoretic Derivation

MPPI relies on the duality between free energy and relative entropy (KL divergence). The optimal control distribution $p^*$ is proportional to the exponential of the trajectory cost:

\[
p^*(\tau) \propto \exp\left(-\frac{1}{\lambda} S(\tau)\right)
\]

where $S(\tau)$ is the cost of a trajectory $\tau$ and $\lambda$ is a temperature parameter.

### Update Law

In practice, we approximate the optimal control by sampling $K$ trajectories around a nominal control sequence $\mathbf{u}_{nom}$. For each sample $k$, we apply a perturbation $\epsilon_k \sim \mathcal{N}(0, \Sigma)$:

\[
\mathbf{u}_{t, k} = \mathbf{u}_{nom, t} + \epsilon_{t, k}
\]

The cost for the $k$-th trajectory is computed as:

\[
S_k = \phi(\mathbf{x}_{T, k}) + \sum_{t=0}^{T-1} \left( q(\mathbf{x}_{t, k}) + \lambda \mathbf{u}_{nom, t}^T \Sigma^{-1} \epsilon_{t, k} \right)
\]

The weights for each trajectory are computed using the softmax function:

\[
w_k = \frac{\exp(-\frac{1}{\lambda} (S_k - \beta))}{\sum_{j=1}^K \exp(-\frac{1}{\lambda} (S_j - \beta))}
\]

where $\beta = \min_k S_k$ for numerical stability.

The control sequence is then updated by computing the weighted average of the perturbations:

\[
\mathbf{u}_{new, t} = \mathbf{u}_{nom, t} + \sum_{k=1}^K w_k \epsilon_{t, k}
\]

## Smooth MPPI (SMPPI)

Standard MPPI assumes the control inputs are independent across time steps, which can lead to jerky or non-smooth control signals. Smooth MPPI (SMPPI) addresses this by lifting the control problem to a higher-order space (e.g., controlling acceleration instead of velocity).

### State Augmentation

In SMPPI, the nominal trajectory $U$ represents the derivative of the actual action (e.g., acceleration). The actual action $\mathbf{a}_t$ is part of the state or computed by integrating $U$.

Let $\mathbf{u}_t$ be the command at time $t$ (from the optimizer). The action applied to the system is $\mathbf{a}_t$, updated as:

\[
\mathbf{a}_{t+1} = \mathbf{a}_t + \mathbf{u}_t \Delta t
\]

### Smoothness Cost

SMPPI explicitly penalizes changes in the action sequence to encourage smoothness. The cost function includes a term for the magnitude of the command $\mathbf{u}_t$ (which corresponds to the change in action):

\[
J_{smooth} = \sum_{t=0}^{T-1} ||\mathbf{u}_t||^2 = \sum_{t=0}^{T-1} ||\frac{\mathbf{a}_{t+1} - \mathbf{a}_t}{\Delta t}||^2
\]

This formulation ensures that the generated trajectories are smooth and feasible for systems with actuation limits or bandwidth constraints.

## Kernel MPPI (KMPPI)

Kernel MPPI (KMPPI) parameterizes the control trajectory using a set of basis functions or kernels, rather than optimizing the control input at every time step independently. This reduces the dimensionality of the optimization problem and implicitly enforces smoothness.

### RKHS Formulation

We assume the control trajectory $\mathbf{u}(t)$ lies in a Reproducing Kernel Hilbert Space (RKHS) defined by a kernel $k(t, t')$. The control is represented as a linear combination of basis functions centered at support points $t_i$:

\[
\mathbf{u}(t) = \sum_{i=1}^{M} \alpha_i k(t, t_i)
\]

where $M$ is the number of support points (often $M < T$), and $\alpha_i$ are the weights (parameters) to be optimized.

### Optimization

Instead of perturbing the control inputs $\mathbf{u}_t$ directly, KMPPI perturbs the parameters $\alpha_i$ (or equivalent control points).

Let $\theta$ represent the parameters. We sample perturbations $\delta \theta_k \sim \mathcal{N}(0, \Sigma_\theta)$. The corresponding control trajectory is:

\[
\mathbf{u}_k(t) = \text{Interpolate}(\theta + \delta \theta_k)
\]

The update rule is applied to $\theta$:

\[
\theta_{new} = \theta_{nom} + \sum_{k=1}^K w_k \delta \theta_k
\]

By choosing an appropriate kernel (e.g., RBF kernel), we can control the smoothness and frequency content of the resulting trajectories.
