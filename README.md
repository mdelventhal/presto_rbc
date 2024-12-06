# presto_rbc
#### Prepared December 2024 by [Matt Delventhal](mailto:delventhal.m@gmail.com)

A calibrated basic RBC model with editable parameters

## Theory

A discrete-time adaptation of the model presented in Chapter 5 of David Romer's *Advanced Macroeconomics, 4th Edition*. Another possibly useful exposition can be found [in these slides from Jesus Fernandez-Villaverde](https://www.sas.upenn.edu/~jesusfv/lecture12_rbc.pdf).

### Households
At each point in time the economy is populated by $N_t$ identical households with 1 member each who own capital and are endowed with 1 unit of labor. They rent their capital $k_t$ for return $r_t$ and rent a fraction $l_t$ of their labor for wage $w_t$. They value consumption $c_t$ and leisure $1-l_t$.

The problem of a representative household can be written thus:

\begin{align\*}&\max\limits_{c_t,l_t,k_{t+1}} \left \{\sum\limits_{t=0}^{\infty} \beta^t \left [\ln c_t + \psi \ln \left (1-l_t \right ) \right ]\right \} \\
& \quad \quad \quad \quad \quad \quad \quad \text{ such that } \\
& \quad \quad \quad c_t + k_{t+1} \leq w_t l_t + r_t k_t + (1-\delta) k_t
\end{align\*}

...where $\beta \in (0,1)$ represents time prefence, $\delta \in [0,1]$ represents the depreciation rate of capital, and $\psi>0$ determines the relative weight households place on leisure versus the consumption of goods and services.

### Firms

The economy is populated by a large number of identical firms who seek to maximize profits by combining capital and labor which they rent at rates $r_t$ and $w_t$, which the firms take as given. The aggregate quantity of capital in the economy at each point in time is equal to $N_t k_t = K_t$, and the aggregate labor supply is equal to $N_t l_t = L_t$.

All firms share an identical Cobb-Douglas production function with TFP $A_t > 0$ and capital share $\alpha \in (0,1)$ such that aggregate production $Y_t$ and market equilibrium capital return $r_t$ and wage $w_t$ can be characterized by the following:


\begin{align*}Y_t &= A_t K_t^{\alpha} L_t^{1-\alpha}\\
\\
r_t &= \alpha A_t k_t^{\alpha-1} l_t^{1-\alpha}\\
\\
w_t &= \alpha A_t k_t^{\alpha} l_t^{-\alpha}
\end{align*}

Output per capita $y_t$ can then be characterized as
$$y_t = A_t k_t^{\alpha} l_t^{1-\alpha}$$

### Stochastic TFP

TFP $A_t$ is made up of a deterministic component which grows at constant rate $g_A$ and a stochastic component $\tilde{A}_t$ such that


\begin{align*}\ln A_t = \ln A_0 + (1 + g_A) \cdot t + \ln \tilde{A}_t
\end{align*}

The evolution of $\tilde{A}_t$ is governed according to a first-order auto-regressive process:


\begin{align*}\ln \tilde{A}_t = \rho \ln \tilde{A}_{t-1} + \varepsilon_t
\end{align*}

...where $\rho \in [-1,1]$ governs the strength of shock persistence and $\varepsilon_t$ represents a series of independent random draws from a normal distribution with mean 0 and standard deviation $\sigma$.

### Detrending
Define detrended variables as follows:


\begin{align*}\tilde{k}_t &= \frac{k_t}{\left (A_0 (1+g_A)^t \right )^{\frac{1}{1-\alpha}}}\\
\\
\tilde{c}_t &= \frac{c_t}{\left (A_0 (1+g_A)^t \right )^{\frac{1}{1-\alpha}}}\\
\\
\tilde{y}_t &= \frac{y_t}{\left (A_0 (1+g_A)^t \right )^{\frac{1}{1-\alpha}}}\\
\end{align*}

### Solution

#### Leisure-consumption tradeoff

The household's optimal tradeoff between consumption and leisure for $\alpha \in (0,1)$ is characterized by


\begin{align*}l_t \left (\frac{\psi}{1-\alpha} + 1\right ) + l_t^{\alpha} \frac{\psi}{1-\alpha} \frac{k_t (1-\delta) - k_{t+1}}{A_t k_t^{\alpha}} - 1 = 0
\end{align*}

Taking the choice of $k_{t+1}$ as given, the choice of labor supply $l_t$ can be solved for numerically-or in the special case of $\alpha = \frac{1}{2}$, analytically.

The de-trended equivalent of the above condition is

\begin{align*}l_t \left (\frac{\psi}{1-\alpha} + 1\right ) + l_t^{\alpha} \frac{\psi}{1-\alpha} \frac{\tilde{k}_t (1-\delta) - (1+g_A)^{\frac{-1}{1-\alpha}}\tilde{k}_{t+1}}{\tilde{A}_t \tilde{k}_t^{\alpha}} - 1 = 0
\end{align*}

#### Savings decision

As a shortcut, a log-linearized function is used to determine $k_{t+1}$, such that


\begin{align*}\ln \tilde{k}_{t+1} = a_{kA} \ln \tilde{A}_t + a_{kk} \ln \tilde{k}_t
\end{align*}
