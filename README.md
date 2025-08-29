# PenFolioOp
Portfolio Optimizations for *Pension Funds*

[![codecov](https://codecov.io/gh/quantfinlib/penfolioop/graph/badge.svg?token=Z60B2PYJ44)](https://codecov.io/gh/quantfinlib/penfolioop)
[![tests](https://github.com/quantfinlib/penfolioop/actions/workflows/test.yml/badge.svg)](https://github.com/quantfinlib/penfolioop/actions/workflows/test.yml)
[![docs](https://github.com/quantfinlib/penfolioop/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/quantfinlib/penfolioop/actions/workflows/gh-pages.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/quantfinlib/penfolioop/blob/main/LICENSE)


## Conventions

In this framework, for convenience, we make use of the following conventions:

`expected_returns` $\mathbf{R}$ is an array of the expected returns of the assets and the liabilities.

$$
\mathbf{R} = \begin{bmatrix}
R_1 \\
R_2 \\
\vdots \\
R_n \\
R_L
\end{bmatrix} = \begin{bmatrix}
\mathbf{R}_{A} \\
R_L
\end{bmatrix}
$$

`covariance_matrix` $\Sigma$ is the covariance matrix of the asset and the liability returns.

$$
\Sigma = \begin{bmatrix}
\Sigma_{A} & \Sigma_{AL} \\
\Sigma_{AL} & \sigma^{2}_{L}
\end{bmatrix},
$$

where $\Sigma_{A}$ is the covariance matrix of the asset returns, $\Sigma_{AL}$ is a vector of the covariance between the asset and liability returns, and $\sigma^{2}_{L}$ is the variance of the liability returns.

The output of the optimization process is a weight vector $\mathbf{W}$ consisting of the optimal asset weights, and the liability weight is always set to -1. The optimization process aims to find the asset weights that maximize or minimize the chosen objective function while satisfying the specified constraints.

$$
\mathbf{W} = \begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n \\
-1
\end{bmatrix} = \begin{bmatrix}
\mathbf{W}_{A} \\
-1
\end{bmatrix}
$$

`surplus_return` $\mathbf{R}_{S}$ is the return of the portfolio minus the return of the liabilities.

$$R_{S} = R_{P} - R_{L} = W_{A} ^ {T}R_{A} - R_{L} = W ^ {T} R$$


`surplus_variance` $\sigma^{2}_{S}$ is the variance of the surplus returns: $`\sigma^{2}_{S} = W_{A}^{T} \Sigma_{A} W_{A} - 2 W_{A}^{T} \Sigma_{AL} + \sigma^{2}_{L} = W^{T} \Sigma W `$

## Optimizers

With the defined surplus return and variance, we can now outline the optimization problems.
All the optimizers are subject to these general constraints:

$$
    \begin{align*}
    (1) &\quad& \sum_{i=1}^{n_{assets}} w_{i} = \mathrm{SUM}(W_{A}) = 1, \\
    (2) &\quad& w_{i} \geq 0, \quad \forall i \in \{1, \ldots, n_{assets}\}, \\
    (3) &\quad& w_{L} = -1
    \end{align*}
$$



| optimizer                                | formulation                                                              |   constraints                                                   |
|------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------|
| `max_surplus_return_optimizer`| $\underset{\mathbf{W}}{\mathrm{maximize}} \quad \mathbf{W}^{T}\mathbf{R}$           | $\mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W} \leq$ `surplus_risk_upper_limit`|
| `min_surplus_variance_optimizer`| $\underset{\mathbf{W}}{\mathrm{minimize}} \quad \mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W}$           | $\mathbf{W}^{T}\mathbf{R} \geq$ `surplus_return_lower_limit`|
| `max_surplus_sharpe_ratio_optimizer`| $\underset{\mathbf{W}}{\mathrm{maximize}} \quad \frac{\mathbf{W}^{T}\mathbf{R}}{\sqrt{\mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W}}}$           | `None` |
| `surplus_mean_variance_optimizer`| $\underset{\mathbf{W}}{\mathrm{maximize}} \\ \quad  \mathbf{W}^{T}\mathbf{R} -  \frac{\lambda}{2}  \mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W}$           | `None` |

In the above table, $\lambda$ is a risk aversion parameter that balances the trade-off between maximizing surplus returns and minimizing surplus risk.
In addition to the above optimizers, the user can also call the `efficient_frontier` function to compute the weights of the efficient frontier portfolio.
These portfolios can be found by varying the  `surplus_return_lower_limit` in the following `min_surplus_variance_optimizer` optimizer. In this case, the user needs to provide a range of values for the `surplus_return_lower_limit` parameter.


## Additional Constraints

Asset weight constraints can be applied to ensure that the portfolio adheres to specific investment guidelines. 



## Example

Usage of the library is straightforward. You can create a portfolio object, define your assets, and then use the optimizers to find the optimal asset weights based on your constraints and objectives.
The first step is to create a `Portfolio` object with your assets, their expected returns, and covariances. The last item in the list of assets should be the liability, which is treated differently ,from the other assets in the optimization process. The optimizaters always set the liability weight to -1 and require the other asset weights to be between 0 and 1 and sum to 1.

The user can then define additional constraints on the asset weights, such as requiring a minimum or maximum weight for certain assets or limiting the weight of one or more assets to be less than another.

For a comprehensive description of the constraints, refer to the API documentation.


```python

import numpy as np

from penfolioop.portfolio import Portfolio
from penfolioop.optimizers import max_surplus_return_optimizer

names = ['Asset A', 'Asset B', 'Asset C', 'Liability']
expected_returns = np.array([0.05, 0.07, 0.06, 0.04])
covariance_matrix = np.array([[0.0001, 0.00005, 0.00002, 0.00003],
               [0.00005, 0.0002, 0.00001, 0.00004],
               [0.00002, 0.00001, 0.00015, 0.00002],
               [0.00003, 0.00004, 0.00002, 0.0001]]

portfolio = Portfolio(names=names, expected_returns=expected_returns, covariance_matrix=covariance_matrix)

constraints = [
    {
        'left_indices': ['Asset A', 'Asset B'],
        'operator': '>=',
        'right_value': 0.5
    }
    {
        'left_indices': ['Asset C'],
        'operator': '<=',
        'right_index': ['Asset B']
    }
]

weights = max_surplus_return_optimizer(portfolio=portfolio, asset_constraints=constraints, surplus_risk_upper_limit=0.0001)

