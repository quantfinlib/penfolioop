# Copyright (c) 2025 Mohammadjavad Vakili. All rights reserved.

r"""Portfolio optimization and objective functions.

This module provides various objective functions for pension fund portfolio optimization, including:

- `max_surplus_sharpe_ratio_optimizer`: 

This function Maximizes the surplus portfolio return to surplus portfolio risk.

- `surplus_mean_variance_optimizer`: 

Mean-variance optimization for surplus portfolios.

- `max_surplus_return_optimizer`: 

Maximizes surplus return with the option of an upper limit on the surplus variance.

- `min_surplus_variance_optimizer`: 

Minimizes surplus variance with the option of a lower limit on the surplus return.

- `efficient_frontier`: 

Finds the efficient frontier portfolios.


In all these problems, we aim to find the weight vector that optimally allocates assets in the portfolio.
The weight vector is always an array of asset weights plus a liability weight (the last element of the weight vector), 
where the liability weight is always set to -1. 

Let's assume that we have $n_{assets}$ in our portfolio. Therefore, the weight vector 
is a $n_{assets} + 1$ dimensional vector, where the first $n_{assets}$ elements are the asset weights 
and the last element is the liability weight.

$$
\mathbf{w} = \begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_{n_{assets}} \\
w_L
\end{bmatrix} = \begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_{n_{assets}} \\
-1
\end{bmatrix}
$$, 

where \( w_i \) is the weight of asset \( i \) and \( w_L \) is the weight of the liabilities, which is set to -1.

In a similar fashion, we define the expected return vector as an array containing the 
expected returns of the assets and liabilities. This is a \(n_{assets} + 1\) dimensional vector, 
where the first \(n_{assets}\) elements are the expected returns of the assets and the 
last element is the expected return of the liabilities.

$$
\mathbf{R} = \begin{bmatrix}
r_1 \\
r_2 \\
\vdots \\
r_n \\
r_L
\end{bmatrix},
$$

where \( r_i \) is the expected return of asset \( i \) and \( r_L \) is the expected return of the liabilities.

The covariance matrix is defined as the covariance matrix of assets and liability returns. 
This matrix is a \(n_{assets} + 1\) by \(n_{assets} + 1\) square matrix, where the first \(n_{assets}\) 
rows and columns correspond to the assets and the last row and column correspond to the liabilities.

$$
\mathbf{\Sigma} = \begin{bmatrix}
 \Sigma_{A} , \Sigma_{AL} \\
 \Sigma_{AL} , \sigma^{2}_{L}
\end{bmatrix},
$$

where \( \Sigma_{A} \) is a covariance matrix of the assets, 
\( \Sigma_{AL} \) is the covariance between the assets and liabilities, 
and \( \sigma^{2}_{L} \) is the variance of the liabilities. 
\( \Sigma_{A} \) is a \(n_{assets}\) by \(n_{assets}\) square matrix, where each element 
represents the covariance between the returns of two assets.
\( \Sigma_{AL} \) is a \(n_{assets}\) dimensional vector, where each element represents the 
covariance between the returns of an asset and liability return.
\( \sigma^{2}_{L} \) is the variance of the liability return.

With these conventions at hand, we can compute the surplus return (return of the portfolio in excess of liabilities) 
and the surplus variance (variance of the surplus returns) in the following way.

$$
\begin{align*}
\text{Surplus Return} &= \mathbf{W}^{T} \mathbf{R}  = \sum_{i=1}^{n_{assets}} w_{i} r_{i} - r_{L} \\
\text{Surplus Variance} &= \mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W}  = \sum_{i=1}^{n_{assets}} \sum_{j=1}^{n_{assets}} w_{i} w_{j} \big(\Sigma_{A}\big)_{ij} - 2 \sum_{i=1}^{n_{assets}} w_{i} \big(\Sigma_{AL}\big)_{i} + \sigma^{2}_{L}
\end{align*}
$$

For the sake of clarity on the conventions used in this module, we repeat some of these definitions in the documentation of individual functions.

"""
from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from penfolioop.constraints import generate_constraints, generate_scipy_constraints

if TYPE_CHECKING:
    from penfolioop.portfolio import Portfolio


MINISCULE_WEIGHT_THRESHOLD = 1e-6


def _clean_up_weights(weights: np.ndarray) -> np.ndarray:
    """Clean up the weights by ensuring they sum to 1 and the last weight is -1.

    Parameters
    ----------
    weights : np.ndarray
        The weights to clean up.

    Returns
    -------
    np.ndarray
        The cleaned-up weights.
    """
    weights = weights.copy()
    # set small negative weights to zero
    weights[weights < 0] = 0
    # set miniscule weights to zero
    weights[np.abs(weights) < MINISCULE_WEIGHT_THRESHOLD] = 0
    # make sure the asset weights sum to 1
    weights[:-1] /= weights[:-1].sum()
    # make sure the liability weight is -1
    weights[-1] = -1
    return weights


def clean_up_weight_decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Make a decorator to clean up weights after optimization.

    Parameters
    ----------
    func : callable
        The optimization function to decorate.

    Returns
    -------
    callable
        The decorated optimization function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        # Call the original optimization function
        result = func(*args, **kwargs)
        return _clean_up_weights(result)
    return wrapper


def _negative_surplus_sharpe_ratio_objective(
    weights: np.ndarray, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
) -> float:
    """Construct an objective function to maximize the Sharpe ratio of the portfolio surplus.

    Parameters
    ----------
    weights : np.ndarray
        Weights of the assets in the portfolio.
    expected_returns : np.ndarray
        Expected returns of the assets and liabilities.
    covariance_matrix : np.ndarray
        Covariance matrix of the assets and liabilities.

    Returns
    -------
    float
        Negative of the Sharpe ratio (to be minimized).

    """
    surplus_return = weights.T @ expected_returns
    surplus_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    if surplus_variance <= 0:
        return np.inf  # Avoid division by zero or negative variance

    return -surplus_return / np.sqrt(surplus_variance)


@clean_up_weight_decorator
def max_surplus_sharpe_ratio_optimizer(
    portfolio: Portfolio, asset_constraints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    r"""Optimize the asset weights to achieve a target excess return over the expected liabilities return.

    This problem can be formulated as:

    $$
    \underset{\mathbf{W}}{\mathrm{maximize}} \quad  \frac{\mathbf{W}^{T}\mathbf{R}}{\mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W}},
    $$


    $$
    \begin{align*}
    \mathbf{W} &=& \big[  w_{1}, w_{2}, \ldots, w_{n_{assets}}, -1 \big]^{T}, \\
    \mathbf{R} &=& \big[  R_{1}, R_{2}, \ldots, R_{n_{assets}}, R_{L} \big]^{T}, \\
    \end{align*} \\
    $$

    $$
    \mathbf{\Sigma} = \begin{bmatrix}
    \Sigma_{A} & \Sigma_{AL} \\
    \Sigma_{AL} & \sigma^{2}_{L}
    \end{bmatrix},
    $$

    where $\mathbf{W}$ is the vector of assets and liability weights, $\mathbf{R}$ is the vector of expected returns
    for the assets and liabilities, and $\mathbf{\Sigma}$ is the covariance matrix of the assets and liabilities.

    The last element of $\mathbf{W}$ corresponds to the liabilities. The liability weight is always set to -1.


    The optimization is subject to the following constraints:

    $$
    \begin{align*}
    (1) &\quad& \sum_{i=1}^{n_{assets}} w_{i} = 1, \\
    (2) &\quad& w_{i} \geq 0, \quad \forall i \in \{1, \ldots, n_{assets}\}, \\
    (3) &\quad& w_{n_{assets} + 1} = w_{L} = -1
    \end{align*}
    $$

    If the `asset_constraints` parameter is provided by the user, the optimization will include these additional constraints.
    See `penfolioop.constraints` for more details. A valid `asset_constraints` must fullfill a set of properties which are validated
    by the `penfolioop.constraints.AssetConstraint` class. Users are encouraged to consult with the `penfolioop.constraints`
    module and in particular the `penfolioop.constraints.AssetConstraint` class for more information on how to properly define asset constraints.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object containing asset names, covariance matrix, expected returns.
    asset_constraints : list[dict[str, Any]], optional
        Additional constraints for the optimization problem. Default is None.


    Returns
    -------
    np.ndarray
        Optimized asset weights as a numpy array.

    Raises
    ------
    ValueError
        If the optimization fails or constraints are not satisfied.

    """  # noqa: E501
    n_assets = len(portfolio.names) - 1

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w[:n_assets]) - 1},  # weights sum to 1
        {"type": "eq", "fun": lambda w: w[-1] + 1},                 # last weight is -1
    ]
    if asset_constraints:
        constraints += generate_scipy_constraints(
            asset_constraints=asset_constraints, asset_names=portfolio.names,
        )
    # Bounds
    bounds = [(0, 1)] * n_assets + [(None, None)]  # last weight (liability) unbounded
    # Initial guess
    initial_weights = np.ones(n_assets + 1) / (n_assets)
    initial_weights[-1] = -1  # liabilities weight

    # Solve the optimization problem
    result = minimize(
        _negative_surplus_sharpe_ratio_objective,
        initial_weights,
        args=(portfolio.expected_returns, portfolio.covariance_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    if not result.success:
        msg = "Optimization failed."
        raise ValueError(msg)

    return result.x


@clean_up_weight_decorator
def surplus_mean_variance_optimizer(
    portfolio: Portfolio, risk_aversion: float = 1., asset_constraints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    r"""Optimize the asset weights to maximize the surplus return over the expected liabilities return.

    This optimization problem can be formulated as:

    $$
    \underset{\mathbf{W}}{\mathrm{maximize}} \quad  \mathbf{W}^{T}\mathbf{R} -  \frac{\lambda}{2}  \mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W},
    $$

    $$
    \begin{align*}
    \mathbf{W} &=& \big[  w_{1}, w_{2}, \ldots, w_{n_{assets}}, -1 \big]^{T}, \\
    \mathbf{R} &=& \big[  R_{1}, R_{2}, \ldots, R_{n_{assets}}, R_{L} \big]^{T}, \\
    \end{align*} \\
    $$

    $$
    \mathbf{\Sigma} = \begin{bmatrix}
    \Sigma_{A} & \Sigma_{AL} \\
    \Sigma_{AL} & \sigma^{2}_{L}
    \end{bmatrix},
    $$

    where $\mathbf{W}$ is the vector of assets and liability weights, $\mathbf{R}$ is the vector of expected returns
    for the assets and liabilities, $\mathbf{\Sigma}$ is the covariance matrix of the assets and liabilities, and $\lambda$
    is the risk aversion parameter.

    The last element of $\mathbf{W}$ corresponds to the liabilities. The liability weight is always set to -1.


    The optimization is subject to the following constraints:

    $$
    \begin{align*}
    (1) &\quad& \sum_{i=1}^{n_{assets}} w_{i} = 1, \\
    (2) &\quad& w_{i} \geq 0, \quad \forall i \in \{1, \ldots, n_{assets}\}, \\
    (3) &\quad& w_{n_{assets} + 1} = w_{L} = -1
    \end{align*}
    $$

    If the `asset_constraints` parameter is provided by the user, the optimization will include these additional constraints.
    See `penfolioop.constraints` for more details. A valid `asset_constraints` must fullfill a set of properties which are validated
    by the `penfolioop.constraints.AssetConstraint` class. Users are encouraged to consult with the `penfolioop.constraints`
    module and in particular the `penfolioop.constraints.AssetConstraint` class for more information on how to properly define asset constraints.


    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object containing asset names, covariance matrix, expected returns.
    lmbd : float, optional
        Regularization parameter for the optimization problem. Default is 1.0.
    asset_constraints : list[dict[str, Any]], optional
        Additional constraints for the optimization problem. Default is None.

    Returns
    -------
    np.ndarray
        Optimized asset weights as a numpy array.

    Raises
    ------
    ValueError
        If the optimization fails or constraints are not satisfied.

    """  # noqa: E501
    if risk_aversion < 0:
        msg = "Risk aversion must be non-negative."
        raise ValueError(msg)

    n_assets = len(portfolio.names) - 1
    weights = cp.Variable(n_assets + 1)
    # Objective function: maximize the surplus return over the expected liabilities return
    surplus_return = weights.T @ portfolio.expected_returns
    surplus_variance = cp.quad_form(weights, portfolio.covariance_matrix)
    objective = cp.Maximize(surplus_return - risk_aversion / 2 * surplus_variance)
    # Constraints
    constraints = [
        cp.sum(weights[:n_assets]) == 1,  # Weights must sum to 1
        weights[:n_assets] >= 0,          # No short selling
        weights[-1] == -1,                # Last weight is liabilities
    ]
    if asset_constraints:
        constraints += generate_constraints(
            portfolio_weights=weights, asset_constraints=asset_constraints, asset_names=portfolio.names
        )
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None or problem.status != cp.OPTIMAL:
        msg = "Optimization failed."
        raise ValueError(msg)

    return weights.value


@clean_up_weight_decorator
def max_surplus_return_optimizer(
    portfolio: Portfolio,
    asset_constraints: list[dict[str, Any]] | None = None,
    surplus_risk_upper_limit: float | None = None,
) -> np.ndarray:
    r"""Optimize the asset weights to maximize the surplus return over the expected liabilities return.

    The optimization problem can be formulated as:
    $$
    \underset{\mathbf{W}}{\mathrm{maximize}} \quad \mathbf{W}^{T}\mathbf{R},
    $$

    $$
    \begin{align*}
    \mathbf{W} &=& \big[  w_{1}, w_{2}, \ldots, w_{n_{assets}}, -1 \big]^{T}, \\
    \mathbf{R} &=& \big[  R_{1}, R_{2}, \ldots, R_{n_{assets}}, R_{L} \big]^{T}, \\
    \end{align*} \\
    $$

    $$
    \mathbf{\Sigma} = \begin{bmatrix}
    \Sigma_{A} & \Sigma_{AL} \\
    \Sigma_{AL} & \sigma^{2}_{L}
    \end{bmatrix},
    $$

    where $\mathbf{W}$ is the vector of assets and liability weights, $\mathbf{R}$ is the vector of expected returns
    for the assets and liabilities, and $\mathbf{\Sigma}$ is the covariance matrix of the assets and liabilities.

    The last element of $\mathbf{W}$ corresponds to the liabilities. The liability weight is always set to -1.


    The optimization is subject to the following constraints:

    $$
    \begin{align*}
    (1) &\quad& \sum_{i=1}^{n_{assets}} w_{i} = 1, \\
    (2) &\quad& w_{i} \geq 0, \quad \forall i \in \{1, \ldots, n_{assets}\}, \\
    (3) &\quad& w_{n_{assets} + 1} = w_{L} = -1
    \end{align*}
    $$

    Additionally, if the parameter `surplus_risk_upper_limit` is provided by the user, we will add a surplus risk upper limit
    constraint to the optimization problem:

    $$
    \mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W} \leq \sigma^2
    $$, where $\sigma$ is the surplus risk upper limit.


    If the `asset_constraints` parameter is provided by the user, the optimization will include these additional
    constraints. See `penfolioop.constraints` for more details. A valid `asset_constraints` must fulfill a set
    of properties which are validated by the `penfolioop.constraints.AssetConstraint` class. Users are encouraged
    to consult with the `penfolioop.constraints` module and in particular the `penfolioop.constraints.AssetConstraint`
    class for more information on how to properly define asset constraints.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object containing asset names, covariance matrix, expected returns.
    asset_constraints : list[dict[str, Any]], optional
        Additional constraints for the optimization problem. Default is None.
    surplus_risk_upper_limit : float, optional
        The surplus risk upper limit for the optimization problem. Default is None.

    Returns
    -------
    np.ndarray
        Optimized asset weights as a numpy array.

    Raises
    ------
    ValueError
        If the optimization fails or constraints are not satisfied.

    """
    n_assets = len(portfolio.names) - 1
    weights = cp.Variable(n_assets + 1)

    # Objective function: maximize the surplus return over the expected liabilities return
    surplus_return = weights.T @ portfolio.expected_returns
    objective = cp.Maximize(surplus_return)
    # Constraints
    constraints = [
        cp.sum(weights[:n_assets]) == 1,  # Weights must sum to 1
        weights[:n_assets] >= 0,          # No short selling
        weights[-1] == -1,                # Last weight is liabilities
    ]
    # Apply asset constraints if provided by user
    if asset_constraints:
        constraints += generate_constraints(
            portfolio_weights=weights, asset_constraints=asset_constraints, asset_names=portfolio.names
        )
    # Surplus risk upper limit constraint if provided by user
    if surplus_risk_upper_limit is not None:
        constraints.append(cp.quad_form(weights, portfolio.covariance_matrix) <= surplus_risk_upper_limit ** 2.)
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None or problem.status != cp.OPTIMAL:
        msg = "Optimization failed."
        raise ValueError(msg)

    return weights.value


@clean_up_weight_decorator
def min_surplus_variance_optimizer(
    portfolio: Portfolio, 
    asset_constraints: list[dict[str, Any]] | None = None,
    surplus_return_lower_limit: float | None = None,
) -> np.ndarray:
    r"""Optimize the asset weights to minimize the surplus variance of the portfolio.

    This optimization problem can be formulated as:

    $$
    \underset{\mathbf{W}}{\mathrm{minimize}} \quad \mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W},
    $$

    $$
    \begin{align*}
    \mathbf{W} &=& \big[  w_{1}, w_{2}, \ldots, w_{n_{assets}}, -1 \big]^{T}, \\
    \mathbf{R} &=& \big[  R_{1}, R_{2}, \ldots, R_{n_{assets}}, R_{L} \big]^{T}, \\
    \end{align*} \\
    $$

    $$
    \mathbf{\Sigma} = \begin{bmatrix}
    \Sigma_{A} & \Sigma_{AL} \\
    \Sigma_{AL} & \sigma^{2}_{L}
    \end{bmatrix},
    $$

    where $\mathbf{W}$ is the vector of assets and liability weights, $\mathbf{R}$ is the vector of expected returns
    for the assets and liabilities, and $\mathbf{\Sigma}$ is the covariance matrix of the assets and liabilities.

    The last element of $\mathbf{W}$ corresponds to the liabilities. The liability weight is always set to -1.

    The optimization is subject to the following general constraints:

    $$
    \begin{align*}
    (1) &\quad& \sum_{i=1}^{n_{assets}} w_{i} = 1, \\
    (2) &\quad& w_{i} \geq 0, \quad \forall i \in \{1, \ldots, n_{assets}\}, \\
    (3) &\quad& w_{n_{assets} + 1} = w_{L} = -1
    \end{align*}
    $$

    Additionally, if the parameter `surplus_return_lower_limit` is provided by the user, 
    we will add a surplus return lower limit constraint to the optimization problem:

    $$
    \mathbf{W}^{T} \mathbf{R} \geq \tilde{R}
    $$,

    where $\tilde{R}$ is the surplus return lower limit.


    If the `asset_constraints` parameter is provided by the user, the optimization will include these
    additional constraints. See `penfolioop.constraints` for more details. A valid `asset_constraints`
    must fullfill a set of properties which are validated by the `penfolioop.constraints.AssetConstraint` class.
    Users are encouraged to consult with the `penfolioop.constraints` module and in particular the 
    `penfolioop.constraints.AssetConstraint` class for more information on how to properly define asset constraints.


    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object containing asset names, covariance matrix, expected returns.
    asset_constraints : list[dict[str, Any]], optional
        Additional constraints for the optimization problem. Default is None.
    surplus_return_lower_limit : float, optional
        The surplus return lower limit for the optimization problem. Default is None.

    Returns
    -------
    np.ndarray
        Optimized asset weights as a numpy array.

    Raises
    ------
    ValueError
        If the optimization fails or constraints are not satisfied.

    """
    n_assets = len(portfolio.names) - 1
    weights = cp.Variable(n_assets + 1)

    # Objective function: minimize the surplus variance of the portfolio
    surplus_variance = cp.quad_form(weights, portfolio.covariance_matrix)
    objective = cp.Minimize(surplus_variance)
    # Constraints
    constraints = [
        cp.sum(weights[:n_assets]) == 1,  # Weights must sum to 1
        weights[:n_assets] >= 0,          # No short selling
        weights[-1] == -1,                # Last weight is liabilities
    ]
    # Apply asset constraints if provided by user
    if asset_constraints:
        constraints += generate_constraints(
            portfolio_weights=weights, asset_constraints=asset_constraints, asset_names=portfolio.names
        )
    # Apply surplus return lower limit constraint if provided by user
    if surplus_return_lower_limit is not None:
        constraints.append(weights.T @ portfolio.expected_returns >= surplus_return_lower_limit)
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None or problem.status != cp.OPTIMAL:
        msg = "Optimization failed"
        raise ValueError(msg)

    return weights.value


def efficient_frontier(
    portfolio: Portfolio,
    asset_constraints: list[dict[str, Any]] | None = None,
    surplus_return_range: tuple[float, float] = (0, 1),
) -> dict[str, np.ndarray]:
    r"""Find the efficient frontier of the portfolio.

    This function calculates the weights of the following optimization problem by
    varying the surplus return lower limit $\tilde{R}$.

    $$
    \underset{\mathbf{W}}{\text{minimize}} \quad \mathbf{W}^{T} \mathbf{C} \mathbf{W}
    $$
    subject to

    $$
    \mathbf{W}^{T} \mathbf{R} \geq \tilde{R}.
    $$

    By varying the surplus return lower limit, we get a different set of weights (different portfolios).
    The set of all these optimal portfolios forms the efficient frontier.

    Note that

    $$
    \begin{align*}
    \mathbf{W} &=& \big[  w_{1}, w_{2}, \ldots, w_{n_{assets}}, -1 \big]^{T}, \\
    \mathbf{R} &=& \big[  R_{1}, R_{2}, \ldots, R_{n_{assets}}, R_{L} \big]^{T}, \\
    \end{align*} \\
    $$

    $$
    \mathbf{\Sigma} = \begin{bmatrix}
    \Sigma_{A} & \Sigma_{AL} \\
    \Sigma_{AL} & \sigma^{2}_{L}
    \end{bmatrix},
    $$

    where $\mathbf{W}$ is the vector of assets and liability weights, $\mathbf{R}$ is the vector of expected returns
    for the assets and liabilities, and $\mathbf{\Sigma}$ is the covariance matrix of the assets and liabilities.
    The last element of $\mathbf{W}$ corresponds to the liabilities. The liability weight is always set to -1.

    As always, the following general constraints apply to the weights:

    $$
    \begin{align*}
    (1) &\quad& \sum_{i=1}^{n_{assets}} w_{i} = 1, \\
    (2) &\quad& w_{i} \geq 0, \quad \forall i \in \{1, \ldots, n_{assets}\}, \\
    (3) &\quad& w_{n_{assets} + 1} = w_{L} = -1
    \end{align*}
    $$

    If additional asset constraints are provided, they will be incorporated into the optimization problem.
    See `penfolioop.constraints` for more details.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object containing asset names, covariance matrix, expected returns.
    asset_constraints : list[dict[str, Any]], optional
        Additional constraints for the optimization problem. Default is None.
    surplus_return_range : tuple[float, float], optional
        Range of surplus return values to consider for the optimization. Default is (0, 1).

    Returns
    -------
    dict
        Dictionary containing arrays of weights, surplus returns, and surplus variances.
    """
    target_returns = np.linspace(surplus_return_range[0], surplus_return_range[1], 100)
    weights_placeholder = []
    surplus_return_place_holder = []
    surplus_variance_place_holder = []
    for target_return in target_returns:
        weights = min_surplus_variance_optimizer(
            portfolio=portfolio,
            asset_constraints=asset_constraints,
            surplus_return_lower_limit=target_return,
        )
        weights_placeholder.append(weights)
        surplus_return_place_holder.append(portfolio.surplus_return(weights))
        surplus_variance_place_holder.append(portfolio.surplus_variance(weights))

    return {
        "weights": np.array(weights_placeholder),
        "surplus_returns": np.array(surplus_return_place_holder),
        "surplus_variances": np.array(surplus_variance_place_holder)
    }
