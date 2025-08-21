# Copyright (c) 2025 Mohammadjavad Vakili. All rights reserved.

"""Portfolio optimization algorithms and utilities.

This module provides various optimization functions for pension fund portfolio optimization, including:
- max_surplus_sharp_ratio_optimizer: Maximizes the surplus portfolio return to surplus portfolio risk
- surplus_mean_variance_optimizer: Mean-variance optimization for surplus portfolios
- max_surplus_return_optimizer: Maximizes surplus return
- min_surplus_variance_optimizer: Minimizes surplus variance
- efficient_frontier: Find the efficient frontier
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from penfolioop.constraints import generate_constraints, generate_scipy_constraints

if TYPE_CHECKING:
    from penfolioop.portfolio import Portfolio


def _negative_surplus_sharp_ratio_objective(
        weights: np.ndarray, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
    ) -> float:
    """Objective function to maximize the Sharpe ratio of the portfolio surplus.

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


def max_surplus_sharp_ratio_optimizer(
    portfolio: Portfolio, asset_constraints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Optimize the asset weights to achieve a target excess return over the expected liabilities return.

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

    """
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
        _negative_surplus_sharp_ratio_objective,
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


def surplus_mean_variance_optimizer(
    portfolio: Portfolio, lmbd: float = 1., asset_constraints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Optimize the asset weights to maximize the surplus return over the expected liabilities return.

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

    """
    n_assets = len(portfolio.names) - 1
    weights = cp.Variable(n_assets + 1)

    # Objective function: maximize the surplus return over the expected liabilities return
    surplus_return = weights.T @ portfolio.expected_returns
    surplus_variance = cp.quad_form(weights, portfolio.covariance_matrix)
    objective = cp.Maximize(surplus_return - lmbd / 2 * surplus_variance)
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


def max_surplus_return_optimizer(
    portfolio: Portfolio, asset_constraints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Optimize the asset weights to maximize the surplus return over the expected liabilities return.

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


def min_surplus_variance_optimizer(portfolio: Portfolio, asset_constraints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Optimize the asset weights to minimize the surplus variance of the portfolio.

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
    if asset_constraints:
        constraints += generate_constraints(
            portfolio_weights=weights, asset_constraints=asset_constraints, asset_names=portfolio.names
        )
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None or problem.status != cp.OPTIMAL:
        msg = "Optimization failed"
        raise ValueError(msg)

    return weights.value


def _validate_lmbd_range(lmbd_range: tuple[float, float]) -> None:
    """Validate the lambda range for the efficient frontier optimization.

    Parameters
    ----------
    lmbd_range : tuple[float, float]
        The range of lambda values to consider for the optimization.

    Raises
    ------
    ValueError
        If the lambda range is invalid (e.g., lower bound is greater than upper bound).

    """
    if lmbd_range[0] >= lmbd_range[1]:
        msg = "Invalid lambda range: lower bound must be less than upper bound."
        raise ValueError(msg)


def efficient_frontier(
    portfolio: Portfolio,
    num_points: int = 100,
    asset_constraints: list[dict[str, Any]] | None = None,
    lmbd_range: tuple[float, float] = (0, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the efficient frontier of the portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object containing asset names, covariance matrix, expected returns.
    num_points : int, optional
        Number of points to calculate on the efficient frontier. Default is 100.
    asset_constraints : list[dict[str, Any]], optional
        Additional constraints for the optimization problem. Default is None.
    lmbd_range : tuple[float, float], optional
        Range of lambda values to consider for the optimization. Default is (0, 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing arrays of surplus returns and surplus variances.

    """
    if lmbd_range is None:
        lmbd_range = (0, 1)
    else:
        _validate_lmbd_range(lmbd_range)
    lmbds = np.linspace(lmbd_range[0], lmbd_range[1], num_points)
    surplus_returns = []
    surplus_variances = []
    weights_placeholder = []

    def _optimize_single_lambda(lmbd) -> np.ndarray | None:  # noqa: ANN001
        """Optimize for a single lambda value."""  # noqa: DOC201
        try:
            return surplus_mean_variance_optimizer(
                portfolio=portfolio, lmbd=lmbd, asset_constraints=asset_constraints
            )
        except ValueError:
            return None

    for lmbd in lmbds:
        weights = _optimize_single_lambda(lmbd)
        if weights is not None:
            weights_placeholder.append(weights)
            surplus_returns.append(portfolio.surplus_return(weights))
            surplus_variances.append(portfolio.surplus_variance(weights))
        else:
            # If optimization fails for a particular lambda, append NaN
            weights_placeholder.append(np.nan * np.ones(len(portfolio.names)))
            surplus_returns.append(np.nan)
            surplus_variances.append(np.nan)
    return np.array(weights_placeholder), np.array(surplus_returns), np.array(surplus_variances)
