from penfolioop.portfolio import Portfolio
from penfolioop.optimizers import (
    min_surplus_variance_optimizer,
    surplus_mean_variance_optimizer,
    max_surplus_return_optimizer,
    max_surplus_sharp_ratio_optimizer,
    efficient_frontier
)


import numpy as np

import pytest 


@pytest.fixture
def expected_returns():
    return np.array([0.05, 0.07, 0.06, 0.04])

@pytest.fixture
def covariance_matrix():
    return np.array([[0.0004, 0.0002, 0.0001, 0.0003],
                     [0.0002, 0.0005, 0.0003, 0.0004],
                     [0.0001, 0.0003, 0.0006, 0.0002],
                     [0.0003, 0.0004, 0.0002, 0.0007]])

@pytest.fixture
def names():
    return ["Asset A", "Asset B", "Asset C", "Liability"]


@pytest.fixture
def portfolio(expected_returns, covariance_matrix, names):
    return Portfolio(
        names=names,
        covariance_matrix=covariance_matrix,
        expected_returns=expected_returns
    )


@pytest.fixture
def asset_constraints1():
    return [
        {
            'left_indices': ['Asset A', 'Asset B'],
            'operator': '>=',
            'right_value': 0.1
        }
    ]


@pytest.fixture
def asset_constraints2():
    return [
        {
            'left_indices': ['Asset C'],
            'operator': '<=',
            'right_value': 0.5
        }
    ]


@pytest.fixture
def invalid_asset_constraints():
    return [
        {
            'left_indices': ['Asset A', 'Asset B'],
            'operator': '!=',
            'right_value': 0.1
        },
        {
            'left_indices': ['Asset C'],
            'operator': '<=',
            'right': 0.5
        },
        {
            'left_indices': ['Asset D'],  # Non-existent asset
            'operator': '<=',
            'right_value': 0.2
        }
    ]



def generic_weight_requirements(weights: np.ndarray, expected_length: int):
    """Check if the weights meet the basic requirements."""
    if not isinstance(weights, np.ndarray):
        raise ValueError("Weights must be a numpy array.")
    if weights.ndim != 1:
        raise ValueError("Weights must be a one-dimensional array.")
    if weights.shape[0] != expected_length:
        raise ValueError(f"Weights must have length {expected_length}.")
    if not np.isclose(np.sum(weights[:-1]), 1):
        raise ValueError("Asset weights must sum to 1.")
    if not np.isclose(weights[-1], -1):
        raise ValueError("Last weight must be -1 (for liabilities).")
    if np.any(weights[:-1] < 0):
        raise ValueError("Asset weights must be non-negative.")
    if np.any(weights[:-1] > 1):
        raise ValueError("Asset weights must not exceed 1.")


@pytest.mark.parametrize("optimizer", [
    min_surplus_variance_optimizer,
    surplus_mean_variance_optimizer,
    max_surplus_return_optimizer,
    max_surplus_sharp_ratio_optimizer])
def test_optimizer(portfolio, optimizer, asset_constraints1, asset_constraints2):
    
    # Test with asset constraints 1
    weights = optimizer(portfolio, asset_constraints=asset_constraints1)
    generic_weight_requirements(weights, len(portfolio.names))
    assert weights[0] + weights[1] >= 0.1, "Asset A and Asset B weights should be at least 0.1 combined according to constraints"

    # Test with asset constraints 2
    weights = optimizer(portfolio, asset_constraints=asset_constraints2)
    generic_weight_requirements(weights, len(portfolio.names))
    assert weights[2] <= 0.5, "Asset C weight should be at most 0.5 according to constraints"

    # Test with no asset constraints
    weights = optimizer(portfolio)
    generic_weight_requirements(weights, len(portfolio.names))

    # Test with combined asset constraints
    combined_constraints = asset_constraints1 + asset_constraints2
    weights_combined = optimizer(portfolio, asset_constraints=combined_constraints)
    generic_weight_requirements(weights_combined, len(portfolio.names))
    assert weights_combined[0] + weights_combined[1] >= 0.1, "Asset A and Asset B weights should be at least 0.1 combined according to constraints"
    assert weights_combined[2] <= 0.5, "Asset C weight should be at most 0.5 according to constraints"

    with pytest.raises(ValueError):
        optimizer(portfolio, asset_constraints=invalid_asset_constraints)

    # Test with no asset constraints
    weights_no_constraints = optimizer(portfolio)
    assert isinstance(weights_no_constraints, np.ndarray)
    assert weights_no_constraints.shape[0] == len(portfolio.names)

    # Test with multiple asset constraints
    weights_multiple_constraints = optimizer(portfolio, asset_constraints=asset_constraints2)
    assert isinstance(weights_multiple_constraints, np.ndarray)
    assert weights_multiple_constraints.shape[0] == len(portfolio.names)


@pytest.mark.parametrize('lmbd', [0, 0.1, 1.0, 10.0])
# Test for lambda in mean-variance optimizer
def test_mean_variance_optimizer_lambda(portfolio, lmbd):
    weights = surplus_mean_variance_optimizer(portfolio, lmbd=lmbd)
    generic_weight_requirements(weights, len(portfolio.names))

def test_mean_variance_variance_properties(portfolio):
    lmbds = np.linspace(0, 1, 100)
    vars = []
    rets = []
    for lmbd in lmbds:
        weights = surplus_mean_variance_optimizer(portfolio, lmbd=lmbd)
        variance = portfolio.surplus_variance(weights)
        vars.append(variance)
        return_ = portfolio.surplus_return(weights)
        rets.append(return_)
    vars = np.array(vars)
    rets = np.array(rets)
    assert np.all(vars >= 0), "Surplus variance should be non-negative for all lambda values"
    assert np.all(vars[1:] >= vars[:-1]), "Surplus variance should be non-decreasing with increasing lambda"
    assert np.all(rets[1:] <= rets[:-1]), "Surplus return should be non-increasing with increasing lambda"


def test_efficient_frontier(portfolio):
    num_points = 100
    ws, srs, svs = efficient_frontier(portfolio, num_points=num_points)

    assert isinstance(ws, np.ndarray)
    assert isinstance(srs, np.ndarray)
    assert isinstance(svs, np.ndarray)

    assert ws.shape[0] == num_points
    assert srs.shape[0] == num_points
    assert svs.shape[0] == num_points

    for w in ws:
        generic_weight_requirements(w, len(portfolio.names))

    # Check if surplus returns and variances are calculated correctly
    for i in range(num_points):
        assert np.isclose(srs[i], portfolio.surplus_return(ws[i]))
        assert np.isclose(svs[i], portfolio.surplus_variance(ws[i]))