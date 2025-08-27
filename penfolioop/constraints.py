# Copyright (c) 2025 Mohammadjavad Vakili. All rights reserved.

"""Portfolio constraint generation module.

This module provides functionality for generating optimization constraints
for portfolio management, including asset weight constraints and other
portfolio-specific limitations.


User Constraints
----------------

Users of `penfolioop` can define additional constraints on the asset classes in their portfolio.

A valid constraints is a list of dictionaries, where each dictionary represents a constraint.

Each constraint is a python dictionary and must include the following two keys:

- `left_indices`: A list of asset names on the left-hand side of the constraint.
- `operator`: The comparison operator for the constraint (e.g., "==", "<=", ">=").

Additionally, each dictionary must include ***one and only one*** of the following two keys:

- `right_value`: A numeric value for the right-hand side of the constraint (optional).
- `right_indices`: A list of asset names on the right-hand side of the constraint (optional).

Here are some valid constraints:

```python
[
    {
        "left_indices": ["asset_1", "asset_2"],
        "operator": ">=",
        "right_value": 0.5
    },
    {
        "left_indices": ["asset_3"],
        "operator": "==",
        "right_indices": ["asset_4"]
    }
]
```

In this case, the constraints specify that the combined weight of `asset_1` and `asset_2`
must be at least 0.5, while the weight of `asset_3` must be equal to the weight of `asset_4`.

or 

```python
[
    {
        "left_indices": ["asset_1", "asset_2", "asset_6"],
        "operator": "<=",
        "right_value": ["asset_4"]
    },
    {
        "left_indices": ["asset_3", "asset_5"],
        "operator": ">=",
        "right_value": 0.1
    }
]
```

In this case, the constraints specify that the combined weight of `asset_1`, `asset_2`,
and `asset_6` must be less than or equal to the weight of `asset_4`,
while the combined weight of `asset_3` and `asset_5` must be greater than or equal to 0.1.

When the value corresponding to the `left_indices` is a list of assets,
the constraint is applied to the combined weight of those assets.
When the value of `left_indices` is a single asset, the constraint is applied to that asset's weight.

The only allowed values for `operator` are:

- `==`: Equal to
- `<=`: Less than or equal to
- `>=`: Greater than or equal to

The permitted values for `right_value` and `right_indices` are as follows:

- `right_value`: A numeric value which constrains the combined weight of the assets in `left_indices`.
- `right_indices`: A list of asset names whose combined weight is used for the constraint.


Module content
--------------


- `AssetConstraint`

This class is used to validate the constraints defined by the user.

 - `generate_constraints` 

 Converts the user defined constraints into CVXPY constraints.

 - `generate_scipy_constraints`

 Converts the user defined constraints into constraints that can be used by
 `scipy.optimize.minimize` function.

"""

from __future__ import annotations

from typing import Any, Literal, Self

import cvxpy as cp
import numpy as np
from pydantic import BaseModel, ValidationError, field_validator, model_validator


class AssetConstraint(BaseModel):
    """Model for validating individual asset constraints."""

    left_indices: list[str]
    operator: Literal["==", "<=", ">=", "<", ">"]
    right_value: float | None = None
    right_indices: list[str] | None = None

    # Validator for left_indices
    @field_validator("left_indices")
    @classmethod
    def validate_left_indices(cls, v: list[str]) -> list[str]:
        """Ensure left_indices is not empty.

        Parameters
        ----------
        v : list[str]
            List of asset names for the left-hand side of the constraint.

        Returns
        -------
        list[str]
            Validated list of asset names.

        Raises
        ------
        ValueError
            If left_indices is empty or contains duplicate names.

        """
        if not v:
            msg = "left_indices must not be empty"
            raise ValueError(msg)
        if len(v) != len(set(v)):
            msg = "Asset names in left_indices must be unique"
            raise ValueError(msg)
        return v

    # Validator for the operator
    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: Literal["==", "<=", ">="]) -> Literal["==", "<=", ">="]:
        """Ensure operator is one of the allowed values.

        Parameters
        ----------
        v : Literal["==", "<=", ">="]
            The operator to validate.

        Returns
        -------
        Literal["==", "<=", ">="]
            Validated operator.

        Raises
        ------
        ValueError
            If the operator is not one of the allowed values.

        """
        allowed_operators = {"==", "<=", ">="}
        if v not in allowed_operators:
            msg = f"Operator must be one of {allowed_operators}, got {v}"
            raise ValueError(msg)
        return v

    # Validator for the right_indices in the right side of the constraint
    @field_validator("right_indices")
    @classmethod
    def validate_right_indices(cls, v: list[str] | None) -> list[str] | None:
        """Ensure right_indices is not empty and contains unique names.

        Parameters
        ----------
        v : list[str] | None
            List of asset names for the right-hand side of the constraint.

        Returns
        -------
        list[str] | None
            Validated list of asset names or None if not provided.

        Raises
        ------
        ValueError
            If right_indices is not None and is empty or contains duplicate names.


        """
        if v is not None and not v:
            msg = "right_indices must not be empty if provided"
            raise ValueError(msg)
        if v is not None and len(v) != len(set(v)):
            msg = "Asset names in right_indices must be unique"
            raise ValueError(msg)
        return v

    # Validator for the right_value in the right side of the constraint
    @field_validator("right_value")
    @classmethod
    def validate_right_value(cls, v: float | None) -> float | None:
        """Ensure right_value is a non-negative number if provided.

        Parameters
        ----------
        v : float | None
            The right-hand side value of the constraint.

        Returns
        -------
        float | None
            Validated right value or None if not provided.

        Raises
        ------
        ValueError
            If right_value is not None and is not a number or is negative.

        """
        if v is not None and not isinstance(v, int | float):
            msg = "right_value must be a number if provided"
            raise ValueError(msg)
        if v is not None and (v < 0 or v > 1):
            msg = "right_value must be between 0 and 1 if provided"
            raise ValueError(msg)
        return float(v) if v is not None else None

    # Validator to ensure either right_value or right_indices is provided, but not both
    @model_validator(mode="after")
    def validate_right_side(self) -> Self:
        """Ensure that either right_value or right_indices is provided, but not both.

        Returns
        -------
        Self
            The validated AssetConstraint instance.

        Raises
        ------
        ValueError
            If neither right_value nor right_indices is provided, or if both are provided.

        """
        right_value = self.right_value
        right_indices = self.right_indices
        if right_value is None and right_indices is None:
            msg = "Either right_value or right_indices must be provided"
            raise ValueError(msg)
        if right_value is not None and right_indices is not None:
            msg = "Only one of right_value or right_indices can be provided"
            raise ValueError(msg)
        return self


def _check_constraints(constraints: list[dict[str, Any]]) -> None:
    """Check if the constraints are valid.

    Parameters
    ----------
    constraints : list[dict[str, Any]]
        The asset constraints to validate.

    Raises
    ------
    ValueError
        If the constraint is invalid.

    """
    try:
        for constraint in constraints:
            AssetConstraint(**constraint)
    except ValidationError as e:
        msg = f"Invalid constraint: {e}"
        raise ValueError(msg) from e


def _process_left_side_of_constraint(
    portfolio_weights: cp.Variable,
    left_indices: list[str],
    asset_indices: dict[str, int],
) -> cp.Expression:
    """Process the left-hand side of the constraint.

    Parameters
    ----------
    portfolio_weights : cp.Variable
        The variable representing the portfolio weights.
    left_indices : list[str]
        List of asset names for the left-hand side of the constraint.
    asset_indices : dict[str, int]
        Dictionary mapping asset names to their indices in the portfolio weights.

    Returns
    -------
    cp.Expression
        The left-hand side expression of the constraint.

    """
    left_vec = np.zeros(len(asset_indices))
    for idx in left_indices:
        left_vec[asset_indices[idx]] = 1
    return left_vec @ portfolio_weights


def _process_right_side_of_constraint(
    portfolio_weights: cp.Variable,
    asset_indices: dict[str, int],
    right_value: float | None = None,
    right_indices: list[str] | None = None,
) -> cp.Expression | float:
    """Process the right-hand side of the constraint.

    Parameters
    ----------
    portfolio_weights : cp.Variable
        The variable representing the portfolio weights.
    asset_indices : dict[str, int]
        Dictionary mapping asset names to their indices in the portfolio weights.
    right_value : float | None, optional
        The right-hand side value of the constraint.
    right_indices : list[str] | None, optional
        List of asset names for the right-hand side of the constraint.

    Returns
    -------
    cp.Expression | float
        The right-hand side expression of the constraint or a float if right_value is provided.

    Raises
    ------
    ValueError
        If neither right_value nor right_indices is provided, or if both are provided.

    """
    if right_value is not None:
        return right_value
    if right_indices is not None:
        right_vec = np.zeros(len(asset_indices))
        for idx in right_indices:
            right_vec[asset_indices[idx]] = 1
        return right_vec @ portfolio_weights
    msg = "Constraint must have either 'right_value' or 'right_indices'."
    raise ValueError(msg)


def _process_operator(
    operator: Literal["==", "<=", ">="],
    left_expr: cp.Expression,
    right_expr: cp.Expression | float,
) -> cp.Constraint:
    """Process the operator and return the corresponding constraint.

    Parameters
    ----------
    operator : Literal["==", "<=", ">="]
        The operator to apply to the left and right expressions.

    left_expr : cp.Expression
        The left-hand side expression of the constraint.

    right_expr : cp.Expression | float
        The right-hand side expression of the constraint or a float if right_value is provided.

    Returns
    -------
    cp.Constraint
        The constraint corresponding to the operator applied to the left and right expressions.

    Raises
    ------
    ValueError
        If the operator is not one of the allowed values.

    """
    if operator == "==":
        return left_expr == right_expr
    if operator == "<=":
        return left_expr <= right_expr
    if operator == ">=":
        return left_expr >= right_expr
    msg = f"Unknown operator: {operator}"
    raise ValueError(msg)


def generate_constraints(
    portfolio_weights: cp.Variable,
    asset_names: list[str],
    asset_constraints: list[dict[str, Any]],
) -> list[cp.Constraint]:
    """Generate constraints for the portfolio optimization problem.

    Parameters
    ----------
    portfolio_weights : cp.Variable
        The variable representing the portfolio weights.
    asset_names : list[str]
        List of asset names in the portfolio.
    asset_constraints : list[dict[str, Any]]
        List of asset constraints to apply to the portfolio weights.

    Returns
    -------
    list[cp.Constraint]
        List of cvxpy constraints generated from the asset constraints.

    """
    constraints: list[cp.Constraint] = []

    _check_constraints(constraints=asset_constraints)
    asset_indices: dict[str, int] = {name: i for i, name in enumerate(asset_names)}
    for constraint in asset_constraints:
        # Process left-hand-side of the constraint
        left_expr = _process_left_side_of_constraint(
            portfolio_weights=portfolio_weights,
            left_indices=constraint["left_indices"],
            asset_indices=asset_indices,
        )
        # process right-hand-side of the constraint
        right_expr: cp.Expression | float = _process_right_side_of_constraint(
            portfolio_weights=portfolio_weights,
            right_value=constraint.get("right_value"),
            right_indices=constraint.get("right_indices"),
            asset_indices=asset_indices,
        )
        # Process the operator
        cp_constraint: cp.Constraint = _process_operator(
            operator=constraint["operator"],
            left_expr=left_expr,
            right_expr=right_expr,
        )
        constraints.append(cp_constraint)

    return constraints


def generate_scipy_constraints(
    asset_constraints: list[dict[str, Any]],
    asset_names: list[str],
) -> list[dict[str, Any]]:
    """Generate constraints for scipy optimization.

    Parameters
    ----------
    asset_constraints : list[dict[str, Any]]
        List of asset constraints to apply to the portfolio weights.
    asset_names : list[str]
        List of asset names in the portfolio.

    Returns
    -------
    list[dict[str, Any]]
        List of constraints formatted for scipy optimization.

    """
    _check_constraints(constraints=asset_constraints)
    asset_indices: dict[str, int] = {name: i for i, name in enumerate(asset_names)}
    scipy_constraints = []

    for constraint in asset_constraints:
        def constraint_fun(x, constraint=constraint):  # noqa: ANN001, ANN202
            # Calculate left side
            left_value = sum(x[asset_indices[idx]] for idx in constraint["left_indices"])

            # Calculate right side
            if constraint.get("right_value") is not None:
                right_value = constraint["right_value"]
            else:
                right_value = sum(x[asset_indices[idx]] for idx in constraint["right_indices"])

            # Return constraint value based on operator
            if constraint["operator"] == "<=":
                return right_value - left_value
            if constraint["operator"] == ">=":
                return left_value - right_value
            if constraint["operator"] == "==":
                return left_value - right_value
            msg = f"Unsupported operator: {constraint['operator']}"
            raise ValueError(msg)

        constraint_type = "ineq" if constraint["operator"] in {"<=", ">="} else "eq"
        scipy_constraints.append({
            "type": constraint_type,
            "fun": constraint_fun,
        })

    return scipy_constraints
