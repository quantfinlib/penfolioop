from penfolioop.constraints import AssetConstraint, _check_constraints, _process_left_side_of_constraint, _process_right_side_of_constraint, generate_constraints, generate_scipy_constraints

import numpy as np 

import pytest 

import cvxpy as cp


VALID_OPERATORS = ['==', '>=', '<=', '>', '<']
INVALID_OPERATORS = ['!=', '==>', '>=<', '>>', '<<', 'gt', 'lt', 'eq', 'geq', 'leq']
OPERATORS = VALID_OPERATORS + INVALID_OPERATORS


@pytest.mark.parametrize("operator", OPERATORS)
def test_asset_constraint_operator(operator):

    # A simple valid constraint if operator is valid
    constraint_1 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': operator,
        'right_indices': ['eq']
    }
    if operator in VALID_OPERATORS:
        constraint = AssetConstraint(**constraint_1)
        assert constraint.left_indices == ['emdc', 'emdh']
        assert constraint.operator == operator
        assert constraint.right_indices == ['eq']
    else:
        with pytest.raises(ValueError):
            AssetConstraint(**constraint_1)

    # A simple valid constraint with right value instead of right index 

    constraint_2 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': operator,
        'right_value': 0.5
    }

    if operator in VALID_OPERATORS:
        constraint = AssetConstraint(**constraint_2)
        assert constraint.left_indices == ['emdc', 'emdh']
        assert constraint.operator == operator
        assert constraint.right_value == 0.5
    else:
        with pytest.raises(ValueError):
            AssetConstraint(**constraint_2)

    # A simple invalid constraint with both right indices and right value
    constraint_3 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': operator,
        'right_indices': ['eq'],
        'right_value': 0.5
    }
    with pytest.raises(ValueError):
            AssetConstraint(**constraint_3)
    return None


VALID_LEFT_SIDES = [['emdc', 'emdh'],['eq']]
INVALID_LEFT_SIDES = [0.5, 'emdc', '==', ['eq', 'eq']]
@pytest.mark.parametrize("left_side, operator", [(ls, op) for ls in VALID_LEFT_SIDES + INVALID_LEFT_SIDES for op in VALID_OPERATORS])
def test_asset_constraint_left_side(left_side, operator):
    # A simple valid constraint if left side is valid
    constraint_1 = {
        'left_indices': left_side,
        'operator': operator,
        'right_indices': ['eq']
    }
    if left_side in VALID_LEFT_SIDES:
        constraint = AssetConstraint(**constraint_1)
        assert constraint.left_indices == left_side
        assert constraint.operator == operator
        assert constraint.right_indices == ['eq']
    else:
        with pytest.raises(ValueError):
            AssetConstraint(**constraint_1)
    return None


def test_asset_constraint_right_side():
    # A simple valid constraint with right value
    constraint_1 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_value': 0.5
    }
    constraint = AssetConstraint(**constraint_1)
    assert constraint.left_indices == ['emdc', 'emdh']
    assert constraint.operator == '=='
    assert constraint.right_value == 0.5

    # A simple valid constraint with right indices
    constraint_2 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_indices': ['eq']
    }
    constraint = AssetConstraint(**constraint_2)
    assert constraint.left_indices == ['emdc', 'emdh']
    assert constraint.operator == '=='
    assert constraint.right_indices == ['eq']

    # A simple invalid constraint with both right indices and right value
    constraint_3 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_indices': ['eq'],
        'right_value': 0.5
    }
    with pytest.raises(ValueError):
            AssetConstraint(**constraint_3)

    # A simple invalid constraint with neither right indices nor right value
    constraint_4 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '=='
    }
    with pytest.raises(ValueError):
            AssetConstraint(**constraint_4)
    
    # A simple invalid constraint with both incorrect right value format 
    constraint_5 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_value': [0.5, 0.6]
    }
    with pytest.raises(ValueError):
            AssetConstraint(**constraint_5)

    # A simple invalid constraint with both incorrect right indices format
    constraint_6 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_indices': 0.6
    }
    with pytest.raises(ValueError):
            AssetConstraint(**constraint_6)


    # Another simeple invalid constraint with right value above 1
    constraint_7 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_value': 1.5
    }
    with pytest.raises(ValueError):
            AssetConstraint(**constraint_7)


def test_check_constraints():
    # A simple valid constraint
    constraint_valid_1 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_indices': ['eq']
    }
    # A simple valid constraint with right value
    constraint_valid_2 = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_value': 0.5
    }
    # A simple invalid constraint with both right indices and right value
    constraint_invalid = {
        'left_indices': ['emdc', 'emdh'],
        'operator': '==',
        'right_indices': ['eq'],
        'right_value': 0.5
    }
    _check_constraints([constraint_valid_1, constraint_valid_2])
    with pytest.raises(ValueError):
        _check_constraints([constraint_invalid])
    with pytest.raises(ValueError):
        _check_constraints([constraint_invalid, constraint_valid_1])
    with pytest.raises(ValueError):
        _check_constraints([constraint_invalid, constraint_valid_2])
    

def test_process_left_side_of_constraint(): 
    n_assets = 5
    asset_names = [f'asset_{i}' for i in range(n_assets)]
    asset_names.append('liabilities')
    asset_indices = {name: i for i, name in enumerate(asset_names)}
    portfolio_weights = cp.Variable(n_assets + 1)
    
    valid_choices_for_left_side = [
        ['asset_0', 'asset_1'],
        ['asset_2', 'asset_3'],
        ['asset_0', 'asset_1', 'asset_2', 'asset_3'],
        ['asset_4'],
    ]
    for left_indices in valid_choices_for_left_side:
        left_side_processed = _process_left_side_of_constraint(portfolio_weights=portfolio_weights, left_indices=left_indices, asset_indices=asset_indices)
        expected_vec = np.zeros(n_assets + 1)
        for idx in left_indices:
            expected_vec[asset_indices[idx]] = 1
        expected_expr = expected_vec @ portfolio_weights
        assert str(left_side_processed) == str(expected_expr), f"Expected {expected_expr}, got {left_side_processed}"

def test_process_right_side_of_constraint():
    n_assets = 5
    asset_names = [f'asset_{i}' for i in range(n_assets)]
    asset_names.append('liabilities')
    asset_indices = {name: i for i, name in enumerate(asset_names)}
    portfolio_weights = cp.Variable(n_assets + 1)
    valid_choices_for_right_value = [0.5, 1.0, 0.0]
    for right_value in valid_choices_for_right_value:
        right_side_processed = _process_right_side_of_constraint(portfolio_weights=portfolio_weights, right_value=right_value, asset_indices=asset_indices)
        expected_expr = right_value
        assert str(right_side_processed) == str(expected_expr), f"Expected {expected_expr}, got {right_side_processed}"

    valid_choices_for_right_indices = [['asset_0'], ['asset_1', 'asset_2'], ['asset_3', 'asset_4']]
    for right_indices in valid_choices_for_right_indices:
        right_side_processed = _process_right_side_of_constraint(portfolio_weights=portfolio_weights, right_indices=right_indices, asset_indices=asset_indices)
        expected_vec = np.zeros(n_assets + 1)
        for idx in right_indices:
            expected_vec[asset_indices[idx]] = 1
        expected_expr = expected_vec @ portfolio_weights
        assert str(right_side_processed) == str(expected_expr), f"Expected {expected_expr}, got {right_side_processed}"


def test_generate_constraints():
    n_assets = 5
    asset_names = [f'asset_{i}' for i in range(n_assets)]
    asset_names.append('liabilities')
    asset_indices = {name: i for i, name in enumerate(asset_names)}
    portfolio_weights = cp.Variable(n_assets + 1)
    # A simple valid constraint
    constraint_valid_1 = {
        'left_indices': ['asset_0', 'asset_1'],
        'operator': '==',
        'right_indices': ['liabilities']
    }
    
    # A simple valid constraint with right value
    constraint_valid_2 = {
        'left_indices': ['asset_2', 'asset_3'],
        'operator': '>=',
        'right_value': 0.5
    }
    
    constraints = generate_constraints(portfolio_weights=portfolio_weights, asset_names=asset_names, asset_constraints=[constraint_valid_1, constraint_valid_2])
    
    assert len(constraints) == 2, "Expected 2 constraints"
    
    # Check the first constraint
    left_side_processed_1 = _process_left_side_of_constraint(portfolio_weights=portfolio_weights, left_indices=constraint_valid_1['left_indices'], asset_indices=asset_indices)
    right_side_processed_1 = _process_right_side_of_constraint(portfolio_weights=portfolio_weights, right_indices=constraint_valid_1['right_indices'], asset_indices=asset_indices)

    # Check the second constraint
    left_side_processed_2 = _process_left_side_of_constraint(portfolio_weights=portfolio_weights, left_indices=constraint_valid_2['left_indices'], asset_indices=asset_indices)
    right_side_processed_2 = _process_right_side_of_constraint(portfolio_weights=portfolio_weights, right_value=constraint_valid_2['right_value'], asset_indices=asset_indices)

    assert str(constraints[0] == str(left_side_processed_1) + ' ' + constraint_valid_1['operator'] + ' ' + str(right_side_processed_1)), \
        f"Expected {left_side_processed_1} {constraint_valid_1['operator']} {right_side_processed_1}, got {constraints[0]}"
    assert str(constraints[1] == str(left_side_processed_2) + ' ' + constraint_valid_2['operator'] + ' ' + str(right_side_processed_2)), \
        f"Expected {left_side_processed_2} {constraint_valid_2['operator']} {right_side_processed_2}, got {constraints[1]}"
    

def test_generate_scipy_constraints():
    asset_names = ['asset_0', 'asset_1', 'asset_2']
    asset_constraints = [
        {
            'left_indices': ['asset_0', 'asset_1'],
            'operator': '==',
            'right_value': 0.5
        },
        {
            'left_indices': ['asset_2'],
            'operator': '>=',
            'right_indices': ['asset_1']
        },
        {
            'left_indices': ['asset_0'],
            'operator': '>',
            'right_value': 0.3
        },
        {
            'left_indices': ['asset_1'],
            'operator': '<',
            'right_value': 0.3
        }
    ]
    constraints = generate_scipy_constraints(asset_names=asset_names, asset_constraints=asset_constraints)
    assert len(constraints) == 4
    assert constraints[0]['type'] == 'eq'
    assert constraints[1]['type'] == 'ineq'
    assert constraints[2]['type'] == 'ineq'
    assert constraints[3]['type'] == 'ineq'
    assert constraints[0]['fun'] is not None
    assert constraints[1]['fun'] is not None
    assert constraints[2]['fun'] is not None
    assert constraints[3]['fun'] is not None
    sample_weights = np.array([0.2, 0.3, 0.5])
    assert constraints[0]['fun'](sample_weights) == 0.5 - (0.2 + 0.3)
    assert constraints[1]['fun'](sample_weights) == 0.5 - 0.3
    assert constraints[2]['fun'](sample_weights) == 0.2 - 0.3
    assert constraints[3]['fun'](sample_weights) == 0.3 - 0.3

    for constraint in constraints:
        assert callable(constraint['fun']), "Constraint function must be callable"
        # Test the constraint function with a sample input
        result = constraint['fun'](sample_weights)
        assert isinstance(result, float), "Constraint function must return a float value"
