"""Edit models and estimator settings here; no Stata-log agreement is required."""
from copy import deepcopy

# Controls are source/prepared variable names, not precomputed dummy names.
# Eligibility order preserves the original complete-case specification.
COUNTRY_CONTROLS = ['lgl_systm', 'defendant5', 'high_court']
MODE_CONTROLS = ['mode_electronic_internet', 'mode_press_newspapers',
                 'mode_public_assembly', 'mode_public_speech', 'mode_non_verbal_expression']
CONTROLS = COUNTRY_CONTROLS + MODE_CONTROLS
CONTROL_GROUPS = {'country': COUNTRY_CONTROLS, 'modes': MODE_CONTROLS, 'all': CONTROLS}
# Known controls retain the original design order, independently of eligibility.
DESIGN_CONTROL_ORDER = ['lgl_systm', 'high_court', 'defendant5'] + MODE_CONTROLS
CATEGORIES = {
    'lgl_systm': {'reference': 0, 'levels': {0: 'Common', 1: 'Civil', 2: 'Mixed'},
                  'columns': [(1, 'Civil'), (2, 'Mixed')]},
    'defendant5': {'reference': 4, 'levels': {1: 'Citizen', 2: 'Press', 3: 'Intermediary',
                                            4: 'Government', 5: 'Other/unclear'},
                   'columns': [(1, 'Citizen'), (2, 'Press'), (3, 'Intermediary'), (5, 'Other/unclear')]},
    'high_court': {'reference': 0, 'levels': {0: '0', 1: '1'}, 'columns': [(1, '1.high_court')]},
    **{v: {'reference': 0, 'levels': {0: '0', 1: '1'}, 'columns': [(1, '1.'+v)]} for v in MODE_CONTROLS},
}

SPECIFICATIONS = {
    'indep1': {'focal_predictors': ['court_independence_lag1'], 'controls': []},
    'indep2': {'focal_predictors': ['court_independence_lag1'], 'controls': CONTROLS.copy()},
    'attack1': {'focal_predictors': ['v2jupoatck_lag1'], 'controls': []},
    'attack2': {'focal_predictors': ['v2jupack_lag1'], 'controls': []},
    'attack3': {'focal_predictors': ['v2jureform_lag1'], 'controls': []},
    'attack4': {'focal_predictors': ['v2jupurge_lag1'], 'controls': []},
    'attack5': {'focal_predictors': ['v2jupurge_lag1'], 'controls': CONTROLS.copy()},
    'dejure1': {'focal_predictors': ['wdj_expression_lag1'], 'controls': []},
    'dejure2': {'focal_predictors': ['wdj_expression_lag1'], 'controls': CONTROLS.copy()},
    'dejure3': {'focal_predictors': ['wdj_press_lag1'], 'controls': []},
    'dejure4': {'focal_predictors': ['wdj_press_lag1'], 'controls': CONTROLS.copy()},
    'dejure5': {'focal_predictors': ['wdj_citizen_lag1'], 'controls': []},
    'dejure6': {'focal_predictors': ['wdj_citizen_lag1'], 'controls': CONTROLS.copy()},
    'full1': {'focal_predictors': ['court_independence_lag1', 'v2jupurge_lag1'], 'controls': CONTROLS.copy()},
    'full2': {'focal_predictors': ['court_independence_lag1', 'v2jupurge_lag1', 'wdj_expression_lag1'], 'controls': CONTROLS.copy()},
    'full3': {'focal_predictors': ['court_independence_lag1', 'v2jupurge_lag1', 'wdj_press_lag1'], 'controls': CONTROLS.copy()},
    'full4': {'focal_predictors': ['court_independence_lag1', 'v2jupurge_lag1', 'wdj_citizen_lag1'], 'controls': CONTROLS.copy()},
}

SPECIFICATIONS['full2_reduced'] = deepcopy(SPECIFICATIONS['full2'])
SPECIFICATIONS['full2_reduced']['controls'] = [
    control
    for control in SPECIFICATIONS['full2']['controls']
    if control not in ['defendant5', 'mode_public_assembly']
]

ESTIMATOR_SETTINGS = {'method': 'newton', 'maxiter': 100, 'tol': 1e-12, 'disp': False}
NUMERIC_STORAGE = 'float32'
SPLINE_STORAGE = 'float32'
# These conventions are shared with the verified numerical implementation.
OUTCOME_MAPPING = {'Mixed Outcome': 0, 'Contracts Expression': 1, 'Expands Expression': 2}
CLUSTER_VARIABLE = 'country_id'
SPLINE_KNOTS = [2001, 2016, 2022]

# Example copied experiment (not enabled in the default 17 models):
# SPECIFICATIONS['full2_no_legal_system'] = deepcopy(SPECIFICATIONS['full2'])
# SPECIFICATIONS['full2_no_legal_system']['controls'].remove('lgl_systm')
# A control list can also include '@country' or '@modes' to expand a named group.
