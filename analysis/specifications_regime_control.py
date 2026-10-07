"""Pooled full models with decision-year regime as a categorical control."""
from copy import deepcopy

from specifications import (
    CATEGORIES as BASE_CATEGORIES,
    CLUSTER_VARIABLE,
    CONTROL_GROUPS,
    DESIGN_CONTROL_ORDER as BASE_DESIGN_CONTROL_ORDER,
    ESTIMATOR_SETTINGS,
    NUMERIC_STORAGE,
    OUTCOME_MAPPING,
    SPECIFICATIONS as BASE_SPECIFICATIONS,
    SPLINE_KNOTS,
    SPLINE_STORAGE,
)

CATEGORIES = deepcopy(BASE_CATEGORIES)
CATEGORIES['regime_binary'] = {
    'reference': 'autocracy',
    'levels': {'autocracy': 'Autocracy', 'democracy': 'Democracy'},
    'columns': [('democracy', 'democracy.regime_binary')],
}
DESIGN_CONTROL_ORDER = [*BASE_DESIGN_CONTROL_ORDER, 'regime_binary']
SPECIFICATIONS = {
    name: deepcopy(BASE_SPECIFICATIONS[name])
    for name in ['full1', 'full2', 'full3', 'full4']
}
for specification in SPECIFICATIONS.values():
    specification['controls'].append('regime_binary')
