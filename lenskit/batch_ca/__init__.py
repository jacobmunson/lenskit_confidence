"""
Batch-run predictors and recommenders for evaluation.
"""

from ._predict_ca import predict  # noqa: F401
from ._recommend_ca import recommend  # noqa: F401
from ._multi_ca import MultiEval  # noqa: F401
from ._train_ca import train_isolated  # noqa: F401
