"""Routines for experiments in Thoth not only for Jupyter notebooks."""

from .utils import obtain_location
from .graph import GraphQueryResult
from .utils import packages_info

__title__ = "thoth-lab"
__version__ = "0.2.6"


__all__ = [
    obtain_location.__name__,
    GraphQueryResult.__name__,
    packages_info.__name__,
]
