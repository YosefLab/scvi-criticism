import logging
from importlib.metadata import version

from ._ppc import PosteriorPredictiveCheck
from ._ppc_plot import PPCPlot
from ._settings import settings

settings.verbosity = logging.INFO
# prevent double output
logger = logging.getLogger("scvi_criticism")
logger.propagate = False

__version__ = version("scvi-criticism")
__all__ = ["PosteriorPredictiveCheck", "PPCPlot"]
