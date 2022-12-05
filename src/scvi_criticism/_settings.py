import logging
from typing import Literal, Union

from rich.console import Console
from rich.logging import RichHandler

criticism_logger = logging.getLogger("scvi_criticism")


# from https://github.com/YosefLab/scib-metrics/blob/main/src/scib_metrics/_settings.py
class CriticismConfig:
    """Config manager for scvi-criticism.

    Examples
    --------
    To set the progress bar style, choose one of "rich", "tqdm"

    >>> scvi_criticism.settings.progress_bar_style = "rich"

    To set the verbosity

    >>> import logging
    >>> scvi_criticism.settings.verbosity = logging.INFO
    """

    def __init__(
        self,
        verbosity: int = logging.INFO,
        progress_bar_style: Literal["rich", "tqdm"] = "tqdm",
    ):
        if progress_bar_style not in ["rich", "tqdm"]:
            raise ValueError("Progress bar style must be in ['rich', 'tqdm']")
        self.progress_bar_style = progress_bar_style
        self.verbosity = verbosity

    @property
    def progress_bar_style(self) -> str:
        """Library to use for progress bar."""
        return self._pbar_style

    @progress_bar_style.setter
    def progress_bar_style(self, pbar_style: Literal["tqdm", "rich"]):
        """Library to use for progress bar."""
        self._pbar_style = pbar_style

    @property
    def verbosity(self) -> int:
        """Verbosity level (default `logging.INFO`).

        Returns
        -------
        verbosity: int
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: Union[str, int]):
        """Set verbosity level.

        If "scvi_criticism" logger has no StreamHandler, add one.
        Else, set its level to `level`.

        Parameters
        ----------
        level
            Sets "scvi_criticism" logging level to `level`
        force_terminal
            Rich logging option, set to False if piping to file output.
        """
        self._verbosity = level
        criticism_logger.setLevel(level)
        if len(criticism_logger.handlers) == 0:
            console = Console(force_terminal=True)
            if console.is_jupyter is True:
                console.is_jupyter = False
            ch = RichHandler(level=level, show_path=False, console=console, show_time=False)
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            criticism_logger.addHandler(ch)
        else:
            criticism_logger.setLevel(level)

    def reset_logging_handler(self) -> None:
        """Reset "scvi_criticism" log handler to a basic RichHandler().

        This is useful if piping outputs to a file.

        Returns
        -------
        None
        """
        criticism_logger.removeHandler(criticism_logger.handlers[0])
        ch = RichHandler(level=self._verbosity, show_path=False, show_time=False)
        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)
        criticism_logger.addHandler(ch)


settings = CriticismConfig()
