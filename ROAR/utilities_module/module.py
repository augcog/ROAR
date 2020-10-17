from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self, threaded=False):
        self.threaded = threaded

    @abstractmethod
    def run_in_series(self, **kwargs):
        """
        This is the none-threaded function. It run in series!
        Args:
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def run_in_threaded(self, **kwargs):
        """
        This is the threaded function.
        Args:
            **kwargs:

        Returns:

        """
        pass

    def shutdown(self):
        pass