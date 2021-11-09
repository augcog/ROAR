from abc import ABC, abstractmethod
import time
from pathlib import Path


class Module(ABC):
    def __init__(self, threaded=False, update_interval: float = 0.5,
                 should_save: bool = False, name: str = "module", **kwargs):
        self.threaded = threaded
        self.update_interval = update_interval
        self.should_continue_threaded = True
        self.should_save = should_save
        self.name = name
        self.saving_dir_path: Path = Path(f"data/output/{name}")
        if should_save and self.saving_dir_path.exists() is False:
            self.saving_dir_path.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def run_in_series(self, **kwargs):
        """
        This is the none-threaded function. It run in series!
        Args:
            **kwargs:

        Returns:

        """
        pass

    def run_in_threaded(self, **kwargs):
        """
        This is the threaded function.
        Args:
            **kwargs:

        Returns:

        """
        while self.should_continue_threaded:
            start = time.time()
            self.run_in_series()
            if self.should_save:
                self.save()
            end = time.time()
            if end - start < self.update_interval:
                time.sleep(self.update_interval - (end-start))

    def shutdown(self):
        self.should_continue_threaded = False

    @abstractmethod
    def save(self, **kwargs):
        pass
