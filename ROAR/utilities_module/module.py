from abc import ABC, abstractmethod
import time
from pathlib import Path
from threading import Thread, Event
import datetime

class Module(Thread):
    def __init__(self, event: Event = Event(), threaded=False,
                 update_interval: float = 0.5, should_save: bool = False,
                 name: str = "module", **kwargs):
        Thread.__init__(self)
        self.threaded = threaded
        self.update_interval = update_interval
        self.should_continue_threaded = True
        self.should_save = should_save
        self.name = name
        self.saving_dir_path: Path = Path(f"data/output/{name}")
        if should_save and self.saving_dir_path.exists() is False:
            self.saving_dir_path.mkdir(exist_ok=True, parents=True)
        self.stopped: Event = event
        self.stopped.set()

    @abstractmethod
    def run_in_series(self, **kwargs):
        """
        This is the none-threaded function. It run in series!
        Args:
            **kwargs:

        Returns:

        """
        pass

    def run(self) -> None:
        self.run_in_threaded()

    def run_in_threaded(self, **kwargs):
        """
        This is the threaded function.
        Args:
            **kwargs:

        Returns:

        """
        if self.update_interval <= 0.025:
            while self.should_continue_threaded:
                start = time.time()
                self.run_in_series()
                if self.should_save:
                    self.save()
                end = time.time()
                if end - start < self.update_interval * 1000:
                    time.sleep((end-start)*0.001)

        else:
            while self.should_continue_threaded and self.stopped.wait(self.update_interval):
                self.run_in_series()
                if self.should_save:
                    self.save()

    def shutdown(self):
        self.should_continue_threaded = False
        self.stopped.set()

    @abstractmethod
    def save(self, **kwargs):
        pass
