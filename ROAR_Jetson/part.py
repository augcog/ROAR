from abc import ABC, abstractmethod
import logging


class Part(ABC):
    def __init__(self, name: str = "Unknown Part"):
        self.name = name
        self.logger = logging.getLogger(name)

    def run_threaded(self, should_continue):
        """This method will be run in parallel"""
        while True:
            self.run_step()
            if should_continue() is False:
                break

    @abstractmethod
    def run_step(self):
        """This method will be run in series"""
        pass

    @abstractmethod
    def shutdown(self):
        """Graceful shutdown needs to be implemented"""
        pass

