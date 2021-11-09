from typing import List
try:
    from ROAR_Jetson.part import Part
except:
    from part import Part
import logging
from threading import Thread
import time


class Vehicle:
    """
    An interface that takes care of starting parts automatically
    """

    def __init__(self):
        self._parts: List[Part] = []
        self.logger = logging.getLogger("Vehicle")
        self.should_continue = True
        self.throttle: float = 0
        self.steering: float = 0

    def add(self, part: Part):
        self._parts.append(part)

    def remove(self, part: Part):
        self._parts.remove(part)

    def start_parts(self):
        for part in self._parts:
            t = Thread(target=part.run_threaded, args=(lambda: self.should_continue,))
            t.daemon = True
            t.start()
            self.logger.debug(f"Part {part.name} started")

    def stop_parts(self):
        # if any of them refuse to do so, forcefully kill that thread
        self.logger.info("Shutting Vehicle Down safely")
        self.should_continue = False
        time.sleep(1)  # sleep to give time for threads to die
        try:
            for part in self._parts:
                part.shutdown()  # ask the parts to shut down nicely
        except Exception as e:
            self.logger.error(f"Unable to gracefully shutdown all parts: {e}")

        self.logger.info("Vehicle Shut down safely")
        exit(0) # sometimes python just freeze, just force quit it at this point since all thread must have been shut
