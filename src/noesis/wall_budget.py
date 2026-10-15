"""Process deadline guard, including native simulator setup and video encoding."""
import os
import threading
import time


class WallBudget:
    """Hard process stop at deadline. Ledger must be persisted before entering work.

    A killed episode is deliberately not completed. Resume uses its saved start
    checkpoint and seed, and preserves its interrupted artifacts in a new attempt.
    """
    def __init__(self, deadline):
        self.timer = threading.Timer(max(0., deadline-time.monotonic()), lambda: os._exit(124))
        self.timer.daemon = True

    def __enter__(self):
        self.timer.start()
        return self

    def __exit__(self, *args):
        self.timer.cancel()
