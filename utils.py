import time


class ProgressBar:

    def __init__(self, total: int, width: int = 40, update_rate: int = 100):
        self._total = total
        self._width = width
        self._every = max(1, total // update_rate)
        self._start = time.time()

    def update(self, current: int):
        if current % self._every != 0:
            return
        pct = 100 * current // self._total
        filled = self._width * current // self._total
        bar = "█" * filled + "░" * (self._width - filled)
        print(f"\r[{bar}] {pct}%", end="", flush=True)

    def finish(self):
        elapsed = time.time() - self._start
        print(f" — done in {elapsed:.2f}s")
