"""
Timestamped progress logging
"""

import sys
import time


class Progress:
    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.start = time.time()

    def __call__(self, message, indent=0):
        elapsed = time.time() - self.start
        prefix = f"[{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}]"
        self.stream.write(f"{prefix} {'  ' * indent}{message}\n")
        self.stream.flush()

    def rule(self, title):
        self("")
        self("=" * 70)
        self(title)
        self("=" * 70)

    def elapsed(self):
        return time.time() - self.start


class Timer:
    """
    Context manager reporting how long a stage took.
    """

    def __init__(self, log, message, indent=0):
        self.log = log
        self.message = message
        self.indent = indent

    def __enter__(self):
        self.start = time.time()
        self.log(f"{self.message}...", self.indent)
        return self

    def __exit__(self, *exc):
        self.log(f"{self.message}: done in {time.time() - self.start:.1f}s", self.indent)
        return False
