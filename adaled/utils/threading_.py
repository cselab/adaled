import threading

class ThreadEx(threading.Thread):
    """Extend the builtin `threading.Thread` with exception propagation from
    the thread to the host."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception = None
        self.lock = threading.Lock()

    def run(self, *args, **kwargs):
        try:
            return super().run(*args, **kwargs)
        except Exception as e:
            with self.lock:
                self.exception = e

    def check_exception(self):
        with self.lock:
            if self.exception:
                raise RuntimeError("unhandled exception in a thread") from self.exception

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        self.check_exception()
