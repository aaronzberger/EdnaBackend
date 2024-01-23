from typing import Callable


class WriteBuffer:
    def __init__(self, write_fn: Callable[[str], None], buffer_size: int = 10000):
        """
        A class to write to a buffer, and flush the buffer after a certain size.

        Parameters
        ----------
        write_fn : Callable[[str], None]
            The function to write the buffer to.
        buffer_size : int, optional
            The size of the buffer, by default 10000
        """
        self._buffer: dict[str, str] = {}
        self._write_fn = write_fn
        self._buffer_size = buffer_size

    def write(self, key: str, val: str):
        self._buffer[key] = val
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self):
        self._write_fn(self._buffer)
        self._buffer = {}
