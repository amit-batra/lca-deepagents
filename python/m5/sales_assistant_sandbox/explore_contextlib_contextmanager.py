"""Demonstrate turning a generator into a reusable context manager.

``contextlib.contextmanager`` lets you write a ``with``-statement resource
handler as a small generator function instead of a full class with
``__enter__``/``__exit__``. The code before ``yield`` runs on entry, the
code after ``yield`` runs on exit, and the value passed to ``yield`` becomes
the object bound to the ``as`` name inside the ``with`` block.
"""

import contextlib


class MyDBConnection:
    """A tiny stand-in for a database connection that must be closed."""

    def __init__(self, host: str, port: int):
        """Open a connection to ``host`` on ``port``."""
        self._host = host
        self._port = port
        print(f"Opened a connection to {self._host}:{self._port}")

    def fetch_items(self) -> list[int]:
        """Return a fixed list of rows from the connection."""
        return [1, 2, 3, 4, 5]

    def close(self):
        """Release the connection to ``host`` on ``port``."""
        print(f"Closed the connection to {self._host}:{self._port}")


@contextlib.contextmanager
def managed_connection(host: str, port: int):
    """Yield an open ``MyDBConnection``, guaranteeing it is closed on exit.

    The connection is opened before the ``with`` body and closed in the
    ``finally`` block, so it is always released even if the body raises.
    """
    db_connection: MyDBConnection
    try:
        db_connection = MyDBConnection(host=host, port=port)
        yield db_connection
    finally:
        db_connection.close()


def main():
    """Show the context manager opening and closing around a ``with`` body."""
    with managed_connection("www.host.com", 5432) as db_connection:
        items: list[int] = db_connection.fetch_items()
        print(f"Fetched items: {items}")


if __name__ == "__main__":
    main()
