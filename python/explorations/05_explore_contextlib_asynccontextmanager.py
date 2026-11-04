"""Demonstrate turning an async generator into a reusable async context manager.

``contextlib.asynccontextmanager`` is the async counterpart to
``contextlib.contextmanager``. It lets you write an ``async with``-statement
resource handler as a small *async generator* function instead of a full class
with ``__aenter__``/``__aexit__``. The code before ``yield`` runs on entry, the
code after ``yield`` runs on exit, and the value passed to ``yield`` becomes the
object bound to the ``as`` name inside the ``async with`` block.
"""

import asyncio
import contextlib


class MyDBConnection:
    """A tiny stand-in for an async database connection that must be closed."""

    def __init__(self, host: str, port: int):
        """Open a connection to ``host`` on ``port``."""
        self._host = host
        self._port = port
        print(f"Opened a connection to {self._host}:{self._port}")

    async def fetch_items(self) -> list[int]:
        """Return a fixed list of rows from the connection.

        The ``await`` simulates asynchronous I/O, the kind of work an async
        context manager is useful for guarding.
        """
        await asyncio.sleep(0)
        return [1, 2, 3, 4, 5]

    async def close(self):
        """Release the connection to ``host`` on ``port``."""
        print(f"Closed the connection to {self._host}:{self._port}")


@contextlib.asynccontextmanager
async def managed_connection(host: str, port: int):
    """Yield an open ``MyDBConnection``, guaranteeing it is closed on exit.

    The connection is opened before the ``async with`` body and closed in the
    ``finally`` block, so it is always released even if the body raises.
    """
    db_connection: MyDBConnection
    try:
        db_connection = MyDBConnection(host=host, port=port)
        yield db_connection
    finally:
        await db_connection.close()


async def main():
    """Show the async context manager opening and closing around a body."""
    async with managed_connection("www.host.com", 5432) as db_connection:
        items: list[int] = await db_connection.fetch_items()
        print(f"Fetched items: {items}")


if __name__ == "__main__":
    asyncio.run(main())
