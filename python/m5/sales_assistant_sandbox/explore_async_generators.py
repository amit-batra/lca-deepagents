"""Demonstrate an async generator.

Adding ``async`` to a generator function turns it into an *async generator*: it
returns an async iterator that must be consumed with ``async for`` instead of a
plain ``for``. This is useful when producing values may require ``await``-able
I/O such as database or network access.
"""

import asyncio


async def count_from_to(start: int, end: int):
    """Yield the integers from ``start`` up to and including ``end``.

    As with a sync generator, values are produced lazily one at a time.
    """
    while start <= end:
        yield start
        start += 1


async def main():
    """Consume the async generator and print each value."""
    async for item in count_from_to(-4, 4):
        print(item)


if __name__ == "__main__":
    asyncio.run(main())
