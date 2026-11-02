"""Demonstrate a basic generator function.

A function with one or more ``yield`` statements becomes a generator: calling it
returns a lazy iterator that produces values on demand. The body runs only as far
as the next ``yield`` each time, then suspends until the iterator advances again.
"""


def count_from_to(start: int, end: int) -> int:
    """Yield the integers from ``start`` up to and including ``end``.

    Values are produced lazily one at a time, so the memory cost is that of a
    single value rather than a list holding the whole range.
    """
    while start <= end:
        yield start
        start += 1


def main():
    """Consume the generator with a loop and print each value."""
    for item in count_from_to(4, 8):
        print(item)


if __name__ == "__main__":
    main()
