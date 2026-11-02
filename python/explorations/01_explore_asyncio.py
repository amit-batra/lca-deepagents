"""Asyncio concurrency experiment.

Demonstrates that coroutines scheduled on the same event loop run
*concurrently* rather than sequentially. Each task sleeps for 5 seconds, but
because both are awaited together via ``asyncio.gather``, the total wall-clock
time is ~5 seconds instead of the ~10 seconds sequential execution would take.
"""

import asyncio
from datetime import datetime


async def task_1() -> None:
    """Simulate a 5-second unit of work for task 1."""

    print("Started Task 1")
    # Yield control back to the event loop so other tasks can run.
    await asyncio.sleep(5)
    print("Finished Task 1")


async def task_2() -> None:
    """Simulate a 5-second unit of work for task 2."""

    print("Started Task 2")
    # Yield control back to the event loop so other tasks can run.
    await asyncio.sleep(5)
    print("Finished Task 2")


async def main() -> None:
    """Run both tasks concurrently and report the elapsed time."""
    start: datetime = datetime.now()

    # Gather schedules the coroutines as tasks and awaits them together,
    # so their sleeps overlap instead of running one after the other.
    await asyncio.gather(task_1(), task_2())
    end: datetime = datetime.now()

    # Expect ~5 seconds, not ~10, confirming the tasks ran in parallel.
    print(f"Total time elapsed: {end - start}")


# Entry point: hand main() to the event loop to execute.
if __name__ == "__main__":
    asyncio.run(main())
