"""Contact manipulation tasks; importing this package registers the Gym task."""

# Register string entry points without importing simulation code before AppLauncher.
from . import config  # noqa: F401
