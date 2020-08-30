from pathlib import Path

import pytest


@pytest.fixture
def datadir(request):
    """
    Fixture that gives you Path to test-data dir.

    Returns the data dir in the same dir as the test itself.
    """
    return Path(request.module.__file__).parent / "data"
