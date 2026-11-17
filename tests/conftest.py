import sys
from pathlib import Path

import pytest

# Make `import api...` work no matter where pytest is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.synthetic import make_landmarks_csv


@pytest.fixture
def landmarks(tmp_path):
    """Factory: landmarks(**options) -> path of a synthetic landmarks CSV.

    Options are those of tests.synthetic.make_landmarks_csv
    (seconds, fps, period, duty, facing, left_visible, right_visible).
    """
    counter = {"n": 0}

    def make(**options):
        counter["n"] += 1
        path = tmp_path / f"landmarks_{counter['n']}.csv"
        make_landmarks_csv(path, **options)
        return path

    return make
