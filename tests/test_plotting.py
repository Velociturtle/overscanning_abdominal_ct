import sys, pathlib, pytest
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

pytest.importorskip('numpy')
pytest.importorskip('pandas')
pytest.importorskip('matplotlib')

import plotting

class DummyAx:
    def __init__(self):
        self.limits = None
    def set_ylim(self, a, b):
        self.limits = (a, b)


def test_safe_ylim_sets_limits(monkeypatch):
    import numpy as np
    monkeypatch.setattr(plotting, 'np', np)
    ax = DummyAx()
    plotting.safe_ylim(ax, 10)
    assert ax.limits == (-10.5, 10.5)
