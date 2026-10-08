import pytest
from matplotlib.ticker import NullLocator

from scripts import render_scaling_layout_v2 as render


def test_ticks_preserve_all_line_and_band_coordinates():
    fig, ax = render.legacy.plt.subplots()
    ax.set_xscale('log')
    ax.plot(render.POSITIONS, [.846, .852, .856, .848])
    ax.fill_between(render.POSITIONS, [.83] * 4, [.87] * 4)
    before = render.coordinates(ax)
    assert render.fix_ticks(fig) == before == render.coordinates(ax)
    assert list(ax.get_xticks()) == list(render.POSITIONS)
    assert isinstance(ax.xaxis.get_minor_locator(), NullLocator)
    assert [ax.xaxis.get_major_formatter()(value, index)
            for index, value in enumerate(render.POSITIONS)] == ['1,500', '3,750', '7,499', '14,998']
    render.legacy.plt.close(fig)


def test_wrong_source_positions_rejected():
    fig, ax = render.legacy.plt.subplots()
    ax.set_xscale('log')
    ax.plot([1, 2], [.1, .2])
    with pytest.raises(ValueError, match='source positions'):
        render.fix_ticks(fig)
    render.legacy.plt.close(fig)


def test_multi_axis_figure_rejected():
    fig, _ = render.legacy.plt.subplots(1, 2)
    with pytest.raises(ValueError, match='one historical'):
        render.fix_ticks(fig)
    render.legacy.plt.close(fig)
