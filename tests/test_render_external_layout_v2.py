import pytest

from scripts import render_external_layout_v2 as render


def source_figure(monkeypatch):
    captured = []
    monkeypatch.setattr(render.legacy, '_save', lambda fig, name: captured.append(fig))
    render.legacy.fig_external('en')
    return captured[0]


def test_layout_preserves_bars_error_segments_and_chance_line(monkeypatch):
    fig = source_figure(monkeypatch)
    ax = fig.axes[0]
    before = render.coordinates(ax)
    assert render.fix_chance(fig) == before == render.coordinates(ax)
    chance, = [text for text in ax.texts if text.get_text() == 'chance']
    assert chance.get_position() == (2.48, .508)
    assert chance.get_ha() == 'left'
    assert ax.get_xlim() == (-.6, 3.05)
    assert chance.get_position()[0] > max(rect.get_x() + rect.get_width() for rect in ax.patches)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_bounds = chance.get_window_extent(renderer)
    axis_bounds = ax.get_window_extent(renderer)
    assert axis_bounds.x0 <= text_bounds.x0 < text_bounds.x1 <= axis_bounds.x1
    render.legacy.plt.close(fig)


@pytest.mark.parametrize('corruption', ['bar', 'missing_label', 'duplicate_label'])
def test_wrong_source_rejected_without_layout_changes(monkeypatch, corruption):
    fig = source_figure(monkeypatch)
    ax = fig.axes[0]
    if corruption == 'bar':
        ax.patches[0].set_height(.1)
    elif corruption == 'missing_label':
        next(text for text in ax.texts if text.get_text() == 'chance').remove()
    else:
        ax.text(0, 0, 'chance')
    before = ax.get_xlim()
    with pytest.raises(ValueError):
        render.fix_chance(fig)
    assert ax.get_xlim() == before
    render.legacy.plt.close(fig)


def test_multiple_axes_rejected():
    fig, _ = render.legacy.plt.subplots(1, 2)
    with pytest.raises(ValueError, match='one historical'):
        render.fix_chance(fig)
    render.legacy.plt.close(fig)
