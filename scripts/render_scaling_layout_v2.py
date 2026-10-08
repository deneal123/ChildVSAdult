"""Layout-only rendering of the explicitly historical scaling figure.

Reuse the unchanged legacy producer; preserve curve and band coordinates exactly.
This does not validate overwritten historical checkpoints or establish efficiency.
"""
import json
from pathlib import Path

from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts import make_figures as legacy

POSITIONS = (1500, 3750, 7499, 14998)


def coordinates(ax):
    return dict(lines=[dict(x=line.get_xdata().tolist() if hasattr(line.get_xdata(), 'tolist')
                           else list(line.get_xdata()),
                           y=line.get_ydata().tolist() if hasattr(line.get_ydata(), 'tolist')
                           else list(line.get_ydata())) for line in ax.lines],
                bands=[[path.vertices.tolist() for path in item.get_paths()]
                       for item in ax.collections])


def fix_ticks(fig):
    if len(fig.axes) != 1:
        raise ValueError('one historical scaling axis required')
    ax = fig.axes[0]
    if ax.get_xscale() != 'log' or tuple(ax.lines[0].get_xdata()) != POSITIONS:
        raise ValueError('historical source positions required; no data substitution')
    before = coordinates(ax)
    ax.xaxis.set_major_locator(FixedLocator(POSITIONS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    ax.xaxis.set_minor_locator(NullLocator())
    assert coordinates(ax) == before, 'tick formatting changed scientific coordinates'
    return before


def main():
    out = PROJECT_ROOT / 'metrics/scaling_layout_v2_20261008'
    prefix = PROJECT_ROOT / 'latex/shared/figures/fig_scaling_layout_v2'
    inputs = [Path(__file__), Path(legacy.__file__),
              PROJECT_ROOT / 'latex/shared/figures/fig_scaling.pdf']
    before = [file_record(path) for path in inputs]
    assert not out.exists() and not prefix.with_suffix('.pdf').exists()
    assert not prefix.with_suffix('.png').exists(), 'fresh versioned assets required'
    captured = []

    def save(fig, name):
        assert name == 'fig_scaling'
        captured.append(fix_ticks(fig))
        for suffix in ('.pdf', '.png'):
            fig.savefig(prefix.with_suffix(suffix))
        legacy.plt.close(fig)

    original = legacy._save
    try:
        legacy._save = save
        legacy.fig_scaling('en')
    finally:
        legacy._save = original
    assert [file_record(path) for path in inputs] == before
    out.mkdir()
    metrics = dict(scope='layout-only; historical constants, not checkpoint reproducibility',
                   coordinates=captured[0], ticks=list(POSITIONS),
                   publication_ready=False, scientific_evidence_validated=False)
    summary = out / 'summary.json'
    summary.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    write_experiment_manifest(out / 'summary.manifest.json', experiment='scaling-layout-only-v2',
                              parameters=dict(language='en', minor_ticks=False), metrics=metrics,
                              inputs=inputs,
                              outputs=[summary, prefix.with_suffix('.pdf'), prefix.with_suffix('.png')])
    print('Historical curve/band coordinates unchanged; versioned layout assets rendered.', flush=True)


if __name__ == '__main__':
    main()
