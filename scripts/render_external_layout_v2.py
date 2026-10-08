"""Layout-only historical external figure; no new experiment or CI claims."""
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts import make_figures as legacy


def coordinates(ax):
    return dict(
        lines=[dict(x=list(line.get_xdata()), y=list(line.get_ydata())) for line in ax.lines],
        bars=[list(rect.get_bbox().bounds) for rect in ax.patches],
        errors=[[segment.tolist() for segment in collection.get_segments()]
                for collection in ax.collections],
    )


def fix_chance(fig):
    if len(fig.axes) != 1:
        raise ValueError('one historical external axis required')
    ax = fig.axes[0]
    labels = [text for text in ax.texts if text.get_text() == 'chance']
    if len(labels) != 1 or len(ax.patches) != 6:
        raise ValueError('exact historical six bars and one chance label required')
    if [rect.get_height() for rect in ax.patches] != [.736, .684, .718, .848, .813, .749]:
        raise ValueError('historical bar values required; no substitution')
    before = coordinates(ax)
    ax.set_xlim(-.6, 3.05)
    labels[0].set_position((2.48, .508))
    labels[0].set_ha('left')
    assert coordinates(ax) == before, 'layout changed bar/error/line coordinates'
    return before


def main():
    out = PROJECT_ROOT / 'metrics/external_layout_v2_20261008'
    prefix = PROJECT_ROOT / 'latex/shared/figures/fig_external_layout_v2'
    inputs = [Path(__file__), Path(legacy.__file__),
              PROJECT_ROOT / 'latex/shared/figures/fig_external.pdf']
    before = [file_record(path) for path in inputs]
    assert not out.exists() and not prefix.with_suffix('.pdf').exists()
    assert not prefix.with_suffix('.png').exists(), 'fresh assets required'
    captured = []

    def save(fig, name):
        assert name == 'fig_external'
        captured.append(fix_chance(fig))
        for suffix in ('.pdf', '.png'):
            fig.savefig(prefix.with_suffix(suffix))
        legacy.plt.close(fig)

    original = legacy._save
    try:
        legacy._save = save
        legacy.fig_external('en')
    finally:
        legacy._save = original
    assert [file_record(path) for path in inputs] == before
    out.mkdir()
    metrics = dict(scope='layout only; historical values, not independent evaluations',
                   coordinates=captured[0], publication_ready=False,
                   scientific_evidence_validated=False)
    summary = out / 'summary.json'
    summary.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    write_experiment_manifest(out / 'summary.manifest.json', experiment='external-layout-only-v2',
                              parameters=dict(language='en', chance_position=[2.48, .508]),
                              metrics=metrics, inputs=inputs,
                              outputs=[summary, prefix.with_suffix('.pdf'), prefix.with_suffix('.png')])
    print('Historical bars/errors/lines unchanged; chance label moved to clear margin.', flush=True)


if __name__ == '__main__':
    main()
