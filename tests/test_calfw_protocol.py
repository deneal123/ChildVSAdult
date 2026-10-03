from __future__ import annotations

import pytest

from scripts import calfw_protocol
from scripts.calfw_protocol import parse_pairs, person_of, summarize


def test_fold_token_is_not_binary_label_or_negative_fold():
    pairs = parse_pairs(b'Kim_Jong-Il_0001.jpg 10\r\nKim_Jong-Il_0002.jpg 10\r\n'
                        b'Ada_Name_0001.jpg 0\r\nOther_Name_0001.jpg 0\r\n')
    assert pairs[0].is_same and pairs[0].positive_fold == 10
    assert pairs[0].person_a == 'Kim_Jong-Il'
    assert not pairs[1].is_same and pairs[1].positive_fold is None


@pytest.mark.parametrize('raw', [
    b'', b'A_0001.jpg 1\n', b'A_0001.jpg 1\nA_0002.jpg 2\n',
    b'A_0001.jpg 11\nA_0002.jpg 11\n', b'A_0001.jpg 1\nB_0002.jpg 1\n',
    b'A_0001.jpg 0\nA_0002.jpg 0\n', b'A_0001.jpg 1\nA_0001.jpg 1\n',
    b'../A_0001.jpg 1\nA_0002.jpg 1\n', b'A_0000.jpg 1\nA_0002.jpg 1\n',
    b'A_0001.jpg 01\nA_0002.jpg 01\n', b'A_0001.jpg 1\n\nA_0002.jpg 1\n',
])
def test_rejects_invalid_structure(raw):
    with pytest.raises(ValueError):
        parse_pairs(raw)


def test_person_uses_final_image_index_not_underscores_in_name():
    assert person_of('Person_With_Underscores_0012.jpg') == 'Person_With_Underscores'


def test_summary_does_not_expose_identity_or_claim_bin_binding():
    lines = []
    for fold in range(1, 11):
        for index in range(300):
            lines.extend([f'Private_{fold}_{index}_0001.jpg {fold}',
                          f'Private_{fold}_{index}_0002.jpg {fold}'])
    for index in range(3000):
        lines.extend([f'Negative_A_{index}_0001.jpg 0', f'Negative_B_{index}_0001.jpg 0'])
    result = summarize(parse_pairs(('\n'.join(lines) + '\n').encode()))
    assert result['pairs'] == 6000
    assert not result['bin_binding_verified'] and not result['subject_ci_available']
    assert 'Private_' not in str(result) and 'Negative_A_' not in str(result)
    assert result['negative_fold_assignment'] == 'not encoded in source text'


def test_summary_refuses_short_protocol():
    with pytest.raises(ValueError, match='6000'):
        summarize(parse_pairs(b'A_0001.jpg 1\nA_0002.jpg 1\n'))


def test_cli_refuses_unverified_source_without_output(tmp_path, monkeypatch):
    source = tmp_path / 'pairs.txt'
    source.write_bytes(b'A_0001.jpg 1\nA_0002.jpg 1\n')
    out = tmp_path / 'out'
    monkeypatch.setattr('sys.argv', ['audit', '--pairs', str(source), '--out', str(out)])
    with pytest.raises(ValueError, match='checksum mismatch'):
        calfw_protocol.main()
    assert not out.exists()


def test_cli_preserves_existing_output(tmp_path, monkeypatch):
    out = tmp_path / 'existing'
    out.mkdir()
    sentinel = out / 'keep.txt'
    sentinel.write_text('user-owned', encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['audit', '--pairs', str(tmp_path / 'missing'),
                                   '--out', str(out)])
    with pytest.raises(FileExistsError):
        calfw_protocol.main()
    assert sentinel.read_text(encoding='utf-8') == 'user-owned'
