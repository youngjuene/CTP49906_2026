import pytest
from src.config import load_config
from src.server import Atlas
from test_server_routes import ROSTER_CSV


def test_unreviewed_real_segmentation_refuses_direct_startup_before_model(tmp_path, monkeypatch):
    roster = tmp_path/'roster.csv'; roster.write_text(ROSTER_CSV)
    cfg = load_config(dict(ATLAS_ADMIN_CODE='test', ATLAS_ROSTER=str(roster), ATLAS_DB=str(tmp_path/'a.db')))
    def model(*args):
        raise AssertionError('model should not load before quality gate')
    monkeypatch.setattr('src.server.build_embedder', model)
    with pytest.raises(RuntimeError, match='semantic quality'):
        Atlas(cfg).startup()


def test_development_override_is_explicit_configuration(tmp_path):
    cfg = load_config(dict(ATLAS_ADMIN_CODE='test', ATLAS_ROSTER='unused', ATLAS_ALLOW_UNREVIEWED_SEMANTICS='1'))
    assert cfg.allow_unreviewed_semantics is True
