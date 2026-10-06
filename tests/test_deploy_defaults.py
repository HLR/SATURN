"""With registry defaults, serving/deploy.py deploys exactly the three shipped services.

Pure: imports serving.deploy (settings only) and never touches ray/torch.
"""
from saturn import settings
from saturn.serving.deploy import select_apps

_FLAGS = ("SAPY_ENABLE_VGGT", "SAPY_ENABLE_SAM3")


def _apps():
    """What deploy.main() selects from the registered flags."""
    return select_apps(*(settings.env(f) == "1" for f in _FLAGS))


def test_registry_defaults_deploy_only_shipped_services(monkeypatch):
    for f in _FLAGS:
        monkeypatch.delenv(f, raising=False)
    assert _apps() == {"vggt", "sam3", "oriany"}


def test_oriany_is_always_deployed(monkeypatch):
    for f in _FLAGS:
        monkeypatch.setenv(f, "0")
    assert _apps() == {"oriany"}
    assert select_apps(False, False) == {"oriany"}
