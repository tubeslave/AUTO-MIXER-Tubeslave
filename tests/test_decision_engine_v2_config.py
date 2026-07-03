import inspect

from automixer.config import load_decision_engine_v2_config
from auto_soundcheck_engine import AutoSoundcheckEngine


def test_decision_engine_v2_config_loads_disabled_by_default():
    config = load_decision_engine_v2_config()

    assert config.enabled is False
    assert config.dry_run is True
    assert config.safety.dry_run is True
    assert config.knowledge_path.endswith("mixing_knowledge_base.json")


def test_legacy_soundcheck_engine_defaults_do_not_enable_v2():
    signature = inspect.signature(AutoSoundcheckEngine)

    assert signature.parameters["use_decision_engine_v2"].default is False
    assert signature.parameters["decision_engine_dry_run"].default is False
