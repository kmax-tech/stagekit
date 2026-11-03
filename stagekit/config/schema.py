from hydra.core.config_store import ConfigStore
from stagekit.config.classes import AppConfig

cs = ConfigStore.instance()
cs.store(name="app_schema", node=AppConfig)