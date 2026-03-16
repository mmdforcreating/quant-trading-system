"""
配置管理器 - 读取 config.yaml 并提供属性式访问。

支持：
- 嵌套字典的点号访问: cfg.rolling.step
- 环境变量覆盖: ${HOME} 等路径展开
- 默认值回退: cfg.get("key", default)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


class _AttrDict(dict):
    """允许通过 attribute 访问 dict 键的辅助类。"""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"配置中不存在键: '{key}'")
        if isinstance(val, dict) and not isinstance(val, _AttrDict):
            val = _AttrDict(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any):
        self[key] = value


def _expand_env(obj):
    """递归展开配置值中的环境变量和 ~ 路径。"""
    if isinstance(obj, str):
        obj = os.path.expandvars(obj)
        obj = os.path.expanduser(obj)
        return obj
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    return obj


class ConfigManager(_AttrDict):
    """
    全局配置管理器。

    用法::

        cfg = ConfigManager("configs/config.yaml")
        print(cfg.strategy_name)
        print(cfg.models.catboost.kwargs)
        print(cfg.rolling.step)
    """

    def __init__(self, config_path: str | Path):
        yaml = YAML(typ="safe", pure=True)
        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path) as f:
            raw = yaml.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"配置文件格式错误，期望 dict，得到 {type(raw)}")

        expanded = _expand_env(raw)
        super().__init__(expanded)
        self._config_path = config_path

    @property
    def config_path(self) -> Path:
        return self._config_path

    def get(self, key: str, default: Any = None) -> Any:
        """支持点号分隔的嵌套键查找: cfg.get('rolling.step', 5)"""
        parts = key.split(".")
        obj = self
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                return default
        return obj
