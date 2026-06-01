"""Supervised-learning Mahjong bot for Chinese Official Mahjong."""

__all__ = ["CNNModel", "FeatureAgent"]


def __getattr__(name):
    if name == "CNNModel":
        from .model import CNNModel

        return CNNModel
    if name == "FeatureAgent":
        from .feature import FeatureAgent

        return FeatureAgent
    raise AttributeError(name)
