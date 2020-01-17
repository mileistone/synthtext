from . import _config as CFG


def load_cfg(obj):
    global CFG
    dd = getattr(CFG, obj.__class__.__name__)
    for k, v in dd.items():
        setattr(obj, k, v)
