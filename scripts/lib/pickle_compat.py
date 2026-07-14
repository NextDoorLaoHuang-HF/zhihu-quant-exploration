"""
pandas 3.0 兼容的 pickle 加载器。
处理旧版 pandas 缓存中 StringDtype 的不兼容问题。
"""
import pickle
import pandas as pd
import numpy as np


class _NDArrayBackedCompat:
    """替代 NDArrayBacked，处理旧版 StringDtype 格式。"""
    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            self._ndarray = np.array(state[1], dtype=object)
            self._dtype = pd.StringDtype()
        else:
            raise ValueError(f"Unexpected state: {type(state)}")

    def __getattr__(self, name):
        return getattr(self._ndarray, name)

    def __iter__(self):
        return iter(self._ndarray)

    def __len__(self):
        return len(self._ndarray)

    def __getitem__(self, key):
        return self._ndarray[key]


class CompatUnpickler(pickle._Unpickler):
    """使用纯 Python 实现的 Unpickler，确保 find_class 被正确调用。

    pickle.Unpickler (C 实现) 在某些情况下会跳过 find_class，
    导致无法拦截 NDArrayBacked 的构造。
    """

    def find_class(self, module, name):
        if module == 'pandas._libs.arrays' and name == 'NDArrayBacked':
            return _NDArrayBackedCompat
        return super().find_class(module, name)


def load_pickle_compat(path):
    """加载 pickle 文件，兼容 pandas 3.0 的 StringDtype 问题。"""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except NotImplementedError:
        with open(path, 'rb') as f:
            return CompatUnpickler(f).load()
