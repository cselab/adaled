from adaled.backends.numpy import make_numpy_like_backend

import cupy as cp

class CupyBackend(make_numpy_like_backend('CupyBackend', cp)):
    pass
