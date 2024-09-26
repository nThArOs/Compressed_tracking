
from .....ffi import _wrap_function
#from ._correlation import lib as _lib, ffi as _ffi
import lib as _lib
import cffi as _ffi
__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())
