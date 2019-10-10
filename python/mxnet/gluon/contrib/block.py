from ..block import SymbolBlock, _regroup


class SymbolBlockThreadSafe(SymbolBlock):
    instance = None
    def __new__(cls, outputs, inputs, params=None):
        if not SymbolBlockThreadSafe.instance:
            SymbolBlockThreadSafe.instance = SymbolBlock(outputs, inputs, params)
        return SymbolBlockThreadSafe.instance

    @staticmethod
    def imports(symbol_file, input_names, param_file=None, ctx=None):
        if not SymbolBlockThreadSafe.instance:
            SymbolBlockThreadSafe.instance = SymbolBlock.imports(outputs, inputs, params)
            return SymbolBlockThreadSafe.instance
        else:
            assert False, "Cannot import again in the same process"

    def forward(self, x, *args):
        if isinstance(x, NDArray):
            with x.context:
                return self._call_cached_op(x, *args, thread_safe=True)

        assert isinstance(x, Symbol), \
            "HybridBlock requires the first argument to forward be either " \
            "Symbol or NDArray, but got %s"%type(x)
        args, in_fmt = _flatten([x] + list(args), "input")
        assert in_fmt == self._in_format, "Invalid input format"
        ret = copy.copy(self._cached_graph[1])
        ret._compose(**{k.name: v for k, v in zip(self._cached_graph[0], args)})
        return _regroup(list(ret), self._out_format)
