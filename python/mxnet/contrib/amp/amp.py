# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
"""Functions for enabling AMP (automatic mixed precision)."""
__all__ = ['init', 'init_trainer', 'scale_loss', 'unscale']

from types import MethodType
from array import array
import ctypes
import logging
import contextlib
import numpy as np

from ... import symbol
from ...context import gpu
from ...symbol import Symbol
from ...symbol import contrib as symbol_contrib
from ... import ndarray
from ...ndarray import NDArray, _DTYPE_NP_TO_MX
from . import lists
from ...gluon import trainer
from ... import base
from ...base import c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf
from ... import optimizer as opt
from .loss_scaler import LossScaler

def _cast_symbol_NDArray(s, dtype):
    float_types = (np.float16, np.float32)
    if isinstance(s, Symbol):
        return symbol.amp_cast(s, dtype=dtype)
    elif isinstance(s, NDArray):
        if (s.dtype != dtype and
                s.dtype in float_types and
                s.context.device_type != 'cpu'):
            return ndarray.amp_cast(s, dtype=dtype)
        else:
            return s
    else:
        return s

def _get_fun_to_wrap(name, module, submodule_dict):
    module_internal = getattr(module, "_internal")
    prefix = base._get_op_name_prefix(name)
    if len(prefix) > 0:
        if prefix != '_random_' or name.endswith('_like'):
            func_name = name[len(prefix):]
            cur_module = submodule_dict[prefix]
        else:
            func_name = name
            cur_module = module_internal
    elif name.startswith('_'):
        func_name = name
        cur_module = module_internal
    else:
        func_name = name
        cur_module = module
    return func_name, cur_module

def _wrap_symbol_functions(module, target_dtype, target_precision_ops=None,
                           conditional_fp32_ops=None, fp32_ops=None):
    def _ndarray_wrapper(f, target_dtype, cond_arg=None):
        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                        kwargs[cond_arg[0]] not in cond_arg[1]):
                    return f(*args, **kwargs)
            new_args = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype), args))
            args = tuple(new_args)
            kwargs = {k: _cast_symbol_NDArray(v, target_dtype) for k, v in kwargs.items()}
            return f(*args, **kwargs)
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    def _symbol_wrapper(f, target_dtype, cond_arg=None):
        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                        kwargs[cond_arg[0]] not in cond_arg[1]):
                    return f(*args, **kwargs)
            sym = f(*args, **kwargs)
            inputs = sym.get_children()
            aux = sym.list_auxiliary_states()
            inputs = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype)
                              if x.name not in aux else x, inputs))
            atomic_sym = sym._gen_atomic_symbol()
            wrapped_sym = atomic_sym(*inputs)
            wrapped_sym._set_attr(name=sym.name)
            return wrapped_sym
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    def _symbol_widest_wrapper(f):
        def _new_fun(*args, **kwargs):
            symbols = []
            is_symbol = False
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, (Symbol, NDArray)):
                    symbols.append((args, i, arg))
                    is_symbol = is_symbol or isinstance(arg, Symbol)
            for k, arg in kwargs.items():
                if isinstance(arg, (Symbol, NDArray)):
                    symbols.append((kwargs, k, arg))
                    is_symbol = is_symbol or isinstance(arg, Symbol)
            if not is_symbol:
                # NDArray case
                widest_type = target_dtype
                for _, _, arg in symbols:
                    if isinstance(arg, NDArray):
                        if arg.dtype == np.float32:
                            widest_type = np.float32
                for arr, index, arg in symbols:
                    if arg.dtype != widest_type and arg.dtype == target_dtype:
                        arr[index] = ndarray.amp_cast(arg, dtype=widest_type)
            else:
                # Symbol case
                sym_to_check = list(map(lambda x: x[2], symbols))
                casted_syms = symbol.amp_multicast(*sym_to_check, num_outputs=len(sym_to_check))
                symbols = list(map(lambda x_y: (x_y[0][0], x_y[0][1], x_y[1]),
                                   zip(symbols, casted_syms)))
                for arr, index, arg in symbols:
                    arr[index] = arg

            return f(*args, **kwargs)
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    _wrapper = _symbol_wrapper if module in (symbol, Symbol, symbol_contrib) else _ndarray_wrapper

    submodule_dict = {}
    for op_name_prefix in base._OP_NAME_PREFIX_LIST:
        submodule_dict[op_name_prefix] =\
                getattr(module, op_name_prefix[1:-1])

    wrap_list = target_precision_ops if target_precision_ops is not None \
                    else lists.symbol.FP16_FUNCS
    for fun_name in wrap_list:
        try:
            fun_name, cur_module = _get_fun_to_wrap(fun_name, module, submodule_dict)
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, target_dtype))
            if cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, target_dtype))
        except AttributeError:
            pass

    wrap_list = fp32_ops if fp32_ops is not None else lists.symbol.FP32_FUNCS
    for fun_name in wrap_list:
        try:
            fun_name, cur_module = _get_fun_to_wrap(fun_name, module, submodule_dict)
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, np.float32))
            if cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, np.float32))
        except AttributeError:
            pass

    wrap_list = conditional_fp32_ops if conditional_fp32_ops is not None \
                    else lists.symbol.CONDITIONAL_FP32_FUNCS
    for fun_name, arg, arg_values in wrap_list:
        try:
            fun_name, cur_module = _get_fun_to_wrap(fun_name, module, submodule_dict)
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, np.float32, (arg, arg_values)))
            if cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, np.float32, (arg, arg_values)))
        except AttributeError:
            pass

    for fun_name in lists.symbol.WIDEST_TYPE_CASTS:
        try:
            fun_name, cur_module = _get_fun_to_wrap(fun_name, module, submodule_dict)
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _symbol_widest_wrapper(f_to_wrap))
            if cur_module == module:
                setattr(module.op, fun_name, _symbol_widest_wrapper(f_to_wrap))
        except AttributeError:
            pass

def _wrap_loss_output_functions(module, ls):
    if module == ndarray:
        def _wrapper(f):
            def _scaling_wrapper(*args, **kwargs):
                if 'grad_scale' in kwargs:
                    kwargs['grad_scale'] = kwargs['grad_scale'] * ls.loss_scale
                else:
                    kwargs['grad_scale'] = ls.loss_scale
                return f(*args, **kwargs)
            _scaling_wrapper.__name__ = f.__name__
            _scaling_wrapper.__module__ = f.__module__
            _scaling_wrapper.__doc__ = f.__doc__
            return _scaling_wrapper
    else:
        def _wrapper(f):
            def _warning_wrapper(*args, **kwargs):
                logging.warning("%s does not support dynamic loss scaling "
                                "in symbolic and hybridized execution.", f.__name__)
                return f(*args, **kwargs)
            _warning_wrapper.__name__ = f.__name__
            _warning_wrapper.__module__ = f.__module__
            _warning_wrapper.__doc__ = f.__doc__
            return _warning_wrapper

    for fun_name in lists.symbol.LOSS_OUTPUT_FUNCTIONS:
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap))
        except AttributeError:
            pass

_amp_initialized = False
_amp_loss_scale_initialized = False
_loss_scaler = None

@contextlib.contextmanager
def scale_loss(loss, optimizer_or_trainer):
    assert optimizer_or_trainer._amp_loss_scaler is not None, \
        'Loss scaler is not initialized, did you forget to call amp.init_trainer()?'
    optimizer_or_trainer._scale = (optimizer_or_trainer._amp_original_scale /
                                   optimizer_or_trainer._amp_loss_scaler.loss_scale)
    if isinstance(loss, (list, tuple)):
        yield [l * optimizer_or_trainer._amp_loss_scaler.loss_scale for l in loss]
    else:
        yield optimizer_or_trainer._amp_loss_scaler.loss_scale * loss

def init(target_dtype='float16', target_precision_ops=None,
         conditional_fp32_ops=None, fp32_ops=None):
    """Initialize AMP (automatic mixed precision).

    This needs to be done before model creation.

    Parameters
    ----------
    target_dtype : {'float16'}
        Target low precision type for AMP. Currently only float16 is supported.
    target_precision_ops : list of string
        Override the list of functions casted to FP16. Entries in this list
        are names of the functions casted to FP16.
    conditional_fp32_ops : list of (string, string, list of string)
        Override the list of functions conditionally casted to FP32. The format
        of the list is (name of the function, name of the parameter, list of
        values of the parameter that make the function be casted to FP32).
    fp32_ops : list of string
        Override the list of functions casted to FP32. Entries in this list
        are names of the functions casted to FP32.
    """
    global _amp_initialized
    global _loss_scaler
    if not _amp_initialized:
        assert target_dtype in ['float16', np.float16], \
               "AMP currently supports only float16 as a target_dtype"
        _amp_initialized = True
        logging.info("Using AMP")
        target_dtype = np.dtype(target_dtype)
        _wrap_symbol_functions(symbol, target_dtype, target_precision_ops,
                               conditional_fp32_ops, fp32_ops)
        _wrap_symbol_functions(ndarray, target_dtype, target_precision_ops,
                               conditional_fp32_ops, fp32_ops)
        _loss_scaler = LossScaler()
        _wrap_loss_output_functions(ndarray, _loss_scaler)
        _wrap_loss_output_functions(symbol, _loss_scaler)

def init_trainer(optimizer_or_trainer):
    """Initialize trainer or optimizer to work with AMP dynamic loss scaling.

    Parameters
    ----------
    optimizer_or_trainer : Optimizer or Trainer
        MXNet Optimizer or Gluon trainer to initialize with AMP
    """
    global _amp_loss_scale_initialized
    global _amp_initialized
    global _loss_scaler
    assert _amp_initialized, "AMP not initialized, did you forget to call amp.init()?"
    if not _amp_loss_scale_initialized:
        _amp_loss_scale_initialized = True
        loss_scaler = _loss_scaler
    else:
        loss_scaler = LossScaler()
    #_wrap_output
    if isinstance(optimizer_or_trainer, trainer.Trainer):
        optimizer_or_trainer._amp_loss_scaler = loss_scaler
        optimizer_or_trainer._amp_original_scale = optimizer_or_trainer._scale
        skip_update = optimizer_or_trainer._amp_loss_scaler.wait_and_update
        optimizer_or_trainer._optimizer.old_update_multi_precision = \
                optimizer_or_trainer._optimizer.update_multi_precision
        def new_update_multi_precision(self, index, weight, grad, state):
            if not skip_update():
                self.old_update_multi_precision(index, weight, grad, state)
        optimizer_or_trainer._optimizer.update_multi_precision = \
            MethodType(new_update_multi_precision, optimizer_or_trainer._optimizer)
        launch_check_overflow = optimizer_or_trainer._amp_loss_scaler.launch_check_overflow
        optimizer_or_trainer._old_update = optimizer_or_trainer._update
        def new_update(self, ignore_stale_grad=False):
            launch_check_overflow(self._params)
            self._old_update(ignore_stale_grad)
        optimizer_or_trainer._update = MethodType(new_update, optimizer_or_trainer)

    elif isinstance(optimizer_or_trainer, opt.Optimizer):
        # TODO(ptredak): make it work with the optimizer
        raise TypeError("AMP is currently only compatible with Gluon Trainer")
    else:
        raise TypeError("optimizer_or_trainer should be a Gluon Trainer or "
                        "an optimizer, instead is %s" % type(optimizer_or_trainer))

def unscale(optimizer_or_trainer):
    """Check and unscale the gradients manually. This function should only be used
    if accessing gradients is necessary, e.g. for gradient clipping.

    Parameters
    ----------
    optimizer_or_trainer : Optimizer or Trainer
        MXNet optimizer or Gluon Trainer used when scaling the gradients
    """
    if isinstance(optimizer_or_trainer, trainer.Trainer):
        valid_grads = [p._grad for p in optimizer_or_trainer._params if p._grad is not None]
        for grads in valid_grads:
            # TODO(ptredak): make a bulked unscale
            for g in grads:
                g[:] *= optimizer_or_trainer._scale
        optimizer_or_trainer._scale = 1.
    elif isinstance(optimizer_or_trainer, opt.Optimizer):
        # TODO(ptredak): make it work with the optimizer
        raise TypeError("AMP is currently only compatible with Gluon Trainer")
    else:
        raise TypeError("optimizer_or_trainer should be a Gluon Trainer or "
                        "an optimizer, instead is %s" % type(optimizer_or_trainer))

def convert_symbol(sym, target_dtype="float16", target_dtype_ops=None,
                   fp32_ops=None, conditional_fp32_ops=None,
                   excluded_sym_names=None, data_names=None):
    """Given a symbol object representing a neural network of data type FP32 and target_dtype,
    add cast layers according to the op lists (target_dtype_ops, fp32_ops,
    conditional_fp32_ops) if provided, otherwise use the default
    lists provided by the framework.

    Parameters
    ----------
    sym : Symbol
        FP32 neural network symbol
    target_dtype : str or numpy, optional defaults to float16
        currently only supports float16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_dtype_ops : list of strs, optional
        Override the list of operator names casted to the target_dtype.
        If None, uses the framework's default list to be casted to target_dtype.
    fp32_ops : list of strs, optional
        Override the list of operator names casted to FP32.
        If None, uses the framework's default list to be casted to FP32.
    conditional_fp32_ops : list of (string, string, list of string), optional
        Override the list of functions to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to FP32)
    excluded_sym_names : list of strs, optional
        A list of strings that represent the names of symbols that users want to exclude
        from being casted to FP16 or FP32.
    data_names : list of strs, optional
        A list of strings that represent input data tensor names to the model
    """
    if target_dtype != "float16":
        raise ValueError("Only target_dtype float16 is supported currently")

    if target_dtype_ops is not None:
        assert isinstance(target_dtype_ops, list), "target_dtype_ops should be a list of strs"
    else:
        target_dtype_ops = lists.symbol.FP16_FUNCS

    if fp32_ops is not None:
        assert isinstance(fp32_ops, list), "fp32_ops should be a list of strs"
    else:
        fp32_ops = lists.symbol.FP32_FUNCS

    common_ops = set(target_dtype_ops) & set(fp32_ops)
    assert len(common_ops) == 0, "Ops cannot be in both FP16 list and FP32 list {}".format(common_ops)

    original_combined_ops = set(lists.symbol.FP16_FUNCS + lists.symbol.FP32_FUNCS)
    original_fp16_fp32_ops = set(lists.symbol.FP16_FP32_FUNCS)
    combined_ops = set(target_dtype_ops + fp32_ops)
    all_fp16_fp32_ops = set(lists.symbol.FP16_FUNCS + lists.symbol.FP32_FUNCS + lists.symbol.FP16_FP32_FUNCS)

    assert combined_ops.issubset(all_fp16_fp32_ops), "Can only choose ops from one of the three lists for fp16_ops and fp32_ops" \
                                                     " 1. amp.list_fp16_ops()" \
                                                     " 2. amp.list_fp32_ops()" \
                                                     " 3. amp.list_fp16_fp32_ops()"

    widest_dtype_ops = lists.symbol.WIDEST_TYPE_CASTS

    if conditional_fp32_ops is not None:
        assert isinstance(conditional_fp32_ops, list) << "conditional_fp32_ops should be a list"
    else:
        conditional_fp32_ops = lists.symbol.CONDITIONAL_FP32_FUNCS

    conditional_op_names = []
    param_names = []
    param_vals = []
    indptr = [0]
    for conditional_fp32_op in conditional_fp32_ops:
        assert isinstance(conditional_fp32_op[0], str) and isinstance(conditional_fp32_op[1], str) \
            and isinstance(conditional_fp32_op[2], list), "conditional_fp32_ops should be a list of (str, str, list of str)"
        param_vals += conditional_fp32_op[2]
        indptr.append(len(param_vals))
        param_names.append(conditional_fp32_op[1])
        conditional_op_names.append(conditional_fp32_op[0])

    if excluded_sym_names is not None:
        assert isinstance(excluded_sym_names, list), "excluded_sym_names should be a list of strs"
    else:
        excluded_sym_names = []

    target_dtype = _DTYPE_NP_TO_MX[np.dtype(target_dtype).type]

    attr_dict = sym.attr_dict()
    if not data_names:
        data_names = []
        for sym_name in sym.list_inputs():
            if not sym_name in attr_dict:
                data_names.append(sym_name)
                continue
            if not "__dtype__" in attr_dict[sym_name]:
                data_names.append(sym_name)
    model_param_names = list(set(sym.list_inputs()) - set(data_names))

    str_keys = []
    sdata = []
    for k in data_names:
        str_keys.append(k)
        sdata.append(0)
    keys = c_str_array(str_keys)

    out = SymbolHandle()
    check_call(_LIB.MXReducePrecisionSymbol(sym.handle,
                                            ctypes.byref(out),
                                            mx_uint(len(sdata)),
                                            c_array_buf(ctypes.c_int, array('i', sdata)),
                                            mx_uint(len(indptr)),
                                            c_array_buf(ctypes.c_int, array('i', indptr)),
                                            ctypes.byref(ctypes.c_int(target_dtype)),
                                            mx_uint(len(target_dtype_ops)),
                                            mx_uint(len(fp32_ops)),
                                            mx_uint(len(widest_dtype_ops)),
                                            mx_uint(len(conditional_op_names)),
                                            mx_uint(len(excluded_sym_names)),
                                            mx_uint(len(model_param_names)),
                                            c_str_array(target_dtype_ops),
                                            c_str_array(fp32_ops),
                                            c_str_array(widest_dtype_ops),
                                            c_str_array(conditional_op_names),
                                            c_str_array(excluded_sym_names),
                                            c_str_array(param_names),
                                            c_str_array(param_vals),
                                            c_str_array(model_param_names),
                                            keys))
    return Symbol(out)

def convert_model(sym, arg_params, aux_params, target_dtype="float16", target_dtype_ops=None,
                  fp32_ops=None, conditional_fp32_ops=None, excluded_sym_names=None):
    """API for converting a model from FP32 model to a mixed precision model.
    MXNet tries to convert the FP32 model to mixed precision model by adding
    cast layers using amp_cast and amp_multicast operators. The decision on
    which cast layer to add is based on hardcoded lists for Automatic Mixed Precision
    in MXNet. These lists can be overridden by the user by providing their own lists
    using : targe_precision_ops, fp32_ops, widest_precision_ops, conditional_fp32_ops

    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    target_dtype : str
        Currently only supports float16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_dtype_ops : list of strs
        Override the list of operator names casted to target_dtype.
        If None, uses the framework's default list to be casted to target dtype.
    fp32_ops : list of strs
        Override the lists of operator names casted to FP32.
        If None, uses the framework's default list to be casted to FP32.
    widest_dtype_ops : list of strs
        A list of op names provided by user which should run in widest precision among its inputs.
        If None, uses the framework's default list of widest_precision_ops.
    conditional_fp32_ops : list of (string, string, list of string)
        Override the list of operators to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to
        fp32)
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being quantized.
    """
    if excluded_sym_names is None:
        excluded_sym_names = []
        if not isinstance(excluded_sym_names, list):
            raise ValueError('excluded_sym_names must be a list of strings representing'
                             ' the names of the symbols that should not be casted,'
                             ' while received type %s' % str(type(excluded_sym_names)))

    if target_dtype != "float16":
        raise ValueError("Only target_dtype float16 is supported currently")
    param_names = arg_params.keys() + aux_params.keys()
    data_names = list(set(sym.list_inputs()) - set(param_names))

    sym = convert_symbol(sym, target_dtype, target_dtype_ops,
                         fp32_ops, conditional_fp32_ops,
                         excluded_sym_names, data_names)
    return sym, arg_params, aux_params

def convert_hybrid_block(block, target_dtype="float16", target_dtype_ops=None,
                         fp32_ops=None, conditional_fp32_ops=None,
                         excluded_sym_names=None, input_names=['data'], ctx=gpu(0)):
    """Given a hybrid block/symbol block representing a FP32 model and a target_dtype,
    return a block with mixed precision support which can be used for inference.

    Parameters
    ----------
    block : HybridBlock or SymbolBlock object
        FP32 HybridBlock or SymbolBlock object
    target_dtype : str or numpy
        currently only supports fp16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_precision_ops : list of strs
        Override the list of operator names casted to target_dtype.
        If None, uses the framework's default list to be casted to FP32.
    conditional_fp32_ops : list of (str, str, list of str)
        Override the list of functions to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to FP32
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being quantized
    """
    from ...gluon import HybridBlock, SymbolBlock
    from ...gluon import block as blk
    if isinstance(block, HybridBlock):
        if isinstance(input_names, str):
            input_names = [input_names]
        inputs, sym = block._cached_graph
        converted_sym = convert_symbol(sym, target_dtype, target_dtype_ops,
                                       fp32_ops, conditional_fp32_ops,
                                       excluded_sym_names)

        arg_names = set(converted_sym.list_arguments())
        aux_names = set(converted_sym.list_auxiliary_states())
        arg_dict = {}
        # collect params
        for name, param in block.collect_params().items():
            if name in arg_names:
                arg_dict['arg:%s'%name] = param._reduce()
            else:
                assert name in aux_names
                arg_dict['aux:%s'%name] = param._reduce()

        ret = SymbolBlock(converted_sym, inputs)

        ret.collect_params().load_dict(arg_dict, ctx=ctx)
        return ret

def list_fp16_ops():
    """Get the default list of FP16 ops for AMP
    """
    return lists.symbol.FP16_FUNCS

def list_fp32_ops():
    """Get the default list of FP32 ops for AMP
    """
    return lists.symbol.FP32_FUNCS

def list_fp16_fp32_ops():
    """Get the default list of ops which run in both FP16 and FP32
    """
    return lists.symbol.FP16_FP32_FUNCS
