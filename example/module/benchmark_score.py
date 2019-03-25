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

"""
Benchmark the scoring performance on various CNNs
"""
#from common import find_mxnet
#from common.util import get_gpus
import mxnet as mx
import mxnet.gluon.model_zoo.vision as models
from importlib import import_module
import logging
import argparse
import time
import numpy as np
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='SymbolAPI-based CNN inference performance benchmark')
parser.add_argument('--network', type=str, default='all',
                                 choices=['all', 'alexnet', 'vgg-16', 'resnetv1-50', 'resnet-50',
                                          'resnet-152', 'inception-bn', 'inception-v3',
                                          'inception-v4', 'inception-resnet-v2', 'mobilenet',
                                          'densenet121', 'squeezenet1.1', 'mlp'])
parser.add_argument('--batch-size', type=int, default=0,
                     help='Batch size to use for benchmarking. Example: 32, 64, 128.'
                          'By default, runs benchmark for batch sizes - 1, 32, 64, 128, 256')

opt = parser.parse_args()

def get_symbol(batch_size, dtype):
    #sym = mx.sym.load("mlp.json")
    data = mx.symbol.Variable('data')
    if dtype == "float16":
        data = mx.sym.Cast(data, np.float16)
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
    if dtype == "float16":
        fc3 = mx.sym.Cast(fc3, np.float32)
    softmax = mx.symbol.SoftmaxOutput(fc3, name = 'softmax')
    return (softmax, [('data', (batch_size, 784))])

def score(network, dev, batch_size, num_batches, dtype):
    # get mod
    sym, data_shape = get_symbol(batch_size, dtype)
    mod = mx.mod.Module(symbol=sym, context=dev)
    mod.bind(for_training     = False,
             inputs_need_grad = False,
             data_shapes      = data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    # get data
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=dev) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    # run
    dry_run = 5                 # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

    # return num images per second
    return num_batches*batch_size/(time.time() - tic)

if __name__ == '__main__':
    if opt.network == 'all':
        networks = ['alexnet', 'vgg-16', 'resnetv1-50', 'resnet-50',
                    'resnet-152', 'inception-bn', 'inception-v3', 
                    'inception-v4', 'inception-resnet-v2', 
                    'mobilenet', 'densenet121', 'squeezenet1.1', 'mlp']
        logging.info('It may take some time to run all models, '
                     'set --network to run a specific one')
    else:
        networks = [opt.network]
    #devs = [mx.gpu(0)] if len(get_gpus()) > 0 else []
    devs = [mx.gpu(0)]
    # Enable USE_MKLDNN for better CPU performance
    #devs.append(mx.cpu())

    if opt.batch_size == 0:
        batch_sizes = [1, 32, 64, 128, 256]
        logging.info('run batchsize [1, 32, 64, 128, 256] by default, '
                     'set --batch-size to run a specific one')
    else:
        batch_sizes = [opt.batch_size]

    for net in networks:
        logging.info('network: %s', net)
        if net in ['densenet121', 'squeezenet1.1']:
            logging.info('network: %s is converted from gluon modelzoo', net)
            logging.info('you can run benchmark/python/gluon/benchmark_gluon.py for more models')
        for d in devs:
            logging.info('device: %s', d)
            logged_fp16_warning = False
            for b in batch_sizes:
                for dtype in ['float32', 'float16']:
                    if d == mx.cpu() and dtype == 'float16':
                        #float16 is not supported on CPU
                        continue
                    elif net in ['inception-bn', 'alexnet'] and dtype == 'float16':
                        if not logged_fp16_warning:
                            logging.info('Model definition for {} does not support float16'.format(net))
                            logged_fp16_warning = True
                    else:
                        speed = score(network=net, dev=d, batch_size=b, num_batches=10, dtype=dtype)
                        logging.info('batch size %2d, dtype %s, images/sec: %f', b, dtype, speed)
