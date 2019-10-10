import numpy as np
import time
from multiprocessing import Process, current_process
import mxnet as mx

def test():
    sym, arg_params, aux_params = mx.model.load_checkpoint("resnet-18", 0)
    mod = mx.mod.Module(sym, context=mx.gpu())
    mod.bind(data_shapes=[['data', (1, 3, 224, 224)]])
    mod.set_params(arg_params, aux_params)
    mod.forward(mx.io.DataBatch(data=[mx.nd.random.uniform(0, 1, shape=(1, 3, 224, 224), ctx=mx.gpu())]))
    mx.nd.waitall()


if __name__ == "__main__":
    start_time = time.time()
    runs = [Process(target=test) for i in range(22)]  # 1 or 2 or N process is the same error

    for p in runs:
        p.start()
    for p in runs:
        p.join()
    end_time = time.time() - start_time
    print("Time elapsed: " + str(end_time))
