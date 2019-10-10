import os
import threading
import mxnet as mx
import tempfile
from mxnet.gluon.contrib.block import SymbolBlockThreadSafe
from time import sleep

tmp = tempfile.mkdtemp()
tmpfile = os.path.join(tmp, "resnet34_fp32")
ctx = mx.cpu()

net_fp32 = mx.gluon.model_zoo.vision.resnet34_v2(pretrained=True, ctx=ctx, root=tmp)
net_fp32.hybridize()
data = mx.nd.zeros((1, 3, 224, 224), dtype="float32", ctx=ctx)
net_fp32.forward(data)
net_fp32.export(tmpfile, 0)
net_fp32.forward(data)
sym_file = tmpfile + '-symbol.json'
params_file = tmpfile + '-0000.params'
sm = mx.sym.load(sym_file)
inputs = mx.sym.var('data', dtype='float32')
net_fp32 = SymbolBlockThreadSafe(sm, inputs)
net_fp32.collect_params().load(params_file, ctx=ctx)
fp32_data = mx.nd.zeros((1,3,224,224), dtype='float32', ctx=ctx)
prediction = net_fp32.forward(fp32_data)
mx.nd.waitall()
num_threads = 1
thread_list = []
datas = []
for i in range(num_threads):
    datas.append(mx.nd.random.uniform(i * 1000, i * 1000 + 1000, (1, 3, 224, 224)))
result_expected = {}
'''
for i, data in enumerate(datas):
    result_expected[i] = net_fp32.forward(data)
'''

mx.nd.waitall()

results = {}

def _worker(i, data, net):
    print("Inside worker forward")
    sleep(40)
    results[i] = net.forward(data)

for i in range(num_threads):
    thread = threading.Thread(target=_worker, args=(i, datas[i], net_fp32))
    thread_list.append(thread)
    thread.start()
for i in range(num_threads):
    thread_list[i].join()

mx.nd.waitall()

'''
for key, val in results.items():
    mx.test_utils.assert_almost_equal(results[key].asnumpy(), result_expected[key].asnumpy(), rtol=1e-5, atol=1e-5)
'''

print("Completed successfully")
