import os
import mxnet as mx
import numpy as np
from mxnet.base import MXNetError
from common import assertRaises
from mxnet.test_utils import *

TEST_DATA = [
    [
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0]
    ],  [
        [5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0],
        [7.0, 8.0, 9.0],
    ],  [
        [8.0, 9.0, 10.0],
        [9.0, 10.0, 11.0],
        [10.0, 11.0, 12.0]
    ],  [
        [11.0, 12.0, 13.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
]

TEST_LABELS = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 0.0, 0.0]
]

def combined_data():
    labels = [item for sublist in TEST_LABELS for item in sublist]
    vectors = [item for sublist in TEST_DATA for item in sublist]
    merged = [[l] + vs  for l, vs in zip(labels, vectors)]
    return np.array(merged)

def empty_iter():
    data_path = "test.t"
    data_train = mx.io.CSVIter(data_csv=data_path,
                               data_shape=(8,8),
                               batch_size=100)
    for batch in iter(data_train):
        data_train.getdata().asnumpy()

def whitespace_error():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'test.t')
    with open(data_path, 'w') as fout:
        fout.write("         ")
    fout.close()
    data_train = mx.io.CSVIter(data_csv=data_path,
                               data_shape=(1,3),
                               batch_size=3)
    for batch in iter(data_train):
        data_train.getdata().asnumpy()

def test_file_with_strings():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'test.t')
    with open(data_path, mode='w') as fout:
        fout.write("my header, foo, bar\n1,2,3\n3,4,5")
    fout.close()
    data_train = mx.io.CSVIter(data_csv=data_path,
                               data_shape=(1,3),
                               batch_size=3)
    for batch in iter(data_train):
        data_train.getdata().asnumpy()

def test_binary_data_ex():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "test.t")
    with open(data_path, mode="w") as fout:
        fout.write(b'\x07\x08\x07\x07\x08\x07\x07\x08\x07\x07\x08\x07\x07')
    fout.close()
    data_train = mx.io.CSVIter(data_csv=data_path,
                               data_shape=(1,1),
                               batch_size=1)
    for batch in iter(data_train):
        print(data_train.getdata().asnumpy())

def test_csv_encoded_binary_data_ex():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "test.t")
    with open(data_path, mode="w") as fout:
        fout.write(b'\x07\x08\x07,\x07\x08,\x07\x07\x08,\x07\x07\x08\x07\x07')
    fout.close()
    data_train = mx.io.CSVIter(data_csv=data_path,
                               data_shape=(1,4),
                               batch_size=1)
    for batch in iter(data_train):
        print(data_train.getdata().asnumpy())

def test_excel_file():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "excel.csv")
    data_train=mx.io.CSVIter(data_csv=data_path,
                             data_shape=(1, 4),
                             batch_size=5)
    for batch in iter(data_train):
        print(data_train.getdata().asnumpy())

def test_manual_file():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "manual.csv")
    data_train = mx.io.CSVIter(data_csv=data_path,
                               data_shape=(1, 4),
                               batch_size=5)
    for batch in iter(data_train):
        print(data_train.getdata().asnumpy())

def test_atom_file():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "line_endings.csv")
    data_train = mx.io.CSVIter(data_csv=data_path,
                               data_shape=(1, 4),
                               batch_size=5)
    for batch in iter(data_train):
        print(data_train.getdata().asnumpy())

def test_numpy_file():
    data_path = "numpy.csv"
    np.savetxt(data_path,
                  combined_data()[0:10],
                  delimiter=',',
                  fmt='%g')
    data_train = mx.io.CSVIter(data_csv=data_path,
                               data_shape=(1, 4),
                               batch_size=3)
    read_data = []
    for batch in iter(data_train):
        read_data.append(batch.data[0].asnumpy().tolist())
    print(read_data)

def test_float_data():
    csv = "float.csv"
    random_floats = np.random.rand(50, 10).astype(np.float32)
    np.savetxt(csv,
               random_floats,
               delimiter=",",
               fmt="%.20f")
    data_train = mx.io.CSVIter(data_csv=csv, batch_size=10, data_shape=(1, 10))
    read_data = []
    labels_data = []
    for batch in iter(data_train):
        read_data.append(batch.data[0].asnumpy().astype(np.float32).tolist())
        labels_data.append(batch.label[0].asnumpy().astype(np.float32).tolist())

def test_multiple_labels():
    csv = "float.csv"
    label_csv = "float_label.csv"
    random_floats = np.random.rand(50, 10).astype(np.float32)
    random_floats_label = np.random.rand(50, 1).astype(np.int32)
    np.savetxt(csv,
               random_floats,
               delimiter=",",
               fmt="%.20f")
    np.savetxt(label_csv,
               random_floats_label,
               delimiter=",",
               fmt="%.20f")
    data_train = mx.io.CSVIter(data_csv=csv, batch_size=10, data_shape=(1, 10), label_csv=label_csv, label_shape=(1, 1))
    read_data = []
    labels_data = []
    for batch in iter(data_train):
        read_data.append(batch.data[0].asnumpy().astype(np.float32).tolist())
        labels_data.append(batch.label[0].asnumpy().astype(np.float32).tolist())

def smaller_dimensions():
    csv = "small_dimensions.csv"
    np.savetxt(csv,
               combined_data(),
               delimiter=",",
               fmt="%g")
    data_train = mx.io.CSVIter(data_csv=csv, batch_size=3, data_shape=(1, 5))
    read_data = []
    for batch in iter(data_train):
        read_data.append(batch.data[0].asnumpy().astype(np.float32).tolist())

def extra_columns():
    csv = "test_extra_columns.csv"
    np.savetxt(csv,
               combined_data(),
               delimiter=",",
               fmt="%g")
    data_train = mx.io.CSVIter(data_csv=csv, batch_size=3, data_shape=(1,2))
    for batch in iter(data_train):
        batch.data[0].asnumpy()

def test_exc_all():
    assertRaises(MXNetError, empty_iter)
    assertRaises(MXNetError, whitespace_error)
    assertRaises(MXNetError, smaller_dimensions)
    assertRaises(MXNetError, extra_columns)

if __name__ == '__main__':
    import nose
    nose.runmodule()
