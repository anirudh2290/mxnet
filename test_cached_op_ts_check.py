import mxnet as mx
import time
import argparse

def main():
    res = mx.nd.load("result.params")
    res_expected = mx.nd.load("result_expected.params")

    assert len(res_expected) == len(res), "number of ndarrays differ in result_expected and result"
    print(len(res_expected))
    print(len(res))
    for i in range(len(res_expected)):
        try:
            mx.test_utils.assert_almost_equal(res_expected[i].asnumpy(), res[i].asnumpy(), rtol=1e-5, atol=1e-5)
        except AssertionError:
            time.sleep(20)
            raise
    print("Completed successfully")

if __name__ == "__main__":
    main()
