#include <time.h>
#include <dmlc/logging.h>
#include <dmlc/thread_group.h>
#include <dmlc/omp.h>
#include <mxnet/c_api.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <dmlc/timer.h>
#include <cstdio>
#include <thread>
#include <chrono>
#include <vector>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

int main(int argc, char const *argv[]) {
    CHECK(argc == 3) << "One argument expected : num_threads";
    int num_threads = atoi(argv[1]);
    std::string input_ctx = std::string(argv[2]);
    mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();
    mxnet::Context backend_ctx;
    if (input_ctx == "cpu") {
      ctx = mxnet::cpp::Context::cpu();
      backend_ctx = mxnet::Context::CPU(0);
    } else {
      ctx = mxnet::cpp::Context::gpu(0);
      backend_ctx = mxnet::Context::GPU(0);
    }

    Symbol data = Symbol::Variable("data");
    Symbol weight = Symbol::Variable("weight");
    Symbol bias = Symbol::Variable("bias");
    auto out = Operator("Convolution")
            .SetParam("kernel", Shape(2, 2))
            .SetParam("no_bias", false)
            .SetParam("dilate", Shape(1, 1))
            .SetParam("num_group", 1)
            .SetParam("layout", "NCHW")
            .SetParam("stride", Shape(1, 1))
            .SetParam("pad", Shape(0, 0))
            .SetParam("num_filter", 10)
            .SetInput("data", data)
            .SetInput("weight", weight)
            .SetInput("bias", bias)
            .CreateSymbol("fwd");
    CachedOpHandle hdl = CachedOpHandle();
    std::vector<std::string> flag_keys = {"data_indices", "param_indices"};
    std::vector<std::string> flag_vals {"[0]", "[1, 2]"};
    std::vector<const char*> flag_key_cstrs;
    flag_key_cstrs.reserve(flag_keys.size());
    for (size_t i = 0; i < flag_keys.size(); ++i) {
        flag_key_cstrs.emplace_back(flag_keys[i].c_str());
    }
    std::vector<const char*> flag_val_cstrs;
    flag_val_cstrs.reserve(flag_vals.size());
    for (size_t i = 0; i < flag_vals.size(); ++i) {
        flag_val_cstrs.emplace_back(flag_vals[i].c_str());
    }
    int ret = MXCreateCachedOpEx(out.GetHandle(),
                                 flag_keys.size(),
                                 flag_key_cstrs.data(),
                                 flag_val_cstrs.data(),
                                 &hdl);
    if (ret < 0) {
        LOG(INFO) << MXGetLastError();
    }
    LOG(INFO) << "ret code from MXCreateCachedOpEx is " << ret;
    std::vector<mxnet::cpp::NDArray> data_arr;
    std::vector<mxnet::cpp::NDArray> weight_arr;
    std::vector<mxnet::cpp::NDArray> bias_arr;
    std::vector<mxnet::cpp::NDArray> output_arr;
    for (size_t i = 0; i < num_threads; ++i) {
      data_arr.emplace_back(mxnet::cpp::Shape(2, 4, 10, 10), ctx, false, 0);
      weight_arr.emplace_back(mxnet::cpp::Shape(10, 4, 2, 2), ctx, false, 0);
      bias_arr.emplace_back(mxnet::cpp::Shape(10), ctx, false, 0);
      output_arr.emplace_back(mxnet::cpp::Shape(2, 10, 9, 9), ctx, false, 0);
      int begin = 1000 * i;
      int end = begin + 1000;
      mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(data_arr[i]);
      mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(weight_arr[i]);
      mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(bias_arr[i]);
      mxnet::cpp::NDArray::WaitAll();
    }

    std::vector<NDArrayHandle> arr_handles(3);
    std::vector<NDArrayHandle*> nd_ptrs(num_threads);
    std::vector<mxnet::cpp::NDArray> result_expected(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
    int num_output = 0;

    arr_handles[0] = data_arr[i].GetHandle();
    arr_handles[1] = weight_arr[i].GetHandle();
    arr_handles[2] = bias_arr[i].GetHandle();
    const int* stypes;

    int ret4 = MXInvokeCachedOpEx(hdl, 3, arr_handles.data(), &num_output,
                                  &nd_ptrs[i], &stypes);
    if (ret4 < 0) {
        LOG(INFO) << MXGetLastError();
    }
    mxnet::cpp::NDArray::WaitAll();
    result_expected[i] = mxnet::cpp::NDArray(*nd_ptrs[i]);
    }

    std::string name_expected =
        "result_expected.params";
    mxnet::cpp::NDArray::Save(name_expected, result_expected);


    ret = MXCreateCachedOpEx(out.GetHandle(),
                             flag_keys.size(),
                             flag_key_cstrs.data(),
                             flag_val_cstrs.data(),
                             &hdl);
    if (ret < 0) {
        LOG(INFO) << MXGetLastError();
    }


    std::vector<NDArrayHandle*> cached_op_handles(num_threads);
    auto func = [&](int num) {
    int num_output = 0;
    const int* stypes;
    std::vector<NDArrayHandle> arr_handles(3);
    arr_handles[0] = data_arr[num].GetHandle();
    arr_handles[1] = weight_arr[num].GetHandle();
    arr_handles[2] = bias_arr[num].GetHandle();
    int ret2 = MXInvokeCachedOpEx(hdl, 3, arr_handles.data(), &num_output,
                                 &cached_op_handles[num], &stypes);
    if (ret2 < 0) {
        LOG(INFO) << MXGetLastError();
    }
    };
    std::vector<std::thread> worker_threads(num_threads);
    int count = 0;
    for (auto&& i : worker_threads) {
        i = std::thread(func, count);
        count++;
    }
    for (auto&& i : worker_threads) {
        i.join();
    }
    NDArray::WaitAll();
    std::vector<NDArray> res(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        res[i] = NDArray(*cached_op_handles[i]);
    }
    std::string name =
        "result.params";
    NDArray::Save(name, res);
}
