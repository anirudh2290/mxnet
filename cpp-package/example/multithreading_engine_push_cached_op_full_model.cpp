#include <sys/time.h>
#include <time.h>
#include <chrono>
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

double ms_now() {
  double ret;
  struct timeval time;
  gettimeofday(&time, NULL);
  ret = 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
  return ret;
}

int main(int argc, char const *argv[]) {
    int num_threads = atoi(argv[1]);
    std::string input_ctx = std::string(argv[2]);
    mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();
    if (input_ctx == "cpu") {
      ctx = mxnet::cpp::Context::cpu();
    } else {
      ctx = mxnet::cpp::Context::gpu(0);
    }

    Symbol data = Symbol::Variable("data");
    auto out = Symbol::Load("resnet-18-symbol.json");

    std::map<std::string, NDArray> parameters;
    NDArray::Load("resnet-18-0000.params", 0, &parameters);
    int num_inputs = out.ListInputs().size();
    CachedOpHandle hdl = CachedOpHandle();
    std::vector<std::string> flag_keys = {"data_indices", "param_indices"};
    std::string param_indices = "[";
    for (size_t i = 1; i < num_inputs; ++i) {
        param_indices += std::to_string(i);
        param_indices += std::string(", ");
    }
    param_indices += "]";
    std::vector<std::string> flag_vals {"[0]", param_indices};
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
    std::vector<NDArray> data_arr, data_arr2;
    std::vector<NDArray> softmax_arr, softmax_arr2;
    std::vector<NDArray> params;
    for (std::string name : out.ListInputs()) {
      if (name == "arg:data") {
        continue;
      }
      if (parameters.find("arg:" + name) != parameters.end()) {
        params.push_back(parameters["arg:" + name].Copy(ctx));
      } else if (parameters.find("aux:" + name) != parameters.end()) {
        params.push_back(parameters["aux:" + name].Copy(ctx));
      }
    }
    //CHECK(params.size() + 1 == out.ListInputs().size()) << "Please check if softmax_label was missed";
    for (size_t i = 0; i < num_threads; ++i) {
        data_arr.emplace_back(mxnet::cpp::Shape(1, 3, 224, 224), ctx, false, 0);
        data_arr2.emplace_back(mxnet::cpp::Shape(1, 3, 224, 224), ctx, false, 0);
        softmax_arr.emplace_back(mxnet::cpp::Shape(1), ctx, false, 0);
        softmax_arr2.emplace_back(mxnet::cpp::Shape(1), ctx, false, 0);
        int begin = i*1000;
        int end =  begin + 1000;
        mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(data_arr[i]);
        NDArray::WaitAll();
    }

    int ret1 = MXCreateCachedOpEx(out.GetHandle(),
                                  flag_keys.size(),
                                  flag_key_cstrs.data(),
                                  flag_val_cstrs.data(),
                                  &hdl, false);
    if (ret1 < 0) {
       LOG(FATAL) << MXGetLastError();
    }

    std::vector<NDArrayHandle> arr_handles(num_inputs);
    std::vector<NDArrayHandle*> nd_ptrs(num_threads);
    std::vector<NDArray> result_expected(num_threads);

  double ms = ms_now();
    for (size_t i = 0; i < num_threads; ++i) {
    int num_output = 0;

    arr_handles[0] = data_arr[i].GetHandle();
    for (size_t i = 1; i < num_inputs - 1; ++i) {
        arr_handles[i] = params[i - 1].GetHandle();
    }
    arr_handles[num_inputs - 1] = softmax_arr[i].GetHandle();
    const int* stypes;

    int ret4 = MXInvokeCachedOpEx(hdl, num_inputs, arr_handles.data(), &num_output,
                                  &nd_ptrs[i], &stypes);
    if (ret4 < 0) {
        LOG(FATAL) << MXGetLastError();
    }
    }
    NDArray::WaitAll();
    for (size_t i = 0; i < num_threads; ++i) {
        result_expected[i] = NDArray(*nd_ptrs[i]);
    }
  ms = ms_now() - ms;

    LOG(INFO) << "Time for serial inference" << ms;
  ms = ms_now();

    std::vector<NDArrayHandle> arr_handles2(num_inputs);
    std::vector<NDArrayHandle*> nd_ptrs2(num_threads);
    std::vector<NDArray> result_expected2(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
    int num_output = 0;

    arr_handles2[0] = data_arr2[i].GetHandle();
    for (size_t i = 1; i < num_inputs - 1; ++i) {
        arr_handles2[i] = params[i - 1].GetHandle();
    }
    arr_handles2[num_inputs - 1] = softmax_arr2[i].GetHandle();
    const int* stypes;

    int ret4 = MXInvokeCachedOpEx(hdl, num_inputs, arr_handles2.data(), &num_output,
                                  &nd_ptrs2[i], &stypes);
    if (ret4 < 0) {
        LOG(FATAL) << MXGetLastError();
    }
    NDArray::WaitAll();
    result_expected2[i] = NDArray(*nd_ptrs2[i]);
    }
  ms = ms_now() - ms;

    LOG(INFO) << "Time for serial inference" << ms;
    std::string name_orig = "result_expected.params";
    std::string name_orig2 = "result_expected2.params";
    NDArray::Save(name_orig, result_expected);
    NDArray::Save(name_orig2, result_expected2);


    ret1 = MXCreateCachedOpEx(out.GetHandle(),
                              flag_keys.size(),
                              flag_key_cstrs.data(),
                              flag_val_cstrs.data(),
                              &hdl, true);
    if (ret1 < 0) {
       LOG(FATAL) << MXGetLastError();
    }


    ms = ms_now();
    std::vector<NDArrayHandle*> cached_op_handles(num_threads);
    std::vector<NDArrayHandle*> cached_op_handles2(num_threads);
    std::vector<NDArray> res(num_threads);
    std::vector<NDArray> res2(num_threads);
    auto func = [&](int num) {
    double ms = ms_now();
    int num_output = 0;
    const int* stypes;

    std::vector<NDArrayHandle> arr_handles(num_inputs);
    arr_handles[0] = data_arr[num].GetHandle();
    for (size_t i = 1; i < num_inputs - 1; ++i) {
        arr_handles[i] = params[i - 1].GetHandle();
    }
    arr_handles[num_inputs - 1] = softmax_arr[num].GetHandle();

    int ret2 = MXInvokeCachedOpEx(hdl, num_inputs, arr_handles.data(), &num_output,
                                  &(cached_op_handles[num]), &stypes, true);
    if (ret2 < 0) {
        LOG(FATAL) << MXGetLastError();
    }

    res[num] = NDArray(*cached_op_handles[num]);
    res[num].WaitToRead();
    LOG(INFO) << "thread num: " << num << " parallel inference time " << ms_now() - ms;
    double ms2 = ms_now();

    std::vector<NDArrayHandle> arr_handles2(num_inputs);
    arr_handles2[0] = data_arr2[num].GetHandle();
    for (size_t i = 1; i < num_inputs - 1; ++i) {
        arr_handles2[i] = params[i - 1].GetHandle();
    }
    arr_handles2[num_inputs - 1] = softmax_arr2[num].GetHandle();


    int num_output2 = 0;
    const int* stypes2;
    ret2 = MXInvokeCachedOpEx(hdl, num_inputs, arr_handles.data(), &num_output2,
                              &(cached_op_handles2[num]), &stypes2, true);
    if (ret2 < 0) {
        LOG(FATAL) << MXGetLastError();
    }
    res2[num] = NDArray(*cached_op_handles2[num]);
    res2[num].WaitToRead();

    LOG(INFO) << "thread num: " << num << " parallel inference 2 time " << ms_now() - ms2;
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
    ms = ms_now() - ms;
    LOG(INFO) << "Time for parallel inference" << ms;
    /*
    for (size_t i = 0; i < num_threads; ++i) {
      res[i] = NDArray(*cached_op_handles[i]);
    }
    */
    std::string name =
        "result.params";
    std::string name2 = "result2.params";
    NDArray::Save(name, res);
    NDArray::Save(name2, res2);
}
