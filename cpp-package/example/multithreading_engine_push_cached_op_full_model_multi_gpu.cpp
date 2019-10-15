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
    int num_gpus = atoi(argv[1]);
    std::vector<mxnet::cpp::Context> ctx_list;
    for (size_t i = 0; i < num_gpus; ++i) {
        ctx_list.push_back(mxnet::cpp::Context::gpu(i));
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
    std::vector<mxnet::cpp::NDArray> data_arr;
    std::vector<mxnet::cpp::NDArray> softmax_arr;
    std::vector<std::vector<mxnet::cpp::NDArray>> params(num_gpus);
    for (std::string name : out.ListInputs()) {
        if (name == "arg::data") {
            continue;
        }
        if (parameters.find("arg:" + name) != parameters.end()) {
            for (size_t i = 0; i < ctx_list.size(); ++i) {
            params[i].push_back(parameters["arg:" + name].Copy(ctx_list[i]));
            }
        } else if (parameters.find("aux:" + name) != parameters.end()) {
            for (size_t i = 0; i < ctx_list.size(); ++i) {
                params[i].push_back(parameters["aux:" + name].Copy(ctx_list[i]));
            }
        }
    }
    for (size_t i = 0; i < num_gpus; ++i) {
        data_arr.emplace_back(mxnet::cpp::Shape(1, 3, 224, 224), ctx_list[i], false, 0);
        softmax_arr.emplace_back(mxnet::cpp::Shape(1), ctx_list[i], false, 0);
        int begin = i * 1000;
        int end = 1000 + begin;
        mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(data_arr[i]);
        mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(softmax_arr[i]);
        NDArray::WaitAll();
    }
    int ret1 = MXCreateCachedOpEx(out.GetHandle(),
                                  flag_keys.size(),
                                  flag_key_cstrs.data(),
                                  flag_val_cstrs.data(),
                                  &hdl, true);
    if (ret1 < 0) {
        LOG(FATAL) << MXGetLastError();
    }

    std::vector<NDArrayHandle> arr_handles(num_gpus);
    std::vector<NDArrayHandle*> nd_ptrs(num_gpus);
    std::vector<NDArray> result_expected(num_gpus);

    double ms = ms_now();
    for (size_t i = 0; i < num_gpus; ++i) {
        int num_output = 1;
        arr_handles[0] = data_arr[i].GetHandle();
        for (size_t j = 1; j < num_inputs - 1; ++j) {
            arr_handles[j] = params[i][j - 1].GetHandle();
        }
        arr_handles[num_inputs - 1] = softmax_arr[i].GetHandle();
        const int* stypes;
        int ret4 = MXInvokeCachedOpEx(hdl, num_inputs, arr_handles.data(), &num_output,
                                      &nd_ptrs[i], &stypes, true);
        if (ret4 < 0) {
            LOG(FATAL) << MXGetLastError();
        }
        NDArray::WaitAll();
        result_expected[i] = NDArray(*nd_ptrs[i]);
    }
    ms = ms_now() - ms;
    LOG(INFO) << "Time for serial inference: " << ms;
    return 0;

}
