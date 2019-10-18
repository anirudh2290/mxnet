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


struct A {
    std::vector<int*> a;
};

static std::mutex api_lock;
int ThreadSafetyTest() {
  std::unique_lock<std::mutex> lock(api_lock);
  A *ret = dmlc::ThreadLocalStore<A>::Get();
  std::vector<int*> tmp_inputs;
  tmp_inputs.reserve(10);
  for (int i = 0; i < 10; ++i) {
      tmp_inputs.push_back(new int(i));
  }
  ret->a.clear();
  ret->a.reserve(10);
  for (int  i = 0; i < 10; ++i) {
      ret->a.push_back(tmp_inputs[i]);
  }
  LOG(INFO) << dmlc::BeginPtr(ret->a);
  return 0;
}

int main(int argc, char const *argv[]) {
    auto func = [&](int num) {
        ThreadSafetyTest();
        };
    std::vector<std::thread> worker_threads(5);
    int count = 0;
    for (auto&& i : worker_threads) {
        i = std::thread(func, count);
        count++;
    }

    for (auto&& i : worker_threads) {
        i.join();
    }
}
