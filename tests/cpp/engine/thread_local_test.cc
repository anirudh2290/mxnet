/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file engine_shutdown_test.cc
 * \brief Tests engine shutdown for possible crashes
*/
#include <gtest/gtest.h>
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
    std::vector<int> a;
};

static int ThreadSafetyTest(int num, std::vector<int>* tmp_inputs, int**& res) {
    A *ret = dmlc::ThreadLocalStore<A>::Get();
    for (size_t i = num * 10; i < num * 10 + 10; ++i) {
        (*tmp_inputs)[i] = i;
    }
    ret->a.clear();
    ret->a.reserve(10);
    for (size_t i = num * 10; i < num * 10 + 10; ++i) {
        ret->a.push_back((*tmp_inputs)[i]);
    }
    res[num] = dmlc::BeginPtr(ret->a);
    return 0;
}

TEST(ThreadLocal, verify_thread_safety) {
    std::vector<int> tmp_inputs;
    tmp_inputs.resize(50);
    int **outputs;
    *outputs = new int[5];
    for (size_t i = 0; i < 5; i++) {
        outputs[i] = new int[10];
    }
    /*
    for (int i = 0; i < outputs.size(); ++i) {
        *(outputs[i]) = new int[10];
    }
    */

    auto func = [&](int num) {
        ThreadSafetyTest(num, &tmp_inputs, outputs);
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

    for (size_t i = 0; i < 10; i++) {
        LOG(INFO) << outputs[1][i];
    }
}
