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
#ifndef MXNET_POOL_MEMORY_PROFILER_H_
#define MXNET_POOL_MEMORY_PROFILER_H_

#include <mxnet/storage.h>
#include <string>
#include <vector>
#include "./profiler.h"

namespace mxnet {
namespace storage {

/*!
 * \brief Storage allocation/deallocation profiling via ProfileCounters
 */
class PoolMemoryProfiler {
 public:
  /*!
   * \brief Constructor
   */
  explicit PoolMemoryProfiler(const char *domain_name = "Pool Memory")
    : domain_(domain_name) {
  }

  /*!
   * \brief Called when memory allocated from pool
   * \param handle
   */
  void OnPoolHit(const Storage::Handle &handle, bool hit, size_t size) {
    profiler::Profiler *prof = profiler::Profiler::Get();
    if (prof->IsProfiling(profiler::Profiler::kMemory)) {
      Init();
      const size_t idx = prof->DeviceIndex(handle.ctx.dev_type, handle.ctx.dev_id);
      CHECK_LT(idx, mem_counters_pool_hit_.size()) << "Invalid device index: " << idx;
      if (hit) {
        *mem_counters_pool_hit_[idx] += size;
      } else {
        *mem_counters_pool_miss_[idx] += size;
      }
    }
  }

  void OnAllocate(const Storage::Handle &handle, size_t used_size) {
    profiler::Profiler *prof = profiler::Profiler::Get();
    if (prof->IsProfiling(profiler::Profiler::kMemory)) {
      Init();
      const size_t idx = prof->DeviceIndex(handle.ctx.dev_type, handle.ctx.dev_id);
      CHECK_LT(idx, mem_counters_pool_used_.size()) << "Invalid device index: " << idx;
      *mem_counters_pool_used_[idx] += used_size;
      //*mem_counters_pool_total_[idx] += used_size;
    }
  }


  /*!
   * \brief Called to record amount used on pool
   * \param used_size
   */
  void OnPoolAllocate(const Storage::Handle &handle, size_t used_size) {
    profiler::Profiler *prof = profiler::Profiler::Get();
    if (prof->IsProfiling(profiler::Profiler::kMemory)) {
      Init();
      const size_t idx = prof->DeviceIndex(handle.ctx.dev_type, handle.ctx.dev_id);
      CHECK_LT(idx, mem_counters_pool_used_.size()) << "Invalid device index: " << idx;
      *mem_counters_pool_used_[idx] += used_size;
      *mem_counters_pool_free_[idx] -= used_size;
    }
  }

  /*!
   * \brief Called to record amount freed on pool
   * \param free_size
   */
  void OnPoolFree(const Storage::Handle &handle, size_t free_size) {
    profiler::Profiler *prof = profiler::Profiler::Get();
    if (prof->IsProfiling(profiler::Profiler::kMemory)) {
        Init();
        const size_t idx = prof->DeviceIndex(handle.ctx.dev_type, handle.ctx.dev_id);
        CHECK_LT(idx, mem_counters_pool_free_.size()) << "Invalid device index: " << idx;
        *mem_counters_pool_used_[idx] -= free_size;
        *mem_counters_pool_free_[idx] += free_size;
    }
  }

  void OnFree(const Storage::Handle &handle, size_t free_size) {
      profiler::Profiler *prof = profiler::Profiler::Get();
      if (prof->IsProfiling(profiler::Profiler::kMemory)) {
          Init();
          const size_t idx = prof->DeviceIndex(handle.ctx.dev_type, handle.ctx.dev_id);
          CHECK_LT(idx, mem_counters_pool_free_.size()) << "Invalid device index: " << idx;
          if (*mem_counters_pool_used_[idx] <= free_size) {
            *mem_counters_pool_used_[idx] = 0;
          } else {
            *mem_counters_pool_used_[idx] -= free_size;
          }

          if (*mem_counters_pool_free_[idx] <= free_size) {
            *mem_counters_pool_free_[idx] = 0;
          } else {
            *mem_counters_pool_free_[idx] -= free_size;
          }
          //*mem_counters_pool_total_[idx] = 0;
      }
  }

 private:
  /*!
   * \brief Lazy initialization.  No locks occur except for on the first pass
   * (or colliding parallel first passes)
   */
   void Init() {
     if (mem_counters_pool_hit_.empty()) {
       std::unique_lock<std::mutex> lk(init_mutex_);
       // Check again in case of collision and someone else filled it
       if (mem_counters_pool_hit_.empty()) {
         profiler::Profiler *prof = profiler::Profiler::Get();
         const size_t device_count = prof->DeviceCount();
         mem_counters_pool_hit_.reserve(device_count);
         mem_counters_pool_miss_.reserve(device_count);
         mem_counters_pool_used_.reserve(device_count);
         mem_counters_pool_free_.reserve(device_count);
         mem_counters_pool_total_.reserve(device_count);
         for (size_t i = 0, n = device_count; i < n; ++i) {
           std::string name = "Pool:";
           name += prof->DeviceName(i);
           std::string pool_hits = name + " Total Alloc'ed from Pool";
           std::string pool_misses = name + " Total Alloc'ed with CUDA";
           std::string pool_used = name + " Pool Used";
           std::string pool_free = name + " Pool Free";
           std::string pool_total = name + " Pool Total";
           mem_counters_pool_hit_.emplace_back(
               std::make_shared<profiler::ProfileCounter>(pool_hits.c_str(),
                                                          &domain_));
           mem_counters_pool_miss_.emplace_back(
               std::make_shared<profiler::ProfileCounter>(pool_misses.c_str(),
                                                          &domain_));
           mem_counters_pool_used_.emplace_back(
               std::make_shared<profiler::ProfileCounter>(pool_used.c_str(),
                                                          &domain_));
           mem_counters_pool_free_.emplace_back(
               std::make_shared<profiler::ProfileCounter>(pool_free.c_str(),
                                                          &domain_));
           /*
           mem_counters_pool_total_.emplace_back(
               std::make_shared<profiler::ProfileCounter>(pool_total.c_str(),
                                                          &domain_));
           */
         }
       }
     }
   }

   /*! \brief Domain of the memory profiling information */
   profiler::ProfileDomain domain_;
   /*! \brief Mutex for lazy init */
   std::mutex init_mutex_;
   /*! \brief Constant-sized vector of memory profile counters */
   std::vector<std::shared_ptr<profiler::ProfileCounter>>
       mem_counters_pool_hit_;
   std::vector<std::shared_ptr<profiler::ProfileCounter>>
       mem_counters_pool_miss_;
   std::vector<std::shared_ptr<profiler::ProfileCounter>>
       mem_counters_pool_used_;
   std::vector<std::shared_ptr<profiler::ProfileCounter>>
       mem_counters_pool_free_;
   std::vector<std::shared_ptr<profiler::ProfileCounter>>
       mem_counters_pool_total_;
};

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_PROFILER_STORAGE_PROFILER_H_
