// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_ROCM_MEMORY_POOLS_H_
#define IREE_HAL_DRIVERS_ROCM_MEMORY_POOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"
#include "experimental/rocm/api.h"
#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/rocm_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Retained ROCM memory pools for various allocation types.
typedef struct iree_hal_rocm_memory_pools_t {
  // ROCM context the pools are attached to.
  iree_hal_rocm_context_wrapper_t* context;
  // Used exclusively for DEVICE_LOCAL allocations.
  hipMemPool_t device_local;
  // Used for any host-visible/host-local memory types.
  hipMemPool_t other;

  IREE_STATISTICS(struct {
    iree_atomic_int64_t device_bytes_allocated;
    iree_atomic_int64_t device_bytes_freed;
    iree_atomic_int64_t host_bytes_allocated;
    iree_atomic_int64_t host_bytes_freed;
  } statistics;)
} iree_hal_rocm_memory_pools_t;

// Initializes |out_pools| by configuring new ROCM memory pools.
iree_status_t iree_hal_rocm_memory_pools_initialize(
    iree_hal_rocm_context_wrapper_t* context,
    const iree_hal_rocm_memory_pooling_params_t* pooling_params,
    iree_hal_rocm_memory_pools_t* IREE_RESTRICT out_pools);

// Deinitializes the |pools| and releases the underlying ROCM resources.
void iree_hal_rocm_memory_pools_deinitialize(
    iree_hal_rocm_memory_pools_t* pools);

// Merges statistics information from |pools| into |statistics|.
void iree_hal_rocm_memory_pools_merge_statistics(
    iree_hal_rocm_memory_pools_t* pools,
    iree_hal_allocator_statistics_t* statistics);

// Asynchronously allocates a buffer from an appropriate pool.
// The allocation will be stream-ordered on |stream|.
iree_status_t iree_hal_rocm_memory_pools_alloca(
    iree_hal_rocm_memory_pools_t* pools, hipStream_t stream,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

// Asynchronously deallocates a buffer from its pool.
// The deallocation will be stream-ordered on |stream|.
iree_status_t iree_hal_rocm_memory_pools_dealloca(
    iree_hal_rocm_memory_pools_t* pools, hipStream_t stream,
    iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_ROCM_MEMORY_POOLS_H_
