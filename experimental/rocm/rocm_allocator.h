// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_ALLOCATOR_H_
#define IREE_HAL_ROCM_ALLOCATOR_H_

#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/memory_pools.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a ROCM memory allocator.
// |device| and |stream| will be used for management operations.
// |pools| provides memory pools that may be shared across multiple allocators
// and the pointer must remain valid for the lifetime of the allocator.
iree_status_t iree_hal_rocm_allocator_create(
    iree_hal_rocm_context_wrapper_t* context, hipDevice_t device,
    hipStream_t stream, iree_hal_rocm_memory_pools_t* pools,
    iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_ALLOCATOR_H_
