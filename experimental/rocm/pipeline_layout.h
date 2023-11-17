// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_PIPELINE_LAYOUT_H_
#define IREE_HAL_ROCM_PIPELINE_LAYOUT_H_

#include "experimental/rocm/context_wrapper.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The max number of bindings per descriptor set allowed in the ROCM HAL
// implementation.
#define IREE_HAL_ROCM_MAX_DESCRIPTOR_SET_BINDING_COUNT 16

// The max number of descriptor sets allowed in the ROCM HAL implementation.
//
// This depends on the general descriptor set planning in IREE and should adjust
// with it.
#define IREE_HAL_ROCM_MAX_DESCRIPTOR_SET_COUNT 4

// The max number of push constants supported by the ROCM HAL implementation.
#define IREE_HAL_ROCM_MAX_PUSH_CONSTANT_COUNT 64

//===----------------------------------------------------------------------===//
// iree_hal_rocm_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_rocm_descriptor_set_layout_create(
    iree_hal_rocm_context_wrapper_t* context,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

// Return the binding count for the given descriptor set layout.
iree_host_size_t iree_hal_rocm_descriptor_set_layout_binding_count(
    const iree_hal_descriptor_set_layout_t* descriptor_set_layout);

//===----------------------------------------------------------------------===//
// iree_hal_rocm_pipeline_layout_t
//===----------------------------------------------------------------------===//

// Creates the kernel arguments.
iree_status_t iree_hal_rocm_pipeline_layout_create(
    iree_hal_rocm_context_wrapper_t* context, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

// Returns the total number of sets in the given |pipeline_layout|.
iree_host_size_t iree_hal_rocm_pipeline_layout_descriptor_set_count(
    const iree_hal_pipeline_layout_t* pipeline_layout);

// Returns the descriptor set layout of the given |set| in |pipeline_layout|.
const iree_hal_descriptor_set_layout_t*
iree_hal_rocm_pipeline_layout_descriptor_set_layout(
    const iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set);

// Return the base binding index for the given set.
iree_host_size_t iree_hal_rocm_base_binding_index(
    const iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set);

// Returns the total number of descriptor bindings across all sets.
iree_host_size_t iree_hal_rocm_pipeline_layout_total_binding_count(
    const iree_hal_pipeline_layout_t* pipeline_layout);

// Return the base index for push constant data.
iree_host_size_t iree_hal_rocm_push_constant_index(
    const iree_hal_pipeline_layout_t* base_pipeline_layout);

// Return the number of constants in the pipeline layout.
iree_host_size_t iree_hal_rocm_pipeline_layout_num_constants(
    const iree_hal_pipeline_layout_t* base_pipeline_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_PIPELINE_LAYOUT_H_
