// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/graph_command_buffer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "experimental/rocm/rocm_buffer.h"
#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/native_executable.h"
#include "experimental/rocm/pipeline_layout.h"
#include "experimental/rocm/status_util.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"

#define IREE_HAL_ROCM_MAX_BINDING_COUNT 64
// Kernel arguments contains binding and push constants.
#define IREE_HAL_ROCM_MAX_KERNEL_ARG 128

// Command buffer implementation that directly maps to rocm graph.
// This records the commands on the calling thread without additional threading
// indirection.
typedef struct iree_hal_rocm_graph_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_hal_rocm_context_wrapper_t* context;

  // Maintains a reference to all resources used within the command buffer.
  // Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // Used for when we need ROCM to be able to reference memory as it performs
  // asynchronous operations.
  iree_arena_allocator_t arena;

  hipGraph_t graph;
  hipGraphExec_t exec;

  // Keep track of the last node added to the command buffer as we are currently
  // serializing all the nodes (each node depends on the previous one).
  hipGraphNode_t last_node;

  // Iteratively constructed batch of collective operations.
  iree_hal_collective_batch_t collective_batch;

  int32_t push_constant[IREE_HAL_ROCM_MAX_PUSH_CONSTANT_COUNT];

  // Keep track of the current set of kernel arguments.
  void* current_descriptor[];
} iree_hal_rocm_graph_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_rocm_graph_command_buffer_vtable;

static iree_hal_rocm_graph_command_buffer_t*
iree_hal_rocm_graph_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_graph_command_buffer_vtable);
  return (iree_hal_rocm_graph_command_buffer_t*)base_value;
}

hipGraphExec_t iree_hal_rocm_graph_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  if (!iree_hal_rocm_graph_command_buffer_isa(base_command_buffer)) return NULL;
  iree_hal_rocm_graph_command_buffer_t* command_buffer =
      iree_hal_rocm_graph_command_buffer_cast(base_command_buffer);
  return command_buffer->exec;
}

bool iree_hal_rocm_graph_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_rocm_graph_command_buffer_vtable);
}

