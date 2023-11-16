// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/rocm_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/rocm/api.h"
#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/direct_command_buffer.h"
#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/event_pool.h"
#include "experimental/rocm/event_semaphore.h"
#include "experimental/rocm/graph_command_buffer.h"
#include "experimental/rocm/nop_executable_cache.h"
#include "experimental/rocm/pending_queue_actions.h"
#include "experimental/rocm/pipeline_layout.h"
#include "experimental/rocm/rocm_allocator.h"
#include "experimental/rocm/rocm_event.h"
#include "experimental/rocm/status_util.h"
#include "experimental/rocm/tracing.h"
#include "hip/hip_runtime_api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/utils/buffer_transfer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

//===----------------------------------------------------------------------===//
// iree_hal_rocm_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_rocm_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the ROCM symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  // Parameters used to control device behavior.
  iree_hal_rocm_device_params_t params;

  hipDevice_t device;

  // TODO: Support multiple device streams.
  // The hipStream_t used to issue device kernels and allocations.
  hipStream_t dispatch_hip_stream;
  // The hipStream_t used to issue host callback functions.
  hipStream_t callback_hip_stream;

  iree_hal_rocm_tracing_context_t* tracing_context;
  iree_hal_rocm_context_wrapper_t context_wrapper;
  iree_hal_allocator_t* device_allocator;

  // Host/device event pools, used for backing semaphore timepoints.
  iree_event_pool_t* host_event_pool;
  iree_hal_rocm_event_pool_t* device_event_pool;
  // Timepoint pools, shared by various semaphores.
  iree_hal_rocm_timepoint_pool_t* timepoint_pool;

  // A queue to order device workloads and relase to the GPU when constraints
  // are met. It buffers submissions and allocations internally before they
  // are ready. This queue couples with HAL semaphores backed by iree_event_t
  // and hipEvent_t objects.
  iree_hal_rocm_pending_queue_actions_t* pending_queue_actions;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Cache of the direct stream command buffer initialized when in stream mode.
  // TODO: have one cached per stream once there are multiple streams.
  iree_hal_command_buffer_t* stream_command_buffer;
} iree_hal_rocm_device_t;

static const iree_hal_device_vtable_t iree_hal_rocm_device_vtable;

static iree_hal_rocm_device_t* iree_hal_rocm_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_device_vtable);
  return (iree_hal_rocm_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_rocm_device_params_initialize(
    iree_hal_rocm_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->command_buffer_mode = IREE_HAL_ROCM_COMMAND_BUFFER_MODE_DIRECT;
  out_params->stream_tracing = false;
}

static void iree_hal_rocm_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Destroy the pending workload queue.
  iree_hal_rocm_pending_queue_actions_destroy(
      (iree_hal_resource_t*)device->pending_queue_actions);

  iree_hal_command_buffer_release(device->stream_command_buffer);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  iree_hal_rocm_tracing_context_free(device->tracing_context);
  // Destroy various pools for synchronization.
  iree_hal_rocm_timepoint_pool_free(device->timepoint_pool);
  iree_hal_rocm_event_pool_free(device->device_event_pool);
  iree_event_pool_free(device->host_event_pool);

  ROCM_IGNORE_ERROR(device->context_wrapper.syms,
                    hipStreamDestroy(device->dispatch_hip_stream));
  ROCM_IGNORE_ERROR(device->context_wrapper.syms,
                    hipStreamDestroy(device->callback_hip_stream));

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_rocm_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_rocm_device_params_t* params, hipDevice_t rocm_device,
    hipStream_t dispatch_stream, hipStream_t callback_stream, hipCtx_t context,
    iree_hal_rocm_dynamic_symbols_t* syms, iree_event_pool_t* host_event_pool,
    iree_hal_rocm_event_pool_t* device_event_pool,
    iree_hal_rocm_timepoint_pool_t* timepoint_pool,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_rocm_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_rocm_device_vtable, &device->resource);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  uint8_t* buffer_ptr = (uint8_t*)device + sizeof(*device);
  buffer_ptr += iree_string_view_append_to_buffer(
      identifier, &device->identifier, (char*)buffer_ptr);
  device->params = *params;
  device->device = rocm_device;
  device->dispatch_hip_stream = dispatch_stream;
  device->callback_hip_stream = callback_stream;
  device->context_wrapper.rocm_context = context;
  device->context_wrapper.rocm_device = rocm_device;
  device->context_wrapper.host_allocator = host_allocator;
  device->context_wrapper.syms = syms;
  device->host_event_pool = host_event_pool;
  device->device_event_pool = device_event_pool;
  device->timepoint_pool = timepoint_pool;

  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);

  iree_status_t status = iree_hal_rocm_pending_queue_actions_create(
      &device->context_wrapper, &device->block_pool,
      &device->pending_queue_actions);

  // Enable tracing for the (currently only) stream - no-op if disabled.
  if (iree_status_is_ok(status) && device->params.stream_tracing) {
    status = iree_hal_rocm_tracing_context_allocate(
        &device->context_wrapper, device->identifier, dispatch_stream,
        &device->block_pool, host_allocator, &device->tracing_context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_allocator_create(&device->context_wrapper,
                                            device->device, dispatch_stream,
                                            &device->device_allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_rocm_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_rocm_device_params_t* params,
    iree_hal_rocm_dynamic_symbols_t* syms, hipDevice_t device,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_TRACE_ZONE_BEGIN(z0);
  hipCtx_t context;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      ROCM_RESULT_TO_STATUS(syms, hipDevicePrimaryCtxRetain(&context, device)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, ROCM_RESULT_TO_STATUS(syms, hipCtxSetCurrent(context)));

  hipStream_t dispatch_stream;
  iree_status_t status = ROCM_RESULT_TO_STATUS(
      syms, hipStreamCreateWithFlags(&dispatch_stream, hipStreamNonBlocking));

  // Create the default callback stream for the device.
  hipStream_t callback_stream;
  if (iree_status_is_ok(status)) {
    status = ROCM_RESULT_TO_STATUS(
        syms, hipStreamCreateWithFlags(&callback_stream, hipStreamNonBlocking));
  }

  iree_event_pool_t* host_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_event_pool_allocate(params->event_pool_capacity,
                                      host_allocator, &host_event_pool);
  }

  iree_hal_rocm_event_pool_t* device_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_event_pool_allocate(
        syms, params->event_pool_capacity, host_allocator, &device_event_pool);
  }

  iree_hal_rocm_timepoint_pool_t* timepoint_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_timepoint_pool_allocate(
        host_event_pool, device_event_pool, params->event_pool_capacity,
        host_allocator, &timepoint_pool);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_device_create_internal(
        driver, identifier, params, device, dispatch_stream, callback_stream,
        context, syms, host_event_pool, device_event_pool, timepoint_pool,
        host_allocator, out_device);
  }
  if (!iree_status_is_ok(status)) {
    if (dispatch_stream) syms->hipStreamDestroy(dispatch_stream);
    if (timepoint_pool) iree_hal_rocm_timepoint_pool_free(timepoint_pool);
    if (device_event_pool) iree_hal_rocm_event_pool_free(device_event_pool);
    if (host_event_pool) iree_event_pool_free(host_event_pool);
    if (callback_stream) syms->hipStreamDestroy(callback_stream);
    if (dispatch_stream) syms->hipStreamDestroy(dispatch_stream);
    syms->hipDevicePrimaryCtxRelease(device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_string_view_t iree_hal_rocm_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_rocm_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->context_wrapper.host_allocator;
}

static iree_hal_allocator_t* iree_hal_rocm_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_rocm_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_rocm_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_rocm_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  // iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.executable.format"))) {
    *out_value =
        iree_string_view_equal(key, iree_make_cstring_view("rocm-hsaco-fb"))
            ? 1
            : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_rocm_device_trim(iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_rocm_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_rocm_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  switch (device->params.command_buffer_mode) {
    case IREE_HAL_ROCM_COMMAND_BUFFER_MODE_DIRECT:
      return iree_hal_rocm_direct_command_buffer_create(
          base_device, &device->context_wrapper, device->tracing_context, mode,
          command_categories, queue_affinity, binding_capacity,
          &device->block_pool, out_command_buffer);
    case IREE_HAL_ROCM_COMMAND_BUFFER_MODE_GRAPH:
      return iree_hal_rocm_graph_command_buffer_create(
          base_device, &device->context_wrapper, mode, command_categories,
          queue_affinity, binding_capacity, &device->block_pool,
          out_command_buffer);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid command buffer mode");
  }
}

static iree_status_t iree_hal_rocm_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_descriptor_set_layout_create(
      &device->context_wrapper, flags, binding_count, bindings,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_rocm_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_event_create(&device->context_wrapper, out_event);
}

static iree_status_t iree_hal_rocm_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_nop_executable_cache_create(
      &device->context_wrapper, identifier, out_executable_cache);
}

static iree_status_t iree_hal_rocm_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(
      queue_affinity, access, handle, iree_hal_device_allocator(base_device),
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_rocm_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_pipeline_layout_create(
      &device->context_wrapper, set_layout_count, set_layouts, push_constants,
      out_pipeline_layout);
}

static iree_status_t iree_hal_rocm_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_event_semaphore_create(
      &device->context_wrapper, initial_value, device->timepoint_pool,
      device->pending_queue_actions, device->context_wrapper.host_allocator,
      out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_rocm_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement ROCM semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_rocm_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO: queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  // TODO: queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_rocm_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_rocm_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  // TODO(raikonenfnu): Once semaphore is implemented wait for semaphores
  // TODO(thomasraoux): implement semaphores - for now this conservatively
  // synchronizes after every submit.
  // stream work with device->stream, we'll change
  // TODO(raikonenfnu): currently run on default/null stream, when cmd buffer
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_rocm_pending_queue_actions_enqueue_execution(
      device->dispatch_hip_stream, device->callback_hip_stream,
      device->pending_queue_actions, wait_semaphore_list, signal_semaphore_list,
      command_buffer_count, command_buffers);
  if (iree_status_is_ok(status)) {
    // Try to advance the pending workload queue.
    status = iree_hal_rocm_pending_queue_actions_issue(
        device->pending_queue_actions);
  }
  iree_hal_rocm_tracing_context_collect(device->tracing_context);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_rocm_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  // Try to advance the pending workload queue.
  iree_status_t status =
      iree_hal_rocm_pending_queue_actions_issue(device->pending_queue_actions);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_rocm_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "waiting multiple semaphores not yet implemented");
}

static iree_status_t iree_hal_rocm_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_rocm_device_vtable = {
    .destroy = iree_hal_rocm_device_destroy,
    .id = iree_hal_rocm_device_id,
    .host_allocator = iree_hal_rocm_device_host_allocator,
    .device_allocator = iree_hal_rocm_device_allocator,
    .replace_device_allocator = iree_hal_rocm_replace_device_allocator,
    .replace_channel_provider = iree_hal_rocm_replace_channel_provider,
    .trim = iree_hal_rocm_device_trim,
    .query_i64 = iree_hal_rocm_device_query_i64,
    .create_channel = iree_hal_rocm_device_create_channel,
    .create_command_buffer = iree_hal_rocm_device_create_command_buffer,
    .create_descriptor_set_layout =
        iree_hal_rocm_device_create_descriptor_set_layout,
    .create_event = iree_hal_rocm_device_create_event,
    .create_executable_cache = iree_hal_rocm_device_create_executable_cache,
    .import_file = iree_hal_rocm_device_import_file,
    .create_pipeline_layout = iree_hal_rocm_device_create_pipeline_layout,
    .create_semaphore = iree_hal_rocm_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_rocm_device_query_semaphore_compatibility,
    .transfer_range = iree_hal_device_submit_transfer_range_and_wait,
    .queue_alloca = iree_hal_rocm_device_queue_alloca,
    .queue_dealloca = iree_hal_rocm_device_queue_dealloca,
    .queue_read = iree_hal_rocm_device_queue_read,
    .queue_write = iree_hal_rocm_device_queue_write,
    .queue_execute = iree_hal_rocm_device_queue_execute,
    .queue_flush = iree_hal_rocm_device_queue_flush,
    .wait_semaphores = iree_hal_rocm_device_wait_semaphores,
    .profiling_begin = iree_hal_rocm_device_profiling_begin,
    .profiling_flush = iree_hal_rocm_device_profiling_flush,
    .profiling_end = iree_hal_rocm_device_profiling_end,
};
