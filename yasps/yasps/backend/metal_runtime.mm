#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

enum YaspsMetalArgumentKind : uint32_t {
  YASPS_METAL_BUFFER = 0,
  YASPS_METAL_BYTES = 1,
};

struct YaspsMetalArgument {
  uint32_t kind;
  const void *data;
  size_t length;
};

struct YaspsMetalDispatch {
  void *pipeline;
  const YaspsMetalArgument *arguments;
  size_t argument_count;
  size_t grid_size;
  size_t threadgroup_size;
};

namespace {

struct Allocation {
  __strong id<MTLBuffer> buffer;
  size_t length;
};

struct Pipeline {
  __strong id<MTLComputePipelineState> state;
  __strong id<MTLFunction> function;
};

std::mutex allocation_mutex;
std::map<uintptr_t, Allocation> allocations;
id<MTLDevice> device;
id<MTLCommandQueue> command_queue;
id<MTLBuffer> dummy_buffer;
std::once_flag runtime_once;
thread_local double last_gpu_time_ms = 0.0;

void copy_error(char *output, size_t output_size, NSString *message) {
  if (output == nullptr || output_size == 0) {
    return;
  }
  const char *utf8 = message == nil ? "Unknown Metal error" : message.UTF8String;
  std::snprintf(output, output_size, "%s", utf8);
}

bool ensure_runtime(char *error, size_t error_size) {
  std::call_once(runtime_once, [] {
    device = MTLCreateSystemDefaultDevice();
    command_queue = [device newCommandQueue];
    dummy_buffer = [device newBufferWithLength:16
                                       options:MTLResourceStorageModeShared];
  });
  if (device == nil || command_queue == nil || dummy_buffer == nil) {
    copy_error(error, error_size, @"No usable Metal device or command queue");
    return false;
  }
  return true;
}

bool resolve_buffer(const void *pointer, id<MTLBuffer> *buffer, NSUInteger *offset) {
  if (pointer == nullptr) {
    *buffer = dummy_buffer;
    *offset = 0;
    return true;
  }

  const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  std::lock_guard<std::mutex> lock(allocation_mutex);
  auto allocation = allocations.upper_bound(address);
  if (allocation == allocations.begin()) {
    return false;
  }
  --allocation;
  const uintptr_t base = allocation->first;
  if (address < base + allocation->second.length) {
    *buffer = allocation->second.buffer;
    *offset = static_cast<NSUInteger>(address - base);
    return true;
  }
  return false;
}

int dispatch_pipeline(Pipeline *pipeline,
                      const YaspsMetalArgument *arguments,
                      size_t argument_count,
                      size_t grid_size,
                      size_t threadgroup_size,
                      bool use_argument_buffer,
                      char *error,
                      size_t error_size) {
  @autoreleasepool {
    if (pipeline == nullptr) {
      copy_error(error, error_size, @"Cannot dispatch a null Metal pipeline");
      return -1;
    }
    if (grid_size == 0) {
      last_gpu_time_ms = 0.0;
      return 0;
    }
    if (!ensure_runtime(error, error_size)) {
      return -1;
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline->state];

    if (use_argument_buffer) {
      id<MTLArgumentEncoder> argument_encoder =
          [pipeline->function newArgumentEncoderWithBufferIndex:0];
      if (argument_encoder == nil) {
        copy_error(error, error_size,
                   @"Metal could not create the generated argument encoder");
        [encoder endEncoding];
        return -1;
      }
      id<MTLBuffer> encoded_arguments =
          [device newBufferWithLength:argument_encoder.encodedLength
                              options:MTLResourceStorageModeShared];
      [argument_encoder setArgumentBuffer:encoded_arguments offset:0];
      for (size_t index = 0; index < argument_count; ++index) {
        const YaspsMetalArgument &argument = arguments[index];
        if (argument.kind == YASPS_METAL_BUFFER) {
          id<MTLBuffer> buffer = nil;
          NSUInteger offset = 0;
          if (!resolve_buffer(argument.data, &buffer, &offset)) {
            copy_error(error, error_size,
                       [NSString stringWithFormat:
                                     @"Argument %zu does not belong to a YASPS "
                                      "Metal allocation",
                                     index]);
            [encoder endEncoding];
            return -1;
          }
          [argument_encoder setBuffer:buffer offset:offset atIndex:index];
          [encoder useResource:buffer
                         usage:MTLResourceUsageRead |
                               MTLResourceUsageWrite];
        } else if (argument.kind == YASPS_METAL_BYTES) {
          void *destination = [argument_encoder constantDataAtIndex:index];
          if (destination == nullptr) {
            copy_error(error, error_size,
                       [NSString stringWithFormat:
                                     @"Argument-buffer constant %zu was not "
                                      "found",
                                     index]);
            [encoder endEncoding];
            return -1;
          }
          std::memcpy(destination, argument.data, argument.length);
        } else {
          copy_error(error, error_size,
                     [NSString stringWithFormat:@"Unknown argument kind %u",
                                                argument.kind]);
          [encoder endEncoding];
          return -1;
        }
      }
      [encoder setBuffer:encoded_arguments offset:0 atIndex:0];
    } else {
      for (size_t index = 0; index < argument_count; ++index) {
        const YaspsMetalArgument &argument = arguments[index];
        if (argument.kind == YASPS_METAL_BUFFER) {
          id<MTLBuffer> buffer = nil;
          NSUInteger offset = 0;
          if (!resolve_buffer(argument.data, &buffer, &offset)) {
            copy_error(error, error_size,
                       [NSString stringWithFormat:
                                     @"Argument %zu does not belong to a YASPS "
                                      "Metal allocation",
                                     index]);
            [encoder endEncoding];
            return -1;
          }
          [encoder setBuffer:buffer offset:offset atIndex:index];
        } else if (argument.kind == YASPS_METAL_BYTES) {
          [encoder setBytes:argument.data
                     length:argument.length
                    atIndex:index];
        } else {
          copy_error(error, error_size,
                     [NSString stringWithFormat:@"Unknown argument kind %u",
                                                argument.kind]);
          [encoder endEncoding];
          return -1;
        }
      }
    }

    const NSUInteger maximum =
        pipeline->state.maxTotalThreadsPerThreadgroup;
    NSUInteger width = threadgroup_size == 0
                           ? pipeline->state.threadExecutionWidth
                           : threadgroup_size;
    width = std::max<NSUInteger>(1, std::min<NSUInteger>(width, maximum));
    width = std::min<NSUInteger>(width, grid_size);
    [encoder dispatchThreads:MTLSizeMake(grid_size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.status == MTLCommandBufferStatusError) {
      copy_error(error, error_size,
                 command_buffer.error.localizedDescription);
      return -1;
    }

    if (command_buffer.GPUEndTime >= command_buffer.GPUStartTime) {
      last_gpu_time_ms =
          (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
    } else {
      last_gpu_time_ms = 0.0;
    }
    return 0;
  }
}

int dispatch_pipeline_batch(const YaspsMetalDispatch *dispatches,
                            size_t dispatch_count,
                            char *error,
                            size_t error_size) {
  @autoreleasepool {
    if (!ensure_runtime(error, error_size)) {
      return -1;
    }
    if (dispatch_count == 0) {
      last_gpu_time_ms = 0.0;
      return 0;
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    for (size_t dispatch_index = 0;
         dispatch_index < dispatch_count;
         ++dispatch_index) {
      const YaspsMetalDispatch &dispatch = dispatches[dispatch_index];
      Pipeline *pipeline = static_cast<Pipeline *>(dispatch.pipeline);
      if (pipeline == nullptr) {
        copy_error(error, error_size,
                   @"Cannot dispatch a null Metal pipeline");
        return -1;
      }
      if (dispatch.grid_size == 0) {
        continue;
      }

      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pipeline->state];
      for (size_t index = 0;
           index < dispatch.argument_count;
           ++index) {
        const YaspsMetalArgument &argument = dispatch.arguments[index];
        if (argument.kind == YASPS_METAL_BUFFER) {
          id<MTLBuffer> buffer = nil;
          NSUInteger offset = 0;
          if (!resolve_buffer(argument.data, &buffer, &offset)) {
            copy_error(
                error,
                error_size,
                [NSString stringWithFormat:
                              @"Batch argument %zu in dispatch %zu does not "
                               "belong to a YASPS Metal allocation",
                              index, dispatch_index]);
            [encoder endEncoding];
            return -1;
          }
          [encoder setBuffer:buffer offset:offset atIndex:index];
        } else if (argument.kind == YASPS_METAL_BYTES) {
          [encoder setBytes:argument.data
                     length:argument.length
                    atIndex:index];
        } else {
          copy_error(
              error,
              error_size,
              [NSString stringWithFormat:
                            @"Unknown argument kind %u in dispatch %zu",
                            argument.kind, dispatch_index]);
          [encoder endEncoding];
          return -1;
        }
      }

      const NSUInteger maximum =
          pipeline->state.maxTotalThreadsPerThreadgroup;
      NSUInteger width =
          dispatch.threadgroup_size == 0
              ? pipeline->state.threadExecutionWidth
              : dispatch.threadgroup_size;
      width = std::max<NSUInteger>(
          1, std::min<NSUInteger>(width, maximum));
      width = std::min<NSUInteger>(width, dispatch.grid_size);
      [encoder dispatchThreads:MTLSizeMake(dispatch.grid_size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
      copy_error(error, error_size,
                 command_buffer.error.localizedDescription);
      return -1;
    }
    if (command_buffer.GPUEndTime >= command_buffer.GPUStartTime) {
      last_gpu_time_ms =
          (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
    } else {
      last_gpu_time_ms = 0.0;
    }
    return 0;
  }
}

}  // namespace

extern "C" {

const char *yasps_metal_device_name() {
  @autoreleasepool {
    if (!ensure_runtime(nullptr, 0)) {
      return nullptr;
    }
    return device.name.UTF8String;
  }
}

uint64_t yasps_metal_current_allocated_size() {
  @autoreleasepool {
    if (!ensure_runtime(nullptr, 0)) {
      return 0;
    }
    return static_cast<uint64_t>(device.currentAllocatedSize);
  }
}

uint64_t yasps_metal_recommended_working_set_size() {
  @autoreleasepool {
    if (!ensure_runtime(nullptr, 0)) {
      return 0;
    }
    return static_cast<uint64_t>(
        device.recommendedMaxWorkingSetSize);
  }
}

void *yasps_metal_alloc(size_t length) {
  @autoreleasepool {
    if (!ensure_runtime(nullptr, 0)) {
      return nullptr;
    }
    const size_t allocated_length = std::max<size_t>(length, 1);
    id<MTLBuffer> buffer =
        [device newBufferWithLength:allocated_length
                           options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      return nullptr;
    }
    void *contents = buffer.contents;
    std::lock_guard<std::mutex> lock(allocation_mutex);
    allocations.emplace(reinterpret_cast<uintptr_t>(contents),
                        Allocation{buffer, allocated_length});
    return contents;
  }
}

void yasps_metal_free(void *pointer) {
  if (pointer == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(allocation_mutex);
  allocations.erase(reinterpret_cast<uintptr_t>(pointer));
}

void yasps_metal_memcpy(void *destination, const void *source, size_t length) {
  if (length != 0) {
    std::memmove(destination, source, length);
  }
}

void *yasps_metal_pipeline_create(const char *metallib_path,
                                  const char *function_name,
                                  char *error,
                                  size_t error_size) {
  @autoreleasepool {
    if (!ensure_runtime(error, error_size)) {
      return nullptr;
    }

    NSString *path = [NSString stringWithUTF8String:metallib_path];
    NSURL *url = [NSURL fileURLWithPath:path];
    NSError *library_error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:url
                                                error:&library_error];
    if (library == nil) {
      copy_error(error, error_size, library_error.localizedDescription);
      return nullptr;
    }

    NSString *name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
      copy_error(error, error_size,
                 [NSString stringWithFormat:@"Metal function '%@' was not found",
                                            name]);
      return nullptr;
    }

    NSError *pipeline_error = nil;
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function
                                              error:&pipeline_error];
    if (state == nil) {
      copy_error(error, error_size, pipeline_error.localizedDescription);
      return nullptr;
    }
    return new Pipeline{state, function};
  }
}

void yasps_metal_pipeline_destroy(void *opaque_pipeline) {
  delete static_cast<Pipeline *>(opaque_pipeline);
}

int yasps_metal_dispatch(void *opaque_pipeline,
                         const YaspsMetalArgument *arguments,
                         size_t argument_count,
                         size_t grid_size,
                         size_t threadgroup_size,
                         char *error,
                         size_t error_size) {
  return dispatch_pipeline(static_cast<Pipeline *>(opaque_pipeline),
                           arguments, argument_count, grid_size,
                           threadgroup_size, false, error, error_size);
}

int yasps_metal_dispatch_argument_buffer(
    void *opaque_pipeline,
    const YaspsMetalArgument *arguments,
    size_t argument_count,
    size_t grid_size,
    size_t threadgroup_size,
    char *error,
    size_t error_size) {
  return dispatch_pipeline(static_cast<Pipeline *>(opaque_pipeline),
                           arguments, argument_count, grid_size,
                           threadgroup_size, true, error, error_size);
}

int yasps_metal_dispatch_batch(
    const YaspsMetalDispatch *dispatches,
    size_t dispatch_count,
    char *error,
    size_t error_size) {
  return dispatch_pipeline_batch(
      dispatches, dispatch_count, error, error_size);
}

double yasps_metal_last_gpu_time_ms() {
  return last_gpu_time_ms;
}

}  // extern "C"
