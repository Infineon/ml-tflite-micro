/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstring>
#include <cstdint>

#include "cy_nn_kernel.h"

namespace tflite {
namespace micro {
namespace nnlite {

#ifndef NNLITE_COPY_THRESHOLD_BYTES
#define NNLITE_COPY_THRESHOLD_BYTES 128
#endif
constexpr size_t kNNLiteThresholdBytes = NNLITE_COPY_THRESHOLD_BYTES;

/**
 * \brief  Accelerated block copy
 * 
 * Falls back to cpu for small copies where accelerator setup overhead would dominate.
 */
inline void accel_memcpy(void* dst, const void* src, size_t bytes) {
  if (bytes < kNNLiteThresholdBytes) {
    // For small tensors, it's more efficient to copy the data with the CPU
    // than setting up the nnlite's many registers...
    memcpy(dst, src, bytes);
  } else {
    Cy_NNLite_Byte_Copy(static_cast<const int8_t *>(src), static_cast<int8_t *>(dst), bytes);
  }
}

/**
 * \brief Repeated Accelerated block copies
 * 
 * Falls back to cpu for small copies where accelerator setup overhead would dominate.
 * 
 * For large copies, we want to use the NNLite accelerator.  However, for some operations 
 * like concatenation, we may have many small copies to do. 
 * In this case, it is more efficient to setup the NNLite once and then trigger subsequent copies with a cheaper "redo" 
 * that doesn't require all the register setup. 
 * 
 * \param dst destination pointer
 * \param src source pointer
 * \param bytes number of bytes to copy for this call
 * \param total_bytes total number of bytes that will be copied across all calls for this sequence of 
 *        copies.
 * \param Initialize accelerator (first call) for sequence of copies 
 * 
 * \note It must of course be guaranteed the the accelerator setup from the initial call is still valid for subsequent calls.  
 * */

inline void accel_multi_memcpy(void* dst, const void* src, size_t bytes, size_t total_bytes, bool initialize) {
  if( total_bytes < kNNLiteThresholdBytes ) {
    // For small total copying ... just copy the data with the CPU
    memcpy(dst, src, bytes);
  } else if (initialize) {
    Cy_NNLite_Byte_Copy(static_cast<const int8_t *>(src), static_cast<int8_t *>(dst), bytes);
  } else if ( bytes < kNNLiteThresholdBytes/4 ) {
    // For really small tensors, even if setup already, it is more efficient to copy the data with the CPU
    // than setting up the nnlite's many registers...
    memcpy(dst, src, bytes);
  } else {
    // For subsequent copies, after initialization, we can assume the nnlite is already setup and just trigger the copy.
    Cy_NNLite_Redo_Byte_Copy(static_cast<const int8_t *>(src), static_cast<int8_t *>(dst), bytes);
  }
}

}  // namespace nnlite
}  // namespace micro
}  // namespace tflite
