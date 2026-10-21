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
#include <cassert>
#include "tensorflow/lite/c/c_api_types.h"
#include "cy_nn_kernel.h"

namespace tflite {
namespace micro {
namespace nnlite {


/**
 * \brief  Accelerated size-unlimited initial block copy
 *
 */
TfLiteStatus accel_initial_memcpy(int8_t *dst, const int8_t* src, uint32_t bytes);

/**
 * \brief  Accelerated size-unlimited follow-on block copy
 *
 * For large copies, we want to use the NNLite accelerator.  However, for some operations
 * like concatenation, we may have many small copies to do.
 * In this case, it is more efficient to setup the NNLite once and then trigger subsequent copies with a cheaper "redo"
 * that doesn't require all the register setup.
 *
 *
 * \note It is of course the responsibility of the caller to ensure the accelerator setup from
 * the copying call (e.g. accel_initial_memcpy) is (still) valid when this function is called.
 */

TfLiteStatus accel_follow_on_memcpy(int8_t* dst, const int8_t* src, uint32_t bytes);

/**
 * \brief  Accelerated block copy
 *
 * Falls back to cpu for small copies where accelerator setup overhead would dominate.
 */
TfLiteStatus accel_memcpy(int8_t* dst, const int8_t* src, uint32_t bytes);


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
 * \param total_bytes total number of bytes that will be copsied across all calls for this sequence of
 *        copies.
 * \param Initialize accelerator (first call) for sequence of copies
 *
 * \note It must of course be guaranteed the the accelerator setup from the initial call is still valid for subsequent calls.
 * */

TfLiteStatus accel_multi_memcpy(int8_t* dst, const int8_t* src, uint32_t bytes, uint32_t total_bytes, bool initialize);

}  // namespace nnlite
}  // namespace micro
}  // namespace tflite
