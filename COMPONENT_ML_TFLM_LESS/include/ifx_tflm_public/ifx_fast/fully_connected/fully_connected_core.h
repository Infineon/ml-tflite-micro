/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef IFX_TFLM_PUBLIC_IFX_FAST_FULLY_CONNECTED_FULLY_CONNECTED_CORE_H_
#define IFX_TFLM_PUBLIC_IFX_FAST_FULLY_CONNECTED_FULLY_CONNECTED_CORE_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "ifx_common/kernel_primitives.h"

#if IFX_DEBUG_LOGGING
#include <iostream>
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace fully_connected {

template<typename T>
class KernelCore;

// No sign extension feature in assumed streamer for now...
template<>
class KernelCore<int8_t>
{
public:
  static void run(int8_t* output, const int8_t* input, const int8_t* weights,
                  const int32_t* sum_of_weights_factor,
                  int32_t sum_of_inputs_factor, int accum_depth,
                  int output_depth, int32_t output_offset,
                  int32_t output_multiplier, int output_shift,
                  int32_t activation_min,
                  int32_t activation_max) {

    int32_t acc;
    for (int out_c = 0; out_c < output_depth; out_c++) {
      // Multiply and accumulate inputs and weights
      acc = *sum_of_weights_factor + sum_of_inputs_factor;
#if IFX_DEBUG_LOGGINGx
      std::cout << "INITIAL ACC " << *sum_of_weights_factor << "+" << sum_of_inputs_factor << std::endl;
#endif
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input[d];
        int32_t weights_val = weights[d];
#if IFX_DEBUG_LOGGINGxx
        std::cout <<  std::dec << input_val << "*" << weights_val << ", ";
#endif
        acc += weights_val *  input_val;
      }

#if IFX_DEBUG_LOGGINGxx
      std::cout << "RAW ACC " << acc << std::endl;
#endif
      // Re-quantize and clamp
      acc =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = ActivationFunctionWithMinMax(acc, activation_min, activation_max);
      *output = static_cast<int8_t>(acc);
      // Increment pointers
      output++;
      sum_of_weights_factor++;
      weights += accum_depth;
    }
  }
};

}  // fully_connected
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif /* IFX_TFLM_PUBLIC_IFX_FAST_FULLY_CONNECTED_FULLY_CONNECTED_CORE_H_ */
