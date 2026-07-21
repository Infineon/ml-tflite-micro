/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef IFX_TFLM_PRIVATE_IFX_MXNNLITE2_PAD_OP_DATA_H_
#define IFX_TFLM_PRIVATE_IFX_MXNNLITE2_PAD_OP_DATA_H_

#include "tensorflow/lite/micro/kernels/pad.h"
#include "tensorflow/lite/c/common.h"

#include "cy_nn_kernel.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pad {

#define PAD_EVAL_FUNC_DECL(name) \
  TfLiteStatus name(TfLiteContext* context, TfLiteNode* node)

typedef PAD_EVAL_FUNC_DECL((*PadEvalVariantFptr));

#undef PAD_EVAL_FUNC_DECL

struct PadOpData {
  tflite::OpData reference_op_data;
  PadEvalVariantFptr eval_function;
  const float* out_scaling_factor;   // Persistent pointer allocated in Prepare
  const int8_t* identity_weights;    // Persistent unit-valued weights for NNLite DW path
};

}  // namespace pad
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // IFX_TFLM_PRIVATE_IFX_MXNNLITE2_PAD_OP_DATA_H_
