/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_SVDF_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_SVDF_OP_DATA_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace svdf {

struct OpData;

#define EVAL_FUNC_DECL(name) \
  void name( \
    TfLiteContext* context, TfLiteNode* node,   \
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* weights_feature, \
    const TfLiteEvalTensor* weights_time, const TfLiteEvalTensor* bias, \
    const TfLiteSVDFParams* params, int scratch_tensor_index,   \
    TfLiteEvalTensor* activation_state, TfLiteEvalTensor* output,   \
    const OpData& data)

typedef EVAL_FUNC_DECL((*EvalVariantFptr));

EVAL_FUNC_DECL(EvalFloatSVDF);
EVAL_FUNC_DECL(EvalIntegerSVDF);


#undef EVAL_FUNC_DECL

struct OpData {
  int32_t effective_scale_1_a;
  int32_t effective_scale_2_a;
  // b versions of each scale are kept at int since the numbers are just the
  // shift value - typically between [-32, 32].
  int effective_scale_1_b;
  int effective_scale_2_b;
  int scratch_tensor_index;
  int scratch_output_tensor_index;

  // Cached tensor zero point values for quantized operations.
  int input_zero_point;
  int output_zero_point;

  // Eval function pointer
  EvalVariantFptr eval_function;
};


}  // namespace svdf
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_SVDF_OP_DATA_H_