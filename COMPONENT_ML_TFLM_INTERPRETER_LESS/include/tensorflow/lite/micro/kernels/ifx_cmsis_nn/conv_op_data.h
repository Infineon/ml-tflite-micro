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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_CONV_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_CONV_OP_DATA_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"

namespace tflite {
namespace ops {
namespace micro {
namespace conv {

struct OpData;

#define EVAL_FUNC_DECL(name) \
  TfLiteStatus name( \
    TfLiteContext* context, TfLiteNode* node, \
    const TfLiteConvParams& params, const OpData& data, \
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* filter, \
    const TfLiteEvalTensor* bias, TfLiteEvalTensor* output, \
    TfLiteEvalTensor* im2col)

typedef EVAL_FUNC_DECL((*EvalVariantFptr));

EVAL_FUNC_DECL(EvalQuantizedPerChannel);
EVAL_FUNC_DECL(EvalFloat);
EVAL_FUNC_DECL(EvalInt16);
EVAL_FUNC_DECL(EvalInt8Ref);

#undef EVAL_FUNC_DECL

struct OpData {
  OpDataConv reference_op_data;

  // Index to buffer for optimizations if applicable.
  int buffer_idx;

  // Weights tensor packing information
  const TfLiteCustomSub8BitPackingDetails *custom_sub8bit_packing;

  //scratch buffer to store unpacked weights for performance variant.
  int unpacked_weights_buff_idx;
  
  // Eval function pointer
  EvalVariantFptr eval_function;
};

}  // namespace conv
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_CONV_OP_DATA_H_