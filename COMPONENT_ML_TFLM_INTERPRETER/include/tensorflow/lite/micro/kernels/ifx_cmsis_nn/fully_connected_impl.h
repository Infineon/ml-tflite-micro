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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_FULLY_CONNECTED_IMPL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_FULLY_CONNECTED_IMPL_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/ifx_cmsis_nn/fully_connected_op_data.h"
#include "tensorflow/lite/micro/kernels/ifx_common/kernel_primitives.h"
#include "tensorflow/lite/micro/kernels/ifx_common/offline_prepare_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace fully_connected {

/*
 * Init function is called once at the beginning to initialize kernels and allocate memory.
 */

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
#if TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
  return tflite::micro::nextOfflineOpUserData();
#else
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = context->AllocatePersistentBuffer(context, sizeof(OpData));
  TFLITE_DCHECK(data != nullptr);
  return data;
#endif
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  return data.eval_function(context, node, data, input, filter, bias,
                                  output);
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                        const OpData& data,
                        const TfLiteEvalTensor* input,
                        const TfLiteEvalTensor* filter,
                        const TfLiteEvalTensor* bias,
                        TfLiteEvalTensor* output) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
  const float* bias_data =
      nullptr != bias ? tflite::micro::GetTensorData<float>(bias) : nullptr;
  tflite::reference_ops::FullyConnected(
        FullyConnectedParamsFloat(params->activation),
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<float>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<float>(filter),
        tflite::micro::GetTensorShape(bias),
        bias_data,
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));

  return kTfLiteOk;
}


// Note that the current function names are not ideal at all (this EvalInt8
// function internally calls EvalQuantizedInt8, and there is similar name
// aliasing in the Eval function too). We will be attempting to have a more
// descriptive naming convention but holding off on that for now, since the
// renaming might be coupled with reducing code duplication and some additional
// refactoring.

// @TODO AStevens Infineon 2022-03-28 - this is pointless wth the
// pre-interpreter supporting Prepare as the float variants would still be linked.
// needs to be dropped or have its own INT8-only Prepare vartiant and
// call EvalQuantizedInt8 directly.
 
TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  if (input->type != kTfLiteInt8) {
    TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }

  return data.eval_function(context, node, data, input, filter, bias,
                              output);
}


}  // namespace fully_connected
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_FULLY_CONNECTED_OP_DATA_H_