/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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


#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"


namespace tflite {
namespace ops {
namespace micro {
namespace conv_common {

/**
 * Slightly patched version of CalculateOpData{DepthwiseConv,Conv} to
 * populate the OpDataConv struct but not crash out if kTfLitePackedAffineQuantization
 * quantization parmaeters are present.
 * 
 */

 template<class CONV_PARAMS, class OP_DATA>
TfLiteStatus CalculateOpData(
    TfLiteContext* context, TfLiteNode* node,
    const CONV_PARAMS& params, int width, int height,
    int filter_width, int filter_height, int out_width, int out_height,
    const TfLiteType data_type, OP_DATA* data, int quant_dimension) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params.padding;
  data->padding = ComputePaddingHeightWidth(
      params.stride_height, params.stride_width, params.dilation_height_factor,
      params.dilation_width_factor, height, width, filter_height, filter_width,
      padding, &out_height, &out_width);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kConvBiasTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    int output_channels = filter->dims->data[quant_dimension];

        // The TFLM kernel utils are of course are unaware of the IFX
    // kTfLitePackedAffineQuantization extension and so will fail validation
    // checks despite this being backward compatible with kTfLiteAffineQuantization. 
    //  So here temporarily set the type to kTfLiteAffineQuantization to avoid having to maintain an
    // evil-twin copy of PopulateConvolutionQuantizationParams
    auto actual_type = filter->quantization.type;
    if (actual_type == kTfLitePackedAffineQuantization) {
      filter->quantization.type = kTfLiteAffineQuantization;
    }

    // The TFLM kernel utils are of course are unaware of the IFX
    // kTfLitePackedAffineQuantization extension and so will fail validation
    // checks despite this being backward compatible with kTfLiteAffineQuantization. 
    //  So here temporarily set the type to kTfLiteAffineQuantization to avoid having to maintain an
    // evil-twin copy of PopulateConvolutionQuantizationParams
    auto res = ::tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params.activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier, data->per_channel_output_shift,
        output_channels);
    filter->quantization.type = actual_type;
    TF_LITE_ENSURE_STATUS(res);
  }

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (has_bias) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}


}  // namespace conv_common
}  // namespace micro
}  // namespace ops
}  // namespace tflite