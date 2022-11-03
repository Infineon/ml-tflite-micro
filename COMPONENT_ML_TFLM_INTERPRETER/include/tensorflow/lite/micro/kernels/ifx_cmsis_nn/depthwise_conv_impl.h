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

#include "tensorflow/lite/micro/kernels/depthwise_conv.h"

#include "CMSIS/NN/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

#include "tensorflow/lite/micro/kernels/ifx_common/kernel_primitives.h"
#include "tensorflow/lite/micro/kernels/ifx_cmsis_nn/depthwise_conv_op_data.h"
#if TF_LITE_MICRO_RECORD_OP_USER_DATA
#include "tflite_u_preint/static_data_utils.h"
#endif
#include "tensorflow/lite/micro/kernels/ifx_common/offline_prepare_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace depthwise_conv {

#if TF_LITE_MICRO_RECORD_OP_USER_DATA
  tflite::micro::DefineStaticOpDataHeaders op_data_info(
    "depthwise_conv",
    "#include \"tensorflow/lite/micro/kernels/ifx_cmsis_nn/depthwise_conv_op_data.h\"",
    "OpData"
  );

tflite::micro::CppItems* record_opuserdata(OpData& od, size_t output_depth) {
  auto init = new tflite::micro::CppItems();

  *init 
    << tflite::micro::TfLiteOpDataConvSubStruct(od.reference_op_data, output_depth)
    << od.buffer_idx
    << od.unpacked_weights_buff_idx;

  if (od.custom_sub8bit_packing) {
    *init << tflite::micro::TfLiteCustomSub8BitPackingDetailsStructPtr(
        "custom_sub8bit_packing", *od.custom_sub8bit_packing);
  } else {
    *init << "nullptr";
  }
  *init << od.eval_function;

  return init;
}
#endif


void* Init(TfLiteContext* context, const char* buffer, size_t length) {
#if TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
  return tflite::micro::nextOfflineOpUserData();
#else
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
#endif
}


void EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteDepthwiseConvParams& params,
                             const OpData& data, const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output) {
  cmsis_nn_dw_conv_params dw_conv_params;  
  dw_conv_params.dilation.h = params.dilation_height_factor;
  dw_conv_params.dilation.w = params.dilation_width_factor;

  // in the optimized implementations.
  dw_conv_params.input_offset = -data.reference_op_data.input_zero_point;
  dw_conv_params.output_offset = data.reference_op_data.output_zero_point;
  dw_conv_params.stride.h = params.stride_height;
  dw_conv_params.stride.w = params.stride_width;
  dw_conv_params.padding.h = data.reference_op_data.padding.height;
  dw_conv_params.padding.w = data.reference_op_data.padding.width;
  // TODO(b/130439627): Use calculated value for clamping.
  dw_conv_params.activation.min = std::numeric_limits<int8_t>::min();
  dw_conv_params.activation.max = std::numeric_limits<int8_t>::max();
  dw_conv_params.ch_mult = params.depth_multiplier;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier =
      data.reference_op_data.per_channel_output_multiplier;
  quant_params.shift = data.reference_op_data.per_channel_output_shift;

  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
  RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);

  TFLITE_DCHECK_LE(dw_conv_params.activation.min,
                   dw_conv_params.activation.max);

  const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);

  if (tflite::micro::GetTensorData<int8_t>(bias)) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  cmsis_nn_dims input_dims;
  input_dims.n = batch_size;
  input_dims.h = input_shape.Dims(1);
  input_dims.w = input_shape.Dims(2);
  input_dims.c = input_shape.Dims(3);

  cmsis_nn_dims filter_dims;
  filter_dims.n = filter_shape.Dims(0);
  filter_dims.h = filter_shape.Dims(1);
  filter_dims.w = filter_shape.Dims(2);
  filter_dims.c = output_depth;

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_depth;

  cmsis_nn_dims output_dims;
  output_dims.n = batch_size;
  output_dims.h = output_shape.Dims(1);
  output_dims.w = output_shape.Dims(2);
  output_dims.c = output_depth;

  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  /* 'size' is unused */
  ctx.size = 0;

  if (data.buffer_idx > -1) {
    ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
  }

  const TfLiteCustomSub8BitPackingDetails* custom;
  // if packed
  if (data.custom_sub8bit_packing) {
    custom = data.custom_sub8bit_packing;
    if (custom->sparsity_coding != TfliteSparseType::DENSE) {
      int8_t* scratch_buffer = static_cast<int8_t*>(
          context->GetScratchBuffer(context, data.unpacked_weights_buff_idx));
      // add unpack routine to scratch
      const unsigned int weights_size = filter_shape.FlatSize();
      auto weights_sparse_data = tflite::micro::GetTensorData<int8_t>(filter);

      tflite::ops::micro::UnpackSparseWeights(weights_sparse_data, weights_size,
                                              output_depth, scratch_buffer);
      // override with unpacked
      filter_data = scratch_buffer;
    }
  }

  TFLITE_DCHECK_EQ(
      arm_depthwise_conv_wrapper_s8(
          &ctx, &dw_conv_params, &quant_params, &input_dims,
          tflite::micro::GetTensorData<int8_t>(input), &filter_dims,
          // reloading from scratch buffer
          filter_data, &bias_dims, tflite::micro::GetTensorData<int32_t>(bias),
          &output_dims, tflite::micro::GetTensorData<int8_t>(output)),
      ARM_MATH_SUCCESS);
}


void EvalFloat(TfLiteContext* context, TfLiteNode* node,
                const TfLiteDepthwiseConvParams& params,
                const OpData& data, const TfLiteEvalTensor* input,
                const TfLiteEvalTensor* filter,
                const TfLiteEvalTensor* bias,
                TfLiteEvalTensor* output) {
    tflite::reference_ops::DepthwiseConv(
        DepthwiseConvParamsFloat(params, data.reference_op_data),
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<float>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<float>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<float>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));
}


TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const OpData& data = *(static_cast<OpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthwiseConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kDepthwiseConvBiasTensor)
          : nullptr;

  (*data.eval_function)(context, node, params, data, input, filter, bias, output);

  return kTfLiteOk;

}

}  // namespace depthwise_conv
}  // namespace micro
}  // namespace ops
}  // namespace tflite