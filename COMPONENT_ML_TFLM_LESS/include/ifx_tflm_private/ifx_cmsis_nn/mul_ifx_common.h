/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/c/common.h"

#include "ifx_cmsis_nn/mul_op_data.h"
#if TF_LITE_MICRO_RECORD_OP_USER_DATA
#include "tflite_u_preint/static_data_utils.h"
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace mul {

constexpr int kInput1Tensor = 0;
constexpr int kInput2Tensor = 1;
constexpr int kOutputTensor = 0;

#if TF_LITE_MICRO_RECORD_OP_USER_DATA

tflite::micro::CppItems* record_opuserdata(const OpData& od);
#endif


TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                   const OpDataMul* data, const TfLiteEvalTensor* input1,
                   const TfLiteEvalTensor* input2, TfLiteEvalTensor* output);

TfLiteStatus EvalQuantizedInt16(TfLiteContext* context, TfLiteNode* node,
                   const OpDataMul* data, const TfLiteEvalTensor* input1,
                   const TfLiteEvalTensor* input2, TfLiteEvalTensor* output);



TfLiteStatus EvalFloatReference(TfLiteContext* context, TfLiteNode* node,
                           const OpDataMul* data,
                           const TfLiteEvalTensor* input1,
                           const TfLiteEvalTensor* input2,
                           TfLiteEvalTensor* output);


}  // namespace mul
}  // namespace micro
}  // namespace ops
}  // namespace tflite
