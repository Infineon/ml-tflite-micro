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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_QUANTIZE_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_QUANTIZE_OP_DATA_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/quantize.h"

namespace tflite {
namespace ops {
namespace micro {
namespace quantize {

struct VariantOpData;

#define EVAL_FUNC_DECL(name) \
  TfLiteStatus name(TfLiteContext* context, TfLiteNode* node)

typedef EVAL_FUNC_DECL((*EvalFptr)) ;

EVAL_FUNC_DECL(EvalNNLite);
EVAL_FUNC_DECL(EvalRef);


/** 
 * ****************************************************************************
 * \brief  operation instance data for NNLite accelerated case
 * 
 * ****************************************************************************
 */
struct VariantOpData {
  enum variant_types {
    SW_KERNEL,  NNLITE_KERNEL
  };

  /**
   * \brief operation variant
   * Strict;y we could of course just key on the function pointer but that is 
   * far from easily debuggable etc so we extravagantly invest an extra 4 bytes.
   */
  
  int     variant_type; //*< Either NNLite or reference
  void    *op_data;   //*< Either SingleOpData or reference OpDataQuantizeReference
                      
  EvalFptr eval_function; // Eval function pointer
};

} // namespace quantize
} // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_QUANTIZE_OP_DATA_H_
