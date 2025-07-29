
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_MUL_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_MUL_OP_DATA_H_


#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/mul.h"

namespace tflite {
namespace ops {
namespace micro {
namespace mul {

struct OpData;

#define EVAL_FUNC_DECL(name) \
  void name(TfLiteContext* context, TfLiteNode* node, \
            const OpDataMul* data, \
            const TfLiteEvalTensor* input1, \
            const TfLiteEvalTensor* input2, TfLiteEvalTensor* output)

typedef EVAL_FUNC_DECL((*EvalVariantFptr)) ;

EVAL_FUNC_DECL(EvalQuantized);
EVAL_FUNC_DECL(EvalFloatReference);

#undef EVAL_FUNC_DECL

struct OpData {
#if 0

  int32_t output_activation_min;
  int32_t output_activation_max;

  int32_t output_multiplier;
  int output_shift;

  // Cached tensor zero point values for quantized operations.
  int32_t input1_zero_point;
  int32_t input2_zero_point;
  int32_t output_zero_point;

  float output_activation_min_f32;
  float output_activation_max_f32;
#else
  ::tflite::OpDataMul common;
#endif
  // Eval function pointer
  EvalVariantFptr eval_function;
};


}  // namespace mul
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif // TENSORFLOW_LITE_MICRO_KERNELS_IFX_CMSIS_NN_MUL_OP_DATA_H_
