/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef IFX_TFLM_PUBLIC_IFX_COMMON_TESTS_LSTM_DATA_H_
#define IFX_TFLM_PUBLIC_IFX_COMMON_TESTS_LSTM_DATA_H_

#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"

namespace tflite {
namespace testing {

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
class LstmNodeContentPackedWeights: public LstmNodeContent<ActivationType, WeightType, BiasType,
          CellType, batch_size, time_steps,
          input_dimension, state_dimension>{
public:

  typedef LstmNodeContent<ActivationType, WeightType, BiasType,
          CellType, batch_size, time_steps,
          input_dimension, state_dimension> baseclass;

  LstmNodeContentPackedWeights(const baseclass& copyfrom) : baseclass(copyfrom) {
    // Ensure TfliteTensor  and TfliteEValTensor elements data pointers are updated to point to 
    // data arrays in this object rather than those in copyfrom which is the state the default
    // copy construct will have set up.
    this->InitializeTensors();
  }

  void SetPackedTensorData(GateData<WeightType, BiasType, input_dimension, state_dimension> &data, int actIndex, int reccIndex) {
    baseclass::tensors_[actIndex].data.data = const_cast<WeightType*>(data.activation_weight);
    baseclass::eval_tensors_[actIndex].data.data = const_cast<WeightType*>(data.activation_weight);
    baseclass::tensors_[reccIndex].data.data = const_cast<WeightType*>(data.recurrent_weight);
    baseclass::eval_tensors_[reccIndex].data.data = const_cast<WeightType*>(data.recurrent_weight);
  }

  void SetTensorQuantizationParam(TfLitePackedAffineQuantization &packedQuantParams) {
    if(packedQuantParams.custom_sub8bit_packing != nullptr) {
      for (unsigned int index = 1; index <= 8; ++index) {
        baseclass::tensors_[index].quantization.type = kTfLitePackedAffineQuantization;
        baseclass::tensors_[index].quantization.params = const_cast<TfLitePackedAffineQuantization*>(&packedQuantParams);
      }
    } else {
      for (unsigned int index = 1; index <= 8; ++index) {
        baseclass::tensors_[index].quantization.type = kTfLiteAffineQuantization;
        baseclass::tensors_[index].quantization.params = nullptr;
      }
    }
  }

  using GateParams = GateData<WeightType, BiasType, input_dimension, state_dimension>;
  /**
   * Try to replaced weights of LSTM gate with packed version
   * 
   * @note Weights are replaced in-place so pointers to the data vectors (e.g. from TfLiteTensor structures) remain valid.
   * @param gateParams    Parameter struct for LSTM gate whose weights are to be packed.
   * @param packing_format   Packing format to be used.
   * @return overflow - Weights could not be packed as packed form larger than dimensions allocated for unpacked data.
   */

  bool packGateWeights(GateParams &gateParams,
                        const TfLiteIntArray &act_weights_dims, const TfLiteIntArray &recc_weights_dims,
                        const TfLiteCustomSub8BitPackingDetails &packing_format) {
  auto packedActWeightsVec = PackedWeights(gateParams.activation_weight, act_weights_dims, &packing_format, 0,  /*num_sops_dims =*/ 0);
  if(input_dimension*state_dimension < packedActWeightsVec.size()) {
    return true;
  } 
  auto packedReccWeightsVec = PackedWeights(gateParams.recurrent_weight, recc_weights_dims, &packing_format, 0, /*num_sops_dims =*/ 0);
  if (state_dimension*state_dimension < packedReccWeightsVec.size()) {
    return true;
  } 
  std::copy(packedActWeightsVec.begin(), packedActWeightsVec.end(), std::begin(gateParams.activation_weight));
  std::copy(packedReccWeightsVec.begin(), packedReccWeightsVec.end(),std::begin(gateParams.recurrent_weight));

  return false;
}


  /**
   * Replace unpacked weights with packed weights
   * 
   * @return Packing aborted as packed weight(s0 would overflow unpacked weight tensors data vector size.  Weight tensor data invalid.
   */

  bool packWeights( const TfLiteCustomSub8BitPackingDetails &packing) {

    TfLiteIntArray* activation_weight_dims = IntArrayFromInts(this->activation_weight_size_);
    TfLiteIntArray* recurrent_weight_dims = IntArrayFromInts(this->recurrent_weight_size_);
    
    // Pack the quantized weights.
    bool overflow = 
      packGateWeights(this->forget_gate_data_, *activation_weight_dims, *recurrent_weight_dims, packing) ||
      packGateWeights(this->input_gate_data_, *activation_weight_dims, *recurrent_weight_dims, packing) ||
      packGateWeights(this->cell_gate_data_, *activation_weight_dims, *recurrent_weight_dims, packing) ||
      packGateWeights(this->output_gate_data_, *activation_weight_dims, *recurrent_weight_dims, packing);

    return overflow;

  }

#if defined(ENABLE_GR_DECODER)

  void SetGRTensorQuantizationParam(TfLitePackedAffineQuantization* packedQuantParams) {
    int index = 0, index1 = 1;
    while(index < 8) {
      if(packedQuantParams[index].custom_sub8bit_packing != nullptr) {
        baseclass::tensors_[index1].quantization.type = kTfLitePackedAffineQuantization;
        baseclass::tensors_[index1].quantization.params = const_cast<TfLitePackedAffineQuantization*>(&packedQuantParams[index]);
      } else {
        baseclass::tensors_[index1].quantization.type = kTfLiteAffineQuantization;
        baseclass::tensors_[index1].quantization.params = nullptr;
      }
      ++index;
      ++index1;
    }
  }


  /**
   * Try to replaced weights of LSTM gate with Golomb-Rice packed version
   * 
   * @note Weights are replaced in-place so pointers to the data vectors (e.g. from TfLiteTensor structures) remain valid.
   * @param gateParams    Parameter struct for LSTM gate whose weights are to be packed.
   * @param gr_mode       Golomb-Rice compression mode
   * @return overflow - Weights could not be packed as packed form larger than dimensions allocated for unpacked data.
   */

   bool packGRGateWeights(GateParams &gateParams, int &index1, int &index2,
        std::mt19937 &random, ifx::mxnnlite::GRCompressionMode gr_mode,
        const int simd_width, TfLiteCustomSub8BitPackingDetails* packing) {
      size_t actWeightSize = input_dimension*state_dimension;
      std::vector<WeightType> act_weights_vec(gateParams.activation_weight, gateParams.activation_weight+actWeightSize);
      auto packedActWeightsVec = GRCompression(random, act_weights_vec, gr_mode, actWeightSize, &packing[index1]);
      if(actWeightSize < packedActWeightsVec.size()) {
        return true;
      }

      // Set packed metadata.
      if( state_dimension >= simd_width)
        packing[index1].compression_type = static_cast<uint8_t>(TfliteCompressionType::FILTERWISE_GR_ENCODING);
      else
        packing[index1].compression_type = static_cast<uint8_t>(TfliteCompressionType::GR_ENCODING);
      std::copy(packedActWeightsVec.begin(), packedActWeightsVec.end(), std::begin(gateParams.activation_weight));

      ++index1;
      size_t reccWeightSize = state_dimension*state_dimension;
      std::vector<WeightType> recc_weights_vec(gateParams.recurrent_weight, gateParams.recurrent_weight+reccWeightSize);
      auto packedReccWeightsVec = GRCompression(random, recc_weights_vec, gr_mode, reccWeightSize, &packing[index2]);
      if (reccWeightSize < packedReccWeightsVec.size()) {
        return true;
      } 

      if( state_dimension >= simd_width)
        packing[index2].compression_type = static_cast<uint8_t>(TfliteCompressionType::FILTERWISE_GR_ENCODING);
      else
        packing[index2].compression_type = static_cast<uint8_t>(TfliteCompressionType::GR_ENCODING);
      std::copy(packedReccWeightsVec.begin(), packedReccWeightsVec.end(),std::begin(gateParams.recurrent_weight));
      ++index2;

      return false;
    }

  /**
   * Replace unpacked weights with Golomb-Rice packed weights
   * 
   * @return Packing aborted as packed weight(s0 would overflow unpacked weight tensors data vector size.  Weight tensor data invalid.
   */

   bool packGRWeights(std::mt19937 &random, ifx::mxnnlite::GRCompressionMode gr_mode,
      const int simd_width, TfLiteCustomSub8BitPackingDetails* packing) { 
    // Pack the quantized weights.
    int index1 = 0, index2 = 4;
    bool overflow = 
      packGRGateWeights(this->input_gate_data_, index1, index2, random, gr_mode,
        simd_width, packing) ||
      packGRGateWeights(this->forget_gate_data_, index1, index2, random, gr_mode,
        simd_width, packing) ||
      packGRGateWeights(this->cell_gate_data_, index1, index2, random, gr_mode,
        simd_width, packing) ||
      packGRGateWeights(this->output_gate_data_, index1, index2, random, gr_mode,
        simd_width, packing);

    return overflow;

  }
#endif // defined(ENABLE_GR_DECODER)

  void filterwiseGateWeights(GateParams &gateParams, uint8_t simdWidth){
    int actWeightsShape[3] = {2, state_dimension, input_dimension};
    TfLiteIntArray* actWeightDims = tflite::testing::IntArrayFromInts(actWeightsShape);
    size_t actWeightSize = input_dimension*state_dimension;
    std::vector<WeightType> act_weights_vec(gateParams.activation_weight, gateParams.activation_weight+actWeightSize);

    int reccWeightsShape[3] = {2, state_dimension, state_dimension};
    TfLiteIntArray* reccWeightDims = tflite::testing::IntArrayFromInts(reccWeightsShape);
    size_t reccWeightSize = state_dimension*state_dimension;
    std::vector<WeightType> recc_weights_vec(gateParams.recurrent_weight, gateParams.recurrent_weight+reccWeightSize);

    auto filterwiseActWeightsVec = filterwiseSlicedWeights(act_weights_vec.data(), *actWeightDims, simdWidth);
    assert(act_weights_vec.size() == filterwiseActWeightsVec.size());
    auto filterwiseReccWeightsVec = filterwiseSlicedWeights(recc_weights_vec.data(), *reccWeightDims, simdWidth);
    assert(recc_weights_vec.size() == filterwiseReccWeightsVec.size());

    std::copy(filterwiseActWeightsVec.begin(), filterwiseActWeightsVec.end(), std::begin(gateParams.activation_weight));
    std::copy(filterwiseReccWeightsVec.begin(), filterwiseReccWeightsVec.end(), std::begin(gateParams.recurrent_weight));

  }

  void filterwiseSlicing(uint8_t simdWidth) {
    filterwiseGateWeights(this->forget_gate_data_, simdWidth);
    filterwiseGateWeights(this->input_gate_data_, simdWidth);
    filterwiseGateWeights(this->cell_gate_data_, simdWidth);
    filterwiseGateWeights(this->output_gate_data_, simdWidth);
  }
};

} //namespace testing
} //namespace tflite

#endif