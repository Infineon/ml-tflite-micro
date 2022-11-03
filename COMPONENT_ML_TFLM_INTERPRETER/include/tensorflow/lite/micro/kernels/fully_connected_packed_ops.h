/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_FULLY_CONNECTED_PACKED_WEIGHTS_H_
#define TENSORFLOW_LITE_MICRO_FULLY_CONNECTED_PACKED_WEIGHTS_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/c/builtin_op_data.h"


#if IFX_DEBUG_LOGGING
#include <iostream>
#endif

namespace tflite {



//
//  Uint8 Quantized fully connect kernel for < 8-bit packed weights
// "little-endian" format (first weight in LSB) ordering assumed.
//
// TODO Use specializations to handle fast case where dimensions
// allow efficient loop-unroll etc.
// accum_container_depth should really be  a params value
//

template <typename CONTAINER_T, size_t bits_per_item, size_t items_per_container>
void FullyConnectedUint8PackedWeights(
        const FullyConnectedParams& params,
        const RuntimeShape& input_shape, const uint8_t* input_data,
        const RuntimeShape& filter_shape, const CONTAINER_T* filter_data,
        const RuntimeShape& bias_shape, const int32_t* bias_data,
        const RuntimeShape& output_shape, uint8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);


#if IFX_DEBUG_LOGGING
  std::cout << "Packed implementation!: filter_offset = " << std::dec << filter_offset << std::endl;
#endif
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const unsigned int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const unsigned int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  const unsigned int accum_container_depth = (accum_depth + (items_per_container-1u))/items_per_container;
  const int32_t mask = (1<<bits_per_item)-1;
#if IFX_DEBUG_LOGGING
  std::cout << "Packed implementation!: accum-depth = " << std::dec << accum_depth << std::endl;
#endif

  unsigned int final_container_begin = accum_depth-(accum_depth%items_per_container);
  for (int b = 0; b < batches; ++b) {
    for (unsigned int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      const uint8_t *input_vals;
      CONTAINER_T filter_vals;
      unsigned int d = 0;
      unsigned int container = 0;
      for (;;) {
        input_vals = &input_data[b * accum_depth + d];
        filter_vals = filter_data[out_c * accum_container_depth + container];
        // Exit loop once last complete container processed...
        // Next container is setup
        if (d >= final_container_begin)
          break;
        // Unrollable loop!!
        for( unsigned int i = 0; i < items_per_container; ++i) {
            int32_t input_val = input_vals[i] + input_offset;
            int32_t filter_val = (filter_vals & mask) + filter_offset;
#if IFX_DEBUG_LOGGING
            std::cout <<  std::dec << input_val << "*" << filter_val << ", ";
#endif
            filter_vals >>= bits_per_item;
            acc += filter_val * input_val;
        }
        d += items_per_container;
        ++container;
      }
      // Remaining items if accum_depth%items_per_container !=0
      // TODO template params to handle no bias / weight container type
      // aligned cases.

      unsigned int i = 0;
      while( d < accum_depth ) {
          int32_t input_val = input_vals[i] + input_offset;
          int32_t filter_val = (filter_vals & mask) + filter_offset;
#if IFX_DEBUG_LOGGING
          std::cout <<  std::dec << input_val << "*" << filter_val << ", ";
#endif
          filter_vals >>= bits_per_item;
          acc += filter_val * input_val;
          ++d;
          ++i;
      }

#if IFX_DEBUG_LOGGING
      std::cout << " == " << acc << std::endl;
#endif

      if (bias_data) {
        acc += bias_data[out_c];
      }

      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<uint8_t>(acc);
    }

  }
}

template <typename CONTAINER_T, size_t bits_per_item, size_t items_per_container>
void FullyConnectedInt8PackedWeights(
        const FullyConnectedParams& params,
        const RuntimeShape& input_shape, const int8_t* input_data,
        const RuntimeShape& filter_shape, const CONTAINER_T* filter_data,
        const RuntimeShape& bias_shape, const int32_t* bias_data,
        const RuntimeShape& output_shape, int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);


#if IFX_DEBUG_LOGGING
  std::cout << "Packed implementation!: bits_per_item = " << std::dec << bits_per_item << std::endl;
#endif
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const unsigned int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const unsigned int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  const unsigned int accum_container_depth = (accum_depth + (items_per_container-1u))/items_per_container;
  const int32_t mask = (1<<bits_per_item)-1;
#if IFX_DEBUG_LOGGING
  std::cout << "Packed implementation!: accum-depth = " << std::dec << accum_depth << std::endl;
    bool once = false;
#endif

  unsigned int final_container_begin = accum_depth-(accum_depth%items_per_container);
  unsigned int shift_amount = sizeof(int32_t) * 8 - bits_per_item;
  for (int b = 0; b < batches; ++b) {
    for (unsigned int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      const int8_t *input_vals;
      CONTAINER_T filter_vals;
      unsigned int d = 0;
      unsigned int container = 0;
      for (;;) {
        input_vals = &input_data[b * accum_depth + d];
        filter_vals = filter_data[out_c * accum_container_depth + container];
        // Exit loop once last complete container processed...
        // Next container is setup
        if (d >= final_container_begin)
          break;
        // Unrollable loop!!
        for( unsigned int i = 0; i < items_per_container; ++i) {
            int32_t input_val = input_vals[i] + input_offset;
            int32_t filter_raw = (filter_vals & mask);
            int32_t filter_val = (filter_raw << shift_amount) >> shift_amount;
#if IFX_DEBUG_LOGGING
          if( !once ) {
                std::cout <<  std::dec << input_val << "*" << filter_val << ", ";
            }
#endif
            filter_vals >>= bits_per_item;
            acc += filter_val * input_val;
        }
        d += items_per_container;
        ++container;
      }
      // Remaining items if accum_depth%items_per_container !=0
      // TODO template params to handle no bias / weight container type
      // aligned cases.

      unsigned int i = 0;
      while( d < accum_depth ) {
          int32_t input_val = input_vals[i] + input_offset;
          int32_t filter_raw = (filter_vals & mask);
          int32_t filter_val = (filter_raw << shift_amount) >> shift_amount;
#if IFX_DEBUG_LOGGING
        if( !once ) {
          std::cout <<  std::dec << input_val << "*" << filter_val << ", ";
        }
#endif
          filter_vals >>= bits_per_item;
          acc += filter_val * input_val;
          ++d;
          ++i;
      }

      if (bias_data) {
        acc += bias_data[out_c];
      }
#if IFX_DEBUG_LOGGING
      if( !once ) {
        std::cout << "RAW ACC " << acc << std::endl;
      }
      once = true;
#endif
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }

  }
}

template<typename CONTAINER_T>
inline void FullyConnectedUint8SparseWeights(
        const FullyConnectedParams& params,
        const RuntimeShape& input_shape, const uint8_t* input_data,
        const RuntimeShape& filter_shape, const CONTAINER_T* filter_data,
        const RuntimeShape& bias_shape, const int32_t* bias_data,
        const RuntimeShape& output_shape, uint8_t* output_data) {


  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);


#if IFX_DEBUG_LOGGING
  std::cout << "Sparse implementation!: filter_offset = " << std::dec << filter_offset << std::endl;
#endif
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const unsigned int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);  
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  const unsigned int filter_size = filter_shape.FlatSize();
  unsigned int accum_run_lens_size = output_depth*2u;
  unsigned int sparsity_map_size = ((filter_size+7u) / 8u);

  for (int b = 0; b < batches; ++b) {

    const uint8_t *sparsity_map_curbyte_i = filter_data + accum_run_lens_size;
    const uint8_t *nonzero_filter_data_i = filter_data + accum_run_lens_size + sparsity_map_size;
    uint8_t sparsity_map_curbyte = 0;
    uint8_t sparsity_curbit_mask = 0;
    for (unsigned int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        if (sparsity_curbit_mask == 0) {
          sparsity_curbit_mask = 1u;
          sparsity_map_curbyte = *sparsity_map_curbyte_i;
          ++sparsity_map_curbyte_i;
        } 
      
        if (sparsity_map_curbyte & sparsity_curbit_mask) {
          int32_t input_val = input_data[b * accum_depth + d];
          int32_t filter_val = *nonzero_filter_data_i;
#if IFX_DEBUG_LOGGING
          std::cout << (filter_val + filter_offset) << "*" << (input_val + input_offset) << ", ";
#endif
          acc += (filter_val + filter_offset) * (input_val + input_offset);
          ++nonzero_filter_data_i;
        } else {
#if IFX_DEBUG_LOGGING
          int32_t input_val = input_data[b * accum_depth + d];
          std::cout << "z*" << (input_val + input_offset) << ", ";
#endif
        }
        sparsity_curbit_mask <<=  1u;
      }

#if IFX_DEBUG_LOGGING
      std::cout << " == " << acc << std::endl;
#endif
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<uint8_t>(acc);
    }
  }
}

template<typename CONTAINER_T>
inline void FullyConnectedInt8SparseWeights(
        const FullyConnectedParams& params,
        const RuntimeShape& input_shape, const int8_t* input_data,
        const RuntimeShape& filter_shape, const CONTAINER_T* filter_data,
        const RuntimeShape& bias_shape, const int32_t* bias_data,
        const RuntimeShape& output_shape, int8_t* output_data) {

  // TODO Adapt to signed variants
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);


#if IFX_DEBUG_LOGGING
  std::cout << "Sparse implementation!: filter_offset = " << std::dec << 0 << std::endl;
#endif
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const unsigned int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  const unsigned int filter_size = filter_shape.FlatSize();
  unsigned int accum_run_lens_size = output_depth*2u;
  unsigned int sparsity_map_size = ((filter_size+7u) / 8u);

  for (int b = 0; b < batches; ++b) {

    const uint8_t *sparsity_map_curbyte_i = reinterpret_cast<const uint8_t*>(
        filter_data + accum_run_lens_size);
    const int8_t *nonzero_filter_data_i = filter_data + accum_run_lens_size + sparsity_map_size;
    uint8_t sparsity_map_curbyte = 0;
    uint8_t sparsity_curbit_mask = 0;
    for (unsigned int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        if (sparsity_curbit_mask == 0) {
          sparsity_curbit_mask = 1u;
          sparsity_map_curbyte = *sparsity_map_curbyte_i;
          ++sparsity_map_curbyte_i;
        }

        if (sparsity_map_curbyte & sparsity_curbit_mask) {
          int32_t input_val = input_data[b * accum_depth + d];
          int32_t filter_val = *nonzero_filter_data_i;
#if IFX_DEBUG_LOGGING
          std::cout << (filter_val) << "*" << (input_val + input_offset) << ", ";
#endif
          acc += (filter_val) * (input_val + input_offset);
          ++nonzero_filter_data_i;
        } else {
#if IFX_DEBUG_LOGGING
          int32_t input_val = input_data[b * accum_depth + d];
          std::cout << "z*" << (input_val + input_offset) << ", ";
#endif
        }
        sparsity_curbit_mask <<=  1u;
      }

#if IFX_DEBUG_LOGGING
      std::cout << " == " << acc << std::endl;
#endif
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

}  // namespace tflite


#endif  // TENSORFLOW_LITE_MICRO_FULLY_CONNECTED_PACKED_WEIGHTS_H_
