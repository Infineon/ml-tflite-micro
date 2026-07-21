# ModusToolbox™ Machine Learning TFLiteU Asset

## What's Included?

Refer to the [README.md](./README.md) for a complete description of the ModusToolbox™ Machine Learning TFLiteU asset.

## What Changed?

### v3.4.0

* Fixed group convolution for CM33 with NNLite™ NPU
* Fixed NNLite™ v2 hardware limits handling with CPU fallbacks
* Added Add, Sub, Mul operations and activation functions chunking for NNLite™ NPU
* Updated codebase to TFLM upstream Nov 2025, patch version 15

### v3.3.0

* Fixed FullyConnected (FC) per-channel quantization in NNLite integration
* Updated codebase to TFLM upstream Nov 2025

### v3.2.0

* Validated Multi-Input/Multi-Output (MIMO) model support.
* Validated runtime quantization parameter support per tensor (zero_point, scale); parameters can be defined at both design time and run time.
* Added and validated RELU_0TO1 and RELU_N1TO1 activation function support for NNLite™ NPU.
* Other minor fixes.

### v3.1.0

* Implemented support for PSOC Edge™ capabilities:
    * Cortex-M33 (CPU-based inference)
    * NNLite™ NPU integration (Cortex-M33 core only)
    * EthosU U55 NPU integration (Cortex-M55 core only)
* Updated TensorFlow up to 2.14.1 and set ethos-u-vela up to 3.11
* Added LSTM models support
* Removed dependency to the Y data as they were not used
* Update configuration JSON schema
* Fixed issue when calibration data were limited to the 200 samples

### v2.0.0

* Initial release of TensorFlow Lite Micro inference
* PSOC6™ as a target platform (Cortex-M4)

## Supported Software and Tools

This version of the ModusToolbox™ Machine Learning TFLiteU asset was validated for the compatibility with the following Software and Tools:

| Software and Tools                                      | Version      |
| :---                                                    | :----:       |
| ModusToolbox™ Software Environment                      | 3.6.0        |
| GCC Compiler                                            | 14.2         |
| ARM Compiler 6                                          | 6.22         |
| LLVM**                                                  | 19.1.5       |

**LLVM supported for CM4- and CM33-only, CM33 with NNLite, softFP only.

## More information

For more information, refer to the following documents:

* [ModusToolbox™ Machine Learning Design Support](https://www.infineon.com/cms/en/design-support/tools/sdk/modustoolbox-software/modustoolbox-machine-learning/)
* [ModusToolbox Software](https://www.infineon.com/design-resources/development-tools/sdk/modustoolbox-software)
* [Infineon Technologies AG](https://www.infineon.com)

---
© 2022-2026, Infineon Technologies AG or an affiliate of Infineon Technologies AG.
