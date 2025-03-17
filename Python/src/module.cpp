//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

PYBIND11_MODULE(pydirectml, module)
{
    module.doc() = "Python binding for DirectML";

    // Enumerations
    //
    py::enum_<DML_TENSOR_DATA_TYPE>(module, "TensorDataType")
        .value("UNKNOWN", DML_TENSOR_DATA_TYPE_UNKNOWN)
        .value("FLOAT32", DML_TENSOR_DATA_TYPE_FLOAT32)
        .value("FLOAT16", DML_TENSOR_DATA_TYPE_FLOAT16)
        .value("UINT32", DML_TENSOR_DATA_TYPE_UINT32)
        .value("UINT16", DML_TENSOR_DATA_TYPE_UINT16)
        .value("UINT8", DML_TENSOR_DATA_TYPE_UINT8)
        .value("INT32", DML_TENSOR_DATA_TYPE_INT32)
        .value("INT16", DML_TENSOR_DATA_TYPE_INT16)
        .value("INT8", DML_TENSOR_DATA_TYPE_INT8)
        .value("FLOAT64", DML_TENSOR_DATA_TYPE_FLOAT64)
        .value("UINT64", DML_TENSOR_DATA_TYPE_UINT64)
        .value("INT64", DML_TENSOR_DATA_TYPE_INT64)
        .export_values();

    py::enum_<DML_TENSOR_FLAGS>(module, "TensorFlags")
        .value("NONE", DML_TENSOR_FLAG_NONE)
        .value("OWNED_BY_DML", DML_TENSOR_FLAG_OWNED_BY_DML)
        .export_values();

    py::enum_<DML_MATRIX_TRANSFORM>(module, "MatrixTransform")
        .value("NONE", DML_MATRIX_TRANSFORM_NONE)
        .value("TRANSPOSE", DML_MATRIX_TRANSFORM_TRANSPOSE)
        .export_values();

    py::enum_<DML_RECURRENT_NETWORK_DIRECTION>(module, "RecurrentNetworkDirection")
        .value("FORWARD", DML_RECURRENT_NETWORK_DIRECTION_FORWARD)
        .value("BACKWARD", DML_RECURRENT_NETWORK_DIRECTION_BACKWARD)
        .value("BIDIRECTIONAL", DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL)
        .export_values();

    py::enum_<dml::GRUOutputOptions>(module, "OutputOptions", py::arithmetic())
        .value("Both", dml::GRUOutputOptions::Both)
        .value("Sequence", dml::GRUOutputOptions::Sequence)
        .value("Single", dml::GRUOutputOptions::Single);

    py::enum_<DML_CONVOLUTION_MODE>(module, "ConvolutionMode")
        .value("CONVOLUTION", DML_CONVOLUTION_MODE_CONVOLUTION)
        .value("CROSS_CORRELATION", DML_CONVOLUTION_MODE_CROSS_CORRELATION)
        .export_values();

    py::enum_<DML_CONVOLUTION_DIRECTION>(module, "ConvolutionDirection")
        .value("FORWARD", DML_CONVOLUTION_DIRECTION_FORWARD)
        .value("BACKWARD", DML_CONVOLUTION_DIRECTION_BACKWARD)
        .export_values();

    py::enum_<DML_INTERPOLATION_MODE>(module, "InterpolationMode")
        .value("NEAREST_NEIGHBOR", DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR)
        .value("LINEAR", DML_INTERPOLATION_MODE_LINEAR)
        .export_values();

    py::enum_<DML_PADDING_MODE>(module, "PaddingMode")
        .value("CONSTANT", DML_PADDING_MODE_CONSTANT)
        .value("EDGE", DML_PADDING_MODE_EDGE)
        .value("REFLECTION", DML_PADDING_MODE_REFLECTION)
        .export_values();

    py::enum_<DML_EXECUTION_FLAGS>(module, "ExecutionFlags", py::arithmetic())
        .value("NONE", DML_EXECUTION_FLAG_NONE)
        .value("ALLOW_HALF_PRECISION_COMPUTATION", DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION)
        .value("DISABLE_META_COMMANDS", DML_EXECUTION_FLAG_DISABLE_META_COMMANDS)
        .value("DESCRIPTORS_VOLATILE", DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE)
        .export_values();

    py::enum_<DML_OPERATOR_TYPE>(module, "OperatorType")
        .value("INVALID", DML_OPERATOR_INVALID)
        .value("ELEMENT_WISE_IDENTITY", DML_OPERATOR_ELEMENT_WISE_IDENTITY)
        .value("ELEMENT_WISE_ABS", DML_OPERATOR_ELEMENT_WISE_ABS)
        .value("ELEMENT_WISE_ACOS", DML_OPERATOR_ELEMENT_WISE_ACOS)
        .value("ELEMENT_WISE_ADD", DML_OPERATOR_ELEMENT_WISE_ADD)
        .value("ELEMENT_WISE_ASIN", DML_OPERATOR_ELEMENT_WISE_ASIN)
        .value("ELEMENT_WISE_ATAN", DML_OPERATOR_ELEMENT_WISE_ATAN)
        .value("ELEMENT_WISE_CEIL", DML_OPERATOR_ELEMENT_WISE_CEIL)
        .value("ELEMENT_WISE_CLIP", DML_OPERATOR_ELEMENT_WISE_CLIP)
        .value("ELEMENT_WISE_COS", DML_OPERATOR_ELEMENT_WISE_COS)
        .value("ELEMENT_WISE_DIVIDE", DML_OPERATOR_ELEMENT_WISE_DIVIDE)
        .value("ELEMENT_WISE_EXP", DML_OPERATOR_ELEMENT_WISE_EXP)
        .value("ELEMENT_WISE_FLOOR", DML_OPERATOR_ELEMENT_WISE_FLOOR)
        .value("ELEMENT_WISE_LOG", DML_OPERATOR_ELEMENT_WISE_LOG)
        .value("ELEMENT_WISE_LOGICAL_AND", DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND)
        .value("ELEMENT_WISE_LOGICAL_EQUALS", DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS)
        .value("ELEMENT_WISE_LOGICAL_GREATER_THAN", DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN)
        .value("ELEMENT_WISE_LOGICAL_LESS_THAN", DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN)
        .value("ELEMENT_WISE_LOGICAL_NOT", DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT)
        .value("ELEMENT_WISE_LOGICAL_OR", DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR)
        .value("ELEMENT_WISE_LOGICAL_XOR", DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR)
        .value("ELEMENT_WISE_MAX", DML_OPERATOR_ELEMENT_WISE_MAX)
        .value("ELEMENT_WISE_MEAN", DML_OPERATOR_ELEMENT_WISE_MEAN)
        .value("ELEMENT_WISE_MIN", DML_OPERATOR_ELEMENT_WISE_MIN)
        .value("ELEMENT_WISE_MULTIPLY", DML_OPERATOR_ELEMENT_WISE_MULTIPLY)
        .value("ELEMENT_WISE_POW", DML_OPERATOR_ELEMENT_WISE_POW)
        .value("ELEMENT_WISE_CONSTANT_POW", DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW)
        .value("ELEMENT_WISE_RECIP", DML_OPERATOR_ELEMENT_WISE_RECIP)
        .value("ELEMENT_WISE_SIN", DML_OPERATOR_ELEMENT_WISE_SIN)
        .value("ELEMENT_WISE_SQRT", DML_OPERATOR_ELEMENT_WISE_SQRT)
        .value("ELEMENT_WISE_SUBTRACT", DML_OPERATOR_ELEMENT_WISE_SUBTRACT)
        .value("ELEMENT_WISE_TAN", DML_OPERATOR_ELEMENT_WISE_TAN)
        .value("ELEMENT_WISE_THRESHOLD", DML_OPERATOR_ELEMENT_WISE_THRESHOLD)
        .value("ELEMENT_WISE_QUANTIZE_LINEAR", DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR)
        .value("ELEMENT_WISE_DEQUANTIZE_LINEAR", DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR)
        .value("ACTIVATION_ELU", DML_OPERATOR_ACTIVATION_ELU)
        .value("ACTIVATION_HARDMAX", DML_OPERATOR_ACTIVATION_HARDMAX)
        .value("ACTIVATION_HARD_SIGMOID", DML_OPERATOR_ACTIVATION_HARD_SIGMOID)
        .value("ACTIVATION_IDENTITY", DML_OPERATOR_ACTIVATION_IDENTITY)
        .value("ACTIVATION_LEAKY_RELU", DML_OPERATOR_ACTIVATION_LEAKY_RELU)
        .value("ACTIVATION_LINEAR", DML_OPERATOR_ACTIVATION_LINEAR)
        .value("ACTIVATION_LOG_SOFTMAX", DML_OPERATOR_ACTIVATION_LOG_SOFTMAX)
        .value("ACTIVATION_PARAMETERIZED_RELU", DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU)
        .value("ACTIVATION_PARAMETRIC_SOFTPLUS", DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS)
        .value("ACTIVATION_RELU", DML_OPERATOR_ACTIVATION_RELU)
        .value("ACTIVATION_SCALED_ELU", DML_OPERATOR_ACTIVATION_SCALED_ELU)
        .value("ACTIVATION_SCALED_TANH", DML_OPERATOR_ACTIVATION_SCALED_TANH)
        .value("ACTIVATION_SIGMOID", DML_OPERATOR_ACTIVATION_SIGMOID)
        .value("ACTIVATION_SOFTMAX", DML_OPERATOR_ACTIVATION_SOFTMAX)
        .value("ACTIVATION_SOFTPLUS", DML_OPERATOR_ACTIVATION_SOFTPLUS)
        .value("ACTIVATION_SOFTSIGN", DML_OPERATOR_ACTIVATION_SOFTSIGN)
        .value("ACTIVATION_TANH", DML_OPERATOR_ACTIVATION_TANH)
        .value("ACTIVATION_THRESHOLDED_RELU", DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU)
        .value("CONVOLUTION", DML_OPERATOR_CONVOLUTION)
        .value("GEMM", DML_OPERATOR_GEMM)
        .value("REDUCE", DML_OPERATOR_REDUCE)
        .value("AVERAGE_POOLING", DML_OPERATOR_AVERAGE_POOLING)
        .value("LP_POOLING", DML_OPERATOR_LP_POOLING)
        .value("MAX_POOLING", DML_OPERATOR_MAX_POOLING)
        .value("ROI_POOLING", DML_OPERATOR_ROI_POOLING)
        .value("SLICE", DML_OPERATOR_SLICE)
        .value("CAST", DML_OPERATOR_CAST)
        .value("SPLIT", DML_OPERATOR_SPLIT)
        .value("JOIN", DML_OPERATOR_JOIN)
        .value("PADDING", DML_OPERATOR_PADDING)
        .value("VALUE_SCALE_2D", DML_OPERATOR_VALUE_SCALE_2D)
        .value("UPSAMPLE_2D", DML_OPERATOR_UPSAMPLE_2D)
        .value("GATHER", DML_OPERATOR_GATHER)
        .value("SPACE_TO_DEPTH", DML_OPERATOR_SPACE_TO_DEPTH)
        .value("DEPTH_TO_SPACE", DML_OPERATOR_DEPTH_TO_SPACE)
        .value("TILE", DML_OPERATOR_TILE)
        .value("TOP_K", DML_OPERATOR_TOP_K)
        .value("BATCH_NORMALIZATION", DML_OPERATOR_BATCH_NORMALIZATION)
        .value("MEAN_VARIANCE_NORMALIZATION", DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION)
        .value("LOCAL_RESPONSE_NORMALIZATION", DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION)
        .value("LP_NORMALIZATION", DML_OPERATOR_LP_NORMALIZATION)
        .value("RNN", DML_OPERATOR_RNN)
        .value("LSTM", DML_OPERATOR_LSTM)
        .value("GRU", DML_OPERATOR_GRU)
        .value("ELEMENT_WISE_SIGN", DML_OPERATOR_ELEMENT_WISE_SIGN)
        .value("ELEMENT_WISE_IS_NAN", DML_OPERATOR_ELEMENT_WISE_IS_NAN)
        .value("ELEMENT_WISE_ERF", DML_OPERATOR_ELEMENT_WISE_ERF)
        .value("ELEMENT_WISE_SINH", DML_OPERATOR_ELEMENT_WISE_SINH)
        .value("ELEMENT_WISE_COSH", DML_OPERATOR_ELEMENT_WISE_COSH)
        .value("ELEMENT_WISE_TANH", DML_OPERATOR_ELEMENT_WISE_TANH)
        .value("ELEMENT_WISE_ASINH", DML_OPERATOR_ELEMENT_WISE_ASINH)
        .value("ELEMENT_WISE_ACOSH", DML_OPERATOR_ELEMENT_WISE_ACOSH)
        .value("ELEMENT_WISE_ATANH", DML_OPERATOR_ELEMENT_WISE_ATANH)
        .value("ELEMENT_WISE_IF", DML_OPERATOR_ELEMENT_WISE_IF)
        .value("ELEMENT_WISE_ADD1", DML_OPERATOR_ELEMENT_WISE_ADD1)
        .value("ACTIVATION_SHRINK", DML_OPERATOR_ACTIVATION_SHRINK)
        .value("MAX_POOLING1", DML_OPERATOR_MAX_POOLING1)
        .value("MAX_UNPOOLING", DML_OPERATOR_MAX_UNPOOLING)
        .value("DIAGONAL_MATRIX", DML_OPERATOR_DIAGONAL_MATRIX)
        .value("SCATTER", DML_OPERATOR_SCATTER)
        .value("ONE_HOT", DML_OPERATOR_ONE_HOT)
        .value("RESAMPLE", DML_OPERATOR_RESAMPLE)
        .export_values();

    // Classes
    //
    py::class_<pydml::Binding>(module, "Binding", py::buffer_protocol())
        .def(py::init([](dml::Expression& expression, py::array_t<float, py::array::c_style | py::array::forcecast> data) {
            return new pydml::Binding(expression, data.request());
            }),
            py::arg("expr"),
            py::arg("data"));

    py::class_<pydml::Device>(module, "Device")
        .def(py::init<bool, bool>(),
            py::arg("use_gpu") = true,
            py::arg("use_debug_layer") = false)
        .def("compute", [](
            pydml::Device& self, 
            pydml::CompiledModel* model,
            std::vector<pydml::Binding*> inputs,
            std::vector<dml::Expression*> outputs) {
                return self.Compute(model->op.Get(), inputs, outputs);
            }, "Calculate the output of the operator from the input data.")
        .def("__repr__",
            [](pydml::Device const& device) {
                return "dml.Device on " + std::string(device.UseGpu() ? "GPU" : "CPU");
            });

    py::class_<dml::Graph>(module, "GraphBuilder")
        .def(py::init([](pydml::Device const& device) {
            return std::unique_ptr<dml::Graph>(new dml::Graph(device.GetDevice()));
            }),
            py::arg("device"))
        .def("build", [](dml::Graph& self, DML_EXECUTION_FLAGS flags, std::vector<dml::Expression> outputs) {
            self; return new pydml::CompiledModel(self, flags, outputs);
            }, "Compile the expressions to a compiled operator.");

    py::class_<pydml::CompiledModel>(module, "Model");

    py::class_<dml::TensorDimensions>(module, "Dimensions");

    py::class_<dml::TensorPolicy>(module, "TensorPolicy")
        .def_property_readonly_static("default", []() { return dml::TensorPolicy::Default(); })
        .def_property_readonly_static("interleaved_channel", []() { return dml::TensorPolicy::InterleavedChannel(); });

    py::class_<dml::TensorDesc>(module, "TensorDesc")
        .def(py::init<DML_TENSOR_DATA_TYPE, dml::TensorDimensions, const dml::TensorPolicy&>(),
            py::arg("data_type"),
            py::arg("sizes"),
            py::arg("tensor_policy") = dml::TensorPolicy::Default())
        .def(py::init<DML_TENSOR_DATA_TYPE, DML_TENSOR_FLAGS, dml::TensorDimensions, const dml::TensorPolicy&>(),
            py::arg("data_type"),
            py::arg("flags"),
            py::arg("sizes"),
            py::arg("tensor_policy") = dml::TensorPolicy::Default())
        .def(py::init([](DML_TENSOR_DATA_TYPE dataType, dml::TensorDimensions sizes, dml::Optional<dml::TensorDimensions> strides) {
            return std::unique_ptr<dml::TensorDesc>(new dml::TensorDesc(
                dataType,
                DML_TENSOR_FLAG_NONE,
                std::move(sizes),
                std::move(strides),
                DMLCalcBufferTensorSize(
                    dataType,
                    static_cast<uint32_t>(sizes.size()),
                    sizes.data(),
                    strides ? std::move(strides)->data() : nullptr),
                0));
            }),
            py::arg("data_type"),
            py::arg("sizes"),
            py::arg("strides"))
        .def(py::init<DML_TENSOR_DATA_TYPE, DML_TENSOR_FLAGS, dml::TensorDimensions, dml::Optional<dml::TensorDimensions>, uint64_t, uint32_t>(),
            py::arg("data_type"),
            py::arg("flags"),
            py::arg("sizes"),
            py::arg("strides"),
            py::arg("total_tensor_size_in_bytes"),
            py::arg("guaranteed_base_offset_alignment"))
        .def_readonly("data_type", &dml::TensorDesc::dataType)
        .def_readonly("flags", &dml::TensorDesc::flags)
        .def_readonly("sizes", &dml::TensorDesc::sizes)
        .def_readonly("strides", &dml::TensorDesc::strides)
        .def_readonly("total_tensor_size_in_bytes", &dml::TensorDesc::totalTensorSizeInBytes)
        .def_readonly("guaranteed_base_offset_alignment", &dml::TensorDesc::guaranteedBaseOffsetAlignment)
        .def("__repr__",
            [](dml::TensorDesc const& tensorDesc) {
                return  "dml.TensorDesc of type " + std::to_string(tensorDesc.dataType) + 
                        " and shape [" + UintVectorToString(tensorDesc.sizes) + ']' +
                        " with flags " + std::to_string(tensorDesc.flags);
            });

    py::class_<pydml::TensorData>(module, "TensorData", py::buffer_protocol())
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> data) { 
            return new pydml::TensorData(data.request()); 
            }))
        .def_buffer([](pydml::TensorData& self) -> py::buffer_info {
            return py::buffer_info(
                self.Get(),
                self.itemSize,
                self.format,
                self.dimensions,
                self.shape,
                self.strides
                ); });

    py::class_<dml::Expression>(module, "Expression")
        .def(py::init<>())
        .def("get_output_desc", &dml::Expression::GetOutputDesc, "Get the expression's output descriptor.")
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self % py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)
        .def(py::self %= py::self)
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self)
        .def(py::self += float())
        .def(py::self -= float())
        .def(py::self *= float())
        .def(py::self /= float())
        .def(-py::self);

    py::class_<DML_SIZE_2D>(module, "Size2D")
        .def(py::init([](uint32_t width, uint32_t height) {
            return new DML_SIZE_2D { width, height };
            }))
        .def_readwrite("width", &DML_SIZE_2D::Width)
        .def_readwrite("height", &DML_SIZE_2D::Height);

    py::class_<dml::FusedActivation>(module, "FusedActivation")
        .def(py::init<DML_OPERATOR_TYPE, float, float>(),
            py::arg("activation"),
            py::arg("param_1") = 0,
            py::arg("param_2") = 0);

    py::class_<dml::MaxPoolingOutputs>(module, "MaxPoolingOutputs")
        .def(py::init([](dml::Expression values, dml::Expression indices) {
            return new dml::MaxPoolingOutputs { values, indices };
            }))
        .def_readwrite("values", &dml::MaxPoolingOutputs::values)
        .def_readwrite("indices", &dml::MaxPoolingOutputs::indices);

    py::class_<dml::GRUOutputs>(module, "GRUOutputs")
        .def(py::init([](dml::Expression sequence, dml::Expression single) {
            return new dml::GRUOutputs { sequence, single };
            }))
        .def_readwrite("sequence", &dml::GRUOutputs::sequence)
        .def_readwrite("single", &dml::GRUOutputs::single);

    // Functions
    //
    module.def("input_tensor", &dml::InputTensor, "Create an input tensor as an expression.",
        py::arg("scope"),
        py::arg("input_index"),
        py::arg("tensor_desc"));

    module.def("convolution", [](
        dml::Expression input,
        dml::Expression filter,
        dml::Optional<dml::Expression> bias,
        DML_CONVOLUTION_MODE mode,
        DML_CONVOLUTION_DIRECTION direction,
        std::vector<uint32_t> strides,
        std::vector<uint32_t> dilations,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding,
        std::vector<uint32_t> outputPadding,
        uint32_t groupCount,
        dml::FusedActivation fusedActivation,
        dml::TensorDimensions outputSizes) {
            return dml::Convolution(input, filter, bias, mode, direction, strides, dilations, startPadding, endPadding, outputPadding, groupCount, fusedActivation, outputSizes);
        }, 
        "Create a builder of the convolution expression.",
        py::arg("input"),
        py::arg("filter"),
        py::arg("bias") = dml::NullOpt,
        py::arg("mode") = DML_CONVOLUTION_MODE_CROSS_CORRELATION,
        py::arg("direction") = DML_CONVOLUTION_DIRECTION_FORWARD,
        py::arg("strides") = std::vector<uint32_t>{},
        py::arg("dilations") = std::vector<uint32_t>{},
        py::arg("start_padding") = std::vector<uint32_t>{},
        py::arg("end_padding") = std::vector<uint32_t>{},
        py::arg("output_padding") = std::vector<uint32_t>{},
        py::arg("group_count") = 1,
        py::arg("fused_activation") = dml::FusedActivation::None(),
        py::arg("output_sizes") = dml::TensorDimensions{});

    module.def("up_sample_2d", &dml::Upsample2D, "Create a two-dimensional up-sample expression.",
        py::arg("input"),
        py::arg("scale_size"),
        py::arg("interpolation_mode"));

    module.def("activation_relu", &dml::ActivationRelu, "Takes an input tensor and applies the function output = max(0, input) across its elements.",
        py::arg("input"));

    module.def("activation_sigmoid", &dml::ActivationSigmoid, "Takes an input tensor and applies the function output = 1 / (1 + exp(-input)) across its elements.",
        py::arg("input"));

    module.def("activation_identity", &dml::ActivationIdentity, "Takes an input tensor and return the tensor as an output.",
        py::arg("input"));

    module.def("add", py::overload_cast<dml::Expression, dml::Expression, dml::FusedActivation>(&dml::Add), "Takes 2 input tensors and performs addition then returns the resulting tensor.",
        py::arg("a"),
        py::arg("b"),
        py::arg("fused_activation") = dml::FusedActivation::None());

    module.def("subtract", &dml::Subtract, "Takes 2 input tensors and performs subtraction then returns the resulting tensor.",
        py::arg("a"),
        py::arg("b"));

    module.def("activation_tanh", &dml::ActivationTanh, "Calculates the hyperbolic tangent of the given input tensor.",
        py::arg("input"));

    module.def("multiply", &dml::Multiply, "Takes 2 input tensors and performs multiplication then returns resulting tensor.",
        py::arg("a"),
        py::arg("b"));

    module.def("divide", &dml::Divide, "Takes 2 input tensors and performs division then returns the resulting tensor.",
        py::arg("a"),
        py::arg("b"));

    module.def("padding", [](
        dml::Expression input,
        DML_PADDING_MODE paddingMode,
        float paddingValue,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding) {
            return dml::Padding(input, paddingMode, paddingValue, startPadding, endPadding);
        },
        "Inflate the input with zeros on the edges.",
        py::arg("input"),
        py::arg("padding_mode"),
        py::arg("padding_value"), 
        py::arg("start_padding"), 
        py::arg("end_padding"));

    module.def("mean_variance_normalization", [](
        dml::Expression input,
        dml::Optional<dml::Expression> scale,
        dml::Optional<dml::Expression> bias,
        std::vector<uint32_t> axes,
        bool normalizeVariance,
        bool normalizeMean,
        float epsilon,
        dml::FusedActivation fusedActivation) {
            return dml::MeanVarianceNormalization(input, scale, bias, axes, normalizeVariance, normalizeMean, epsilon, fusedActivation);
        }, "Normalize inputs using output = scale * (input - mean) / sqrt(variance + epsilon) + bias, where mean and variance are computed per instance per channel.",
        py::arg("input"),
        py::arg("scale") = dml::NullOpt,
        py::arg("bias") = dml::NullOpt,
        py::arg("axes") = std::vector<uint32_t>{},
        py::arg("normalize_variance"),
        py::arg("epsilon"),
        py::arg("fused_activation") = dml::FusedActivation::None());

    module.def("slice", [](
        dml::Expression input,
        std::vector<uint32_t> inputWindowOffsets,
        std::vector<uint32_t> inputWindowSizes,
        std::vector<int32_t> inputWindowStrides) {
            return dml::Slice(input, inputWindowOffsets, inputWindowSizes, inputWindowStrides);
        }, 
        "Produces a slice of the input tensor along multiple axes",
        py::arg("input"),
        py::arg("input_window_offsets"),
        py::arg("input_window_sizes"),
        py::arg("input_window_strides"));

    module.def("value_scale_2d", [](
        dml::Expression input,
        float scale,
        std::vector<float> bias) {
            return dml::ValueScale2D(input, scale, bias);
        },
        "Scales and bias the input image per pixel. output = input * scale + bias[C]",
        py::arg("input"),
        py::arg("scale"),
        py::arg("bias"));

    module.def("activation_linear", &dml::ActivationLinear, " f(input, alpha, beta) = alpha * input + beta",
        py::arg("input"),
        py::arg("alpha"),
        py::arg("beta"));

    module.def("batch_normalization", &dml::BatchNormalization, "normalizes data per channel across all batches by subtracting the mean, dividing by the standard deviation, and adding a bias.",
        py::arg("input"),
        py::arg("mean"),
        py::arg("variance"),
        py::arg("scale"),
        py::arg("bias"),
        py::arg("spatial"),
        py::arg("epsilon"),
        py::arg("fused_activation") = dml::FusedActivation::None());

    module.def("local_response_normalization", &dml::LocalResponseNormalization, "It normalizes over local input regions defined across the channels.",
        py::arg("input"),
        py::arg("cross_channel"),
        py::arg("local_size"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("bias"));

    module.def("gemm", [](
        dml::Expression a,
        dml::Expression b,
        dml::Optional<dml::Expression> c,
        DML_MATRIX_TRANSFORM transA,
        DML_MATRIX_TRANSFORM transB,
        float alpha,
        float beta,
        dml::FusedActivation fusedActivation) {
            return dml::Gemm(a, b, c, transA, transB, alpha, beta, fusedActivation);
        },
        "Matrix product of two matrices",
        py::arg("a"),
        py::arg("b"),
        py::arg("c") = dml::NullOpt,
        py::arg("trans_a") = DML_MATRIX_TRANSFORM_NONE,
        py::arg("trans_b") = DML_MATRIX_TRANSFORM_NONE,
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 1.0f,
        py::arg("fused_activation") = dml::FusedActivation::None());

    module.def("average_pooling", [](
        dml::Expression input,
        std::vector<uint32_t> strides,
        std::vector<uint32_t> windowSizes,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding,
        bool includePadding) {
            return dml::AveragePooling(input, strides, windowSizes, startPadding, endPadding, includePadding);
        },
        "Average all elements in each pool.",
        py::arg("input"),
        py::arg("strides"),
        py::arg("window_sizes"),
        py::arg("start_padding"),
        py::arg("end_padding"),
        py::arg("include_padding"));

    module.def("max_pooling", [](
        dml::Expression input,
        std::vector<uint32_t> windowSizes,
        std::vector<uint32_t> strides,
        std::vector<uint32_t> startPadding,
        std::vector<uint32_t> endPadding,
        std::vector<uint32_t> dilations,
        bool outputIndices) {
            return dml::MaxPooling(input, windowSizes, strides, startPadding, endPadding, dilations, outputIndices);
        },
        "Max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths",
        py::arg("input"),
        py::arg("window_sizes"),
        py::arg("strides") = std::vector<uint32_t>{},
        py::arg("start_padding") = std::vector<uint32_t>{},
        py::arg("end_padding") = std::vector<uint32_t>{},
        py::arg("dilations") = std::vector<uint32_t>{},
        py::arg("output_indices") = false);

    module.def("reinterpret", [](
        dml::Expression input,
        DML_TENSOR_DATA_TYPE newType,
        dml::TensorDimensions newSizes,
        dml::Optional<dml::TensorDimensions> newStrides) {
            return dml::Reinterpret(input, newType, newSizes, newStrides);
        },
        "Return tensor with a different view of the data, like a reinterpret cast using new dimensions that are element-count compatible.",
        py::arg("input"),
        py::arg("new_type"),
        py::arg("new_size"),
        py::arg("new_strides"));

    module.def("activation_soft_max", py::overload_cast<dml::Expression>(&dml::ActivationSoftmax), "Raise all elements to e, and divide all the elements in each batch by that batch's sum.",
        py::arg("input"));

    module.def("join", [](
        std::vector<dml::Expression> inputs,
        uint32_t axis) {
            return dml::Join(inputs, axis);
        },
        "Combine multiple tensors into large output tensor.",
        py::arg("input"),
        py::arg("axis"));

    module.def("gru", [](
        dml::Expression input,
        dml::Expression weight,
        dml::Expression recurrence,
        dml::Optional<dml::Expression> bias,
        dml::Optional<dml::Expression> hiddenInit,
        dml::Optional<dml::Expression> sequenceLengths,
        std::vector<dml::FusedActivation> activationDescs,
        DML_RECURRENT_NETWORK_DIRECTION direction,
        BOOL linearBeforeReset,
        dml::GRUOutputOptions outputOptions) {
            return dml::GRU(input, weight, recurrence, bias, hiddenInit, sequenceLengths, activationDescs, direction, linearBeforeReset, outputOptions);
        },
        "Performs a one-layer gated recurrent unit (GRU) function on the input. This operator uses multiple gates to perform this layer. These gates are performed multiple times in a loop dictated by the sequence length dimension and the sequence_lengths argument.",
        py::arg("input"),
        py::arg("weight"),
        py::arg("recurrence"),
        py::arg("bias") = dml::NullOpt,
        py::arg("hidden_init") = dml::NullOpt,
        py::arg("sequence_lengths") = dml::NullOpt,
        py::arg("activation_descs"),
        py::arg("direction"),
        py::arg("linear_before_reset") = 1,
        py::arg("output_options"));

    module.def("gather", &dml::Gather, "Gathers elements from the input tensor along current axis, using indices tensor to remap indices.",
        py::arg("input"),
        py::arg("indices"),
        py::arg("axis"),
        py::arg("index_dimensions"));
}
