#
# DirectML FNS Candy sample
# Based on the following model: https://github.com/microsoft/Windows-Machine-Learning/blob/master/Samples/FNSCandyStyleTransfer/UWP/cs/Assets/candy.onnx
#

import pydirectml as dml
import numpy as np
from PIL import Image, ImageOps
import sys
import os

argument_count = len(sys.argv)

image_file_path = "DefaultImage.jpg"
tensor_data_path = "candy_tensor_data"

# Get user image input path if any. If none, default to image_file_path value.
if (argument_count >= 2):
    image_file_path = sys.argv[1]

if (argument_count >= 3):
    tensor_data_path = sys.argv[2]

if (os.path.exists(image_file_path) == False):
    print("File not found at: " + str(image_file_path))
    sys.exit(1)

image = Image.open(image_file_path)

image = ImageOps.fit(image, (720, 720), method = 0,  bleed = 0, centering = (0.5, 0.5))

image_tensor = np.array(image, np.float32)

transposed_image = np.transpose(image_tensor, axes=[2,0,1])

# Create a GPU device and build a model graph.
device = dml.Device(True, True)
builder = dml.GraphBuilder(device)

data_type = dml.TensorDataType.FLOAT32
input = dml.input_tensor(builder, 0, dml.TensorDesc(data_type, [1, 3, 720, 720]));
flags = dml.TensorFlags.OWNED_BY_DML

# scalar1
scaler1_bias = [-103.93900299072266, -116.77899932861328, -123.68000030517578]
scaler1 = dml.value_scale_2d(input, 1.0, scaler1_bias)

# pad3
pad3 = dml.padding(scaler1, dml.PaddingMode.REFLECTION, 0.0, [0, 0, 40, 40], [0, 0, 40, 40])

# conv4
conv4_filter = dml.input_tensor(builder, 1, dml.TensorDesc(data_type, flags, [16,3,9,9]))
conv4_bias = dml.input_tensor(builder, 2, dml.TensorDesc(data_type, flags, [1,16,1,1]))
conv4 = dml.convolution(pad3, conv4_filter, conv4_bias, start_padding = [4,4], end_padding = [4,4])

# instance_norm5
instance_norm5_scale = dml.input_tensor(builder, 3, dml.TensorDesc(data_type, [1,16,1,1]))
instance_norm5_bias = dml.input_tensor(builder, 4, dml.TensorDesc(data_type, flags, [1,16,1,1]))
instance_norm5 = dml.mean_variance_normalization(conv4, instance_norm5_scale, instance_norm5_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv7
conv7_filter = dml.input_tensor(builder, 5, dml.TensorDesc(data_type, flags, [32,16,3,3]))
conv7_bias = dml.input_tensor(builder, 6, dml.TensorDesc(data_type, flags, [1,32,1,1]))
conv7 = dml.convolution(instance_norm5, conv7_filter, conv7_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1])

# instance_norm8
instance_norm8_scale = dml.input_tensor(builder, 7, dml.TensorDesc(data_type, [1,32,1,1]))
instance_norm8_bias = dml.input_tensor(builder, 8, dml.TensorDesc(data_type, flags, [1,32,1,1]))
instance_norm8 = dml.mean_variance_normalization(conv7, instance_norm8_scale, instance_norm8_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv10
conv10_filter = dml.input_tensor(builder, 9, dml.TensorDesc(data_type, flags, [64,32,3,3]))
conv10_bias = dml.input_tensor(builder, 10, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv10 = dml.convolution(instance_norm8, conv10_filter, conv10_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1])

# instance_norm11
instance_norm11_scale = dml.input_tensor(builder, 11, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm11_bias = dml.input_tensor(builder, 12, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm11 = dml.mean_variance_normalization(conv10, instance_norm11_scale, instance_norm11_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv16
conv16_filter = dml.input_tensor(builder, 13, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv16_bias = dml.input_tensor(builder, 14, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv16 = dml.convolution(instance_norm11, conv16_filter, conv16_bias)

# instance_norm17
instance_norm17_scale = dml.input_tensor(builder, 15, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm17_bias = dml.input_tensor(builder, 16, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm17 = dml.mean_variance_normalization(conv16, instance_norm17_scale, instance_norm17_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv19
conv19_filter = dml.input_tensor(builder, 17, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv19_bias = dml.input_tensor(builder, 18, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv19 = dml.convolution(instance_norm17, conv19_filter, conv19_bias)

# instance_norm15
instance_norm15_scale = dml.input_tensor(builder, 19, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm15_bias = dml.input_tensor(builder, 20, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm15 = dml.mean_variance_normalization(conv19, instance_norm15_scale, instance_norm15_bias, [0,2,3], 1, 0.000009999999747378752)

# crop21
crop21 = dml.slice(instance_norm11,[0,0,2,2],[1,64,196,196],[1,1,1,1])

# add
add = dml.add(instance_norm15, crop21)

# conv26
conv26_filter = dml.input_tensor(builder, 21, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv26_bias = dml.input_tensor(builder, 22, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv26 = dml.convolution(add, conv26_filter, conv26_bias)

# instance_norm27
instance_norm27_scale = dml.input_tensor(builder, 23, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm27_bias = dml.input_tensor(builder, 24, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm27 = dml.mean_variance_normalization(conv26, instance_norm27_scale, instance_norm27_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv29
conv29_filter = dml.input_tensor(builder, 25, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv29_bias = dml.input_tensor(builder, 26, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv29 = dml.convolution(instance_norm27, conv29_filter, conv29_bias)

# instance_norm25
instance_norm25_scale = dml.input_tensor(builder, 27, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm25_bias = dml.input_tensor(builder, 28, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm25 = dml.mean_variance_normalization(conv29, instance_norm25_scale, instance_norm25_bias, [0,2,3], 1, 0.000009999999747378752)

# crop21
crop31 = dml.slice(add,[0,0,2,2],[1,64,196-2-2,196-2-2],[1,1,1,1])

# add1
add1 = dml.add(instance_norm25, crop31)

#conv36
conv36_filter = dml.input_tensor(builder, 29, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv36_bias = dml.input_tensor(builder, 30, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv36 = dml.convolution(add1, conv36_filter, conv36_bias)

# instance_norm37
instance_norm37_scale = dml.input_tensor(builder, 31, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm37_bias = dml.input_tensor(builder, 32, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm37 = dml.mean_variance_normalization(conv36, instance_norm37_scale, instance_norm37_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv39
conv39_filter = dml.input_tensor(builder, 33, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv39_bias = dml.input_tensor(builder, 34, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv39 = dml.convolution(instance_norm37, conv39_filter, conv39_bias)

# instance_norm35
instance_norm35_scale = dml.input_tensor(builder, 35, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm35_bias = dml.input_tensor(builder, 36, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm35 = dml.mean_variance_normalization(conv39, instance_norm35_scale, instance_norm35_bias, [0,2,3], 1, 0.000009999999747378752)

# crop41
crop41 = dml.slice(add1,[0,0,2,2],[1, 64, 192-2-2, 192-2-2],[1,1,1,1])

# add2
add2 = dml.add(instance_norm35, crop41)

# conv46
conv46_filter = dml.input_tensor(builder, 37, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv46_bias = dml.input_tensor(builder, 38, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv46 = dml.convolution(add2, conv46_filter, conv46_bias)

# instance_norm47
instance_norm47_scale = dml.input_tensor(builder, 39, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm47_bias = dml.input_tensor(builder, 40, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm47 = dml.mean_variance_normalization(conv46, instance_norm47_scale, instance_norm47_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv49
conv49_filter = dml.input_tensor(builder, 41, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv49_bias = dml.input_tensor(builder, 42, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv49 = dml.convolution(instance_norm47, conv49_filter, conv49_bias)

# instance_norm45
instance_norm45_scale = dml.input_tensor(builder, 43, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm45_bias = dml.input_tensor(builder, 44, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm45 = dml.mean_variance_normalization(conv49, instance_norm45_scale, instance_norm45_bias, [0,2,3], 1, 0.000009999999747378752)

# crop51
crop51 = dml.slice(instance_norm35,[0,0,2,2],[1, 64, 188-2-2, 188-2-2],[1,1,1,1])

# add3
add3 = dml.add(instance_norm45, crop51)

# conv56
conv56_filter = dml.input_tensor(builder, 45, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv56_bias = dml.input_tensor(builder, 46, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv56 = dml.convolution(add3, conv56_filter, conv56_bias)

# instance_norm57
instance_norm57_scale = dml.input_tensor(builder, 47, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm57_bias = dml.input_tensor(builder, 48, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm57 = dml.mean_variance_normalization(conv56, instance_norm57_scale, instance_norm57_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv59
conv59_filter = dml.input_tensor(builder, 49, dml.TensorDesc(data_type, flags, [64,64,3,3]))
conv59_bias = dml.input_tensor(builder, 50, dml.TensorDesc(data_type, flags, [1,64,1,1]))
conv59 = dml.convolution(instance_norm57, conv59_filter, conv59_bias)

# instance_norm55
instance_norm55_scale = dml.input_tensor(builder, 51, dml.TensorDesc(data_type, [1,64,1,1]))
instance_norm55_bias = dml.input_tensor(builder, 52, dml.TensorDesc(data_type, flags, [1,64,1,1]))
instance_norm55 = dml.mean_variance_normalization(conv59, instance_norm55_scale, instance_norm55_bias, [0,2,3], 1, 0.000009999999747378752)

# crop61
crop61 = dml.slice(add3,[0,0,2,2],[1, 64, 184-2-2, 184-2-2],[1,1,1,1])

# add4
add4 = dml.add(instance_norm55, crop61)

# conv63
conv63_filter = dml.input_tensor(builder, 53, dml.TensorDesc(data_type, flags, [64,32,3,3]))
conv63_bias = dml.input_tensor(builder, 54, dml.TensorDesc(data_type, flags, [1,32,1,1]))
conv63 = dml.convolution(add4, conv63_filter, conv63_bias, mode = dml.ConvolutionMode.CROSS_CORRELATION, direction = dml.ConvolutionDirection.BACKWARD, strides = [2,2], start_padding = [0,0], end_padding = [0,0], output_sizes = [1,32,362,362])

# crop63
crop63 = dml.slice(conv63,[0,0,1,1],[1, 32, 362-1-1, 362-1-1],[1,1,1,1])

# instance_norm65
instance_norm65_scale = dml.input_tensor(builder, 55, dml.TensorDesc(data_type, [1,32,1,1]))
instance_norm65_bias = dml.input_tensor(builder, 56, dml.TensorDesc(data_type, flags, [1,32,1,1]))
instance_norm65 = dml.mean_variance_normalization(crop63, instance_norm65_scale, instance_norm65_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv67
conv67_filter = dml.input_tensor(builder, 57, dml.TensorDesc(data_type, flags, [32,16,3,3]))
conv67_bias = dml.input_tensor(builder, 58, dml.TensorDesc(data_type, flags, [1,16,1,1]))
conv67 = dml.convolution(instance_norm65, conv67_filter, conv67_bias, mode = dml.ConvolutionMode.CROSS_CORRELATION, direction = dml.ConvolutionDirection.BACKWARD, strides = [2,2], start_padding = [0,0], end_padding = [0,0], output_sizes = [1,16, 722,722])

# crop67
crop67 = dml.slice(conv67,[0,0,1,1],[1, 16, 722-1-1, 722-1-1],[1,1,1,1])

# instance_norm69
instance_norm69_scale = dml.input_tensor(builder, 59, dml.TensorDesc(data_type, [1,16,1,1]))
instance_norm69_bias = dml.input_tensor(builder, 60, dml.TensorDesc(data_type, flags, [1,16,1,1]))
instance_norm69 = dml.mean_variance_normalization(crop67, instance_norm69_scale, instance_norm69_bias, [0,2,3], 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv71
conv71_filter = dml.input_tensor(builder, 61, dml.TensorDesc(data_type, flags, [3,16,9,9]))
conv71_bias = dml.input_tensor(builder, 62, dml.TensorDesc(data_type, flags, [1,3,1,1]))
conv71 = dml.convolution(instance_norm69, conv71_filter, conv71_bias, start_padding = [4,4], end_padding = [4,4], fused_activation = dml.FusedActivation(dml.OperatorType.ACTIVATION_TANH))

# activation11
affine = dml.activation_linear(conv71, 150, 0)

# deprocess_image_2_scaled
mul_b = dml.input_tensor(builder, 63, dml.TensorDesc(data_type, [1,3,720,720], [3,1,0,0]));
image_2_scaled = dml.multiply(affine, mul_b)

# add_output
scale_b = dml.input_tensor(builder, 64, dml.TensorDesc(data_type, [1,3,720,720], [3,1,0,0]));
output_image = dml.add(image_2_scaled, scale_b)

# Compile the expression graph into a compiled operator
op = builder.build(dml.ExecutionFlags.NONE, [output_image])

# Model inputs in the previously designated order
inputs = [
    (conv4_filter, "convolution_W.npy"),
    (conv4_bias, "convolution_B.npy"),
    (instance_norm5_scale, "InstanceNormalization_scale.npy"),
    (instance_norm5_bias, "InstanceNormalization_B.npy"),
    (conv7_filter, "convolution1_W.npy"),
    (conv7_bias, "convolution1_B.npy"),
    (instance_norm8_scale, "InstanceNormalization_scale1.npy"),
    (instance_norm8_bias, "InstanceNormalization_B1.npy"),
    (conv10_filter, "convolution2_W.npy"),
    (conv10_bias, "convolution2_B.npy"),
    (instance_norm11_scale, "InstanceNormalization_scale2.npy"),
    (instance_norm11_bias, "InstanceNormalization_B2.npy"),
    (conv16_filter, "convolution3_W.npy"),
    (conv16_bias, "convolution3_B.npy"),
    (instance_norm17_scale, "InstanceNormalization_scale3.npy"),
    (instance_norm17_bias, "InstanceNormalization_B3.npy"),
    (conv19_filter, "convolution4_W.npy"),
    (conv19_bias, "convolution4_B.npy"),
    (instance_norm15_scale, "InstanceNormalization_scale4.npy"),
    (instance_norm15_bias, "InstanceNormalization_B4.npy"),
    (conv26_filter, "convolution5_W.npy"),
    (conv26_bias, "convolution5_B.npy"),
    (instance_norm27_scale, "InstanceNormalization_scale5.npy"),
    (instance_norm27_bias, "InstanceNormalization_B5.npy"),
    (conv29_filter, "convolution6_W.npy"),
    (conv29_bias, "convolution6_B.npy"),
    (instance_norm25_scale, "InstanceNormalization_scale6.npy"),
    (instance_norm25_bias, "InstanceNormalization_B6.npy"),
    (conv36_filter, "convolution7_W.npy"),
    (conv36_bias, "convolution7_B.npy"),
    (instance_norm37_scale, "InstanceNormalization_scale7.npy"),
    (instance_norm37_bias, "InstanceNormalization_B7.npy"),
    (conv39_filter, "convolution8_W.npy"),
    (conv39_bias, "convolution8_B.npy"),
    (instance_norm35_scale, "InstanceNormalization_scale8.npy"),
    (instance_norm35_bias, "InstanceNormalization_B8.npy"),
    (conv46_filter, "convolution9_W.npy"),
    (conv46_bias, "convolution9_B.npy"),
    (instance_norm47_scale, "InstanceNormalization_scale9.npy"),
    (instance_norm47_bias, "InstanceNormalization_B9.npy"),
    (conv49_filter, "convolution10_W.npy"),
    (conv49_bias, "convolution10_B.npy"),
    (instance_norm45_scale, "InstanceNormalization_scale10.npy"),
    (instance_norm45_bias, "InstanceNormalization_B10.npy"),
    (conv56_filter, "convolution11_W.npy"),
    (conv56_bias, "convolution11_B.npy"),
    (instance_norm57_scale, "InstanceNormalization_scale11.npy"),
    (instance_norm57_bias, "InstanceNormalization_B11.npy"),
    (conv59_filter, "convolution12_W.npy"),
    (conv59_bias,  "convolution12_B.npy"),
    (instance_norm55_scale, "InstanceNormalization_scale12.npy"),
    (instance_norm55_bias, "InstanceNormalization_B12.npy"),
    (conv63_filter, "convolution13_W.npy"),
    (conv63_bias, "convolution13_B.npy"),
    (instance_norm65_scale, "InstanceNormalization_scale13.npy"),
    (instance_norm65_bias, "InstanceNormalization_B13.npy"),
    (conv67_filter, "convolution14_W.npy"),
    (conv67_bias, "convolution14_B.npy"),
    (instance_norm69_scale, "InstanceNormalization_scale14.npy"),
    (instance_norm69_bias, "InstanceNormalization_B14.npy"),
    (conv71_filter, "convolution15_W.npy"),
    (conv71_bias, "convolution15_B.npy"),
    (mul_b, "Mul_B.npy"),
    (scale_b, "scale_B.npy")
]

input_bindings = []
input_bindings.append(dml.Binding(input, transposed_image))
for input, file_name in inputs:
    input_bindings.append(dml.Binding(input, np.load(tensor_data_path +'/' + file_name)))

# Compute the result

output_data = device.compute(op, input_bindings, [output_image])

output_tensor = np.array(output_data[0])

transposed_output_image = np.transpose(output_tensor, axes = [0,2,3,1]).squeeze()

styled_image = Image.fromarray(np.uint8(transposed_output_image))

styled_image.show()
