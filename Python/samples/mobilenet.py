#
# DirectML MoblieNet sample
# Based on the following model: https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx
#

import pydirectml as dml
import numpy as np
from PIL import Image, ImageOps
import sys
import os

argument_count = len(sys.argv)

image_file_path = "DefaultImage.jpg"
tensor_data_path = "mobile_net_data"

if (argument_count == 2):
    image_file_path = sys.argv[1]

if (os.path.exists(image_file_path) == False):
    print("File not found at: " + str(image_file_path))
    sys.exit(1)

# Opens image, converts to RGB (in case grayscale or contains an alpha channel), resizes, and crops to the input size.
image = ImageOps.fit(Image.open(image_file_path).convert("RGB"), (224, 224), method = 0, bleed = 0, centering = (0.5, 0.5))

# Transposes image array from (H x W x C) to (C x H x W) and rescales its value to between 0 and 1.
ndarray_image = np.transpose(np.array(image, np.float32), axes = [2, 0, 1])
rescaled_image = ndarray_image / ndarray_image.max()

# Normalizes the rescaled image values using the model training statistics.
mean = np.array([[[0.485]],[[ 0.456]],[[0.406]]])
standard_deviation = np.array([[[0.229]],[[0.224]],[[0.225]]])
processed_image = (rescaled_image - mean) / standard_deviation

input_bindings = []

def append_input_tensor(builder: dml.GraphBuilder, input_bindings: list, input_tensor: dml.TensorDesc, file_name: str):
    tensor = dml.input_tensor(builder, len(input_bindings), input_tensor)
    if file_name == "":
        input_bindings.append(dml.Binding(tensor, np.zeros(tensor.get_output_desc().sizes)))
    else:
        input_bindings.append(dml.Binding(tensor, np.load(tensor_data_path + "/" + file_name)))
    return tensor

# Create a GPU device, and build a model graph.
device = dml.Device(True, False)
builder = dml.GraphBuilder(device)

data_type = dml.TensorDataType.FLOAT32
input = dml.input_tensor(builder, 0, dml.TensorDesc(data_type, [1,3,224,224]));
flags = dml.TensorFlags.OWNED_BY_DML

input_bindings.append(dml.Binding(input, processed_image))

# conv1 
conv1_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [32,3,3,3]), "mobilenetv20_features_conv0_weight.npy")
conv1_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,32,1,1]), "")
conv1 = dml.convolution(input, conv1_filter, conv1_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1])

# batch_norm1
batch_norm1_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_batchnorm0_running_mean.npy")
batch_norm1_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_batchnorm0_running_var.npy")
batch_norm1_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_batchnorm0_gamma.npy")
batch_norm1_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_batchnorm0_beta.npy")
batch_norm1 = dml.batch_normalization(conv1, batch_norm1_mean, batch_norm1_variance, batch_norm1_scale, batch_norm1_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv2
conv2_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [32,32,1,1]), "mobilenetv20_features_linearbottleneck0_conv0_weight.npy")
conv2_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,32,1,1]), "")
conv2 = dml.convolution(batch_norm1, conv2_filter, conv2_bias)

# batch_norm2
batch_norm2_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm0_running_mean.npy")
batch_norm2_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm0_running_var.npy")
batch_norm2_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm0_gamma.npy")
batch_norm2_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm0_beta.npy")
batch_norm2 = dml.batch_normalization(conv2, batch_norm2_mean, batch_norm2_variance, batch_norm2_scale, batch_norm2_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv3
conv3_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [32,1,3,3]), "mobilenetv20_features_linearbottleneck0_conv1_weight.npy")
conv3_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,32,1,1]), "")
conv3 = dml.convolution(batch_norm2, conv3_filter, conv3_bias, start_padding = [1,1], end_padding = [1,1], group_count = 32)

# batch_norm3
batch_norm3_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm1_running_mean.npy")
batch_norm3_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm1_running_var.npy")
batch_norm3_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm1_gamma.npy")
batch_norm3_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm1_beta.npy")
batch_norm3 = dml.batch_normalization(conv3, batch_norm3_mean, batch_norm3_variance, batch_norm3_scale, batch_norm3_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv4
conv4_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [16,32,1,1]), "mobilenetv20_features_linearbottleneck0_conv2_weight.npy")
conv4_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,16,1,1]), "")
conv4 = dml.convolution(batch_norm3, conv4_filter, conv4_bias)

# batch_norm4
batch_norm4_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,16,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm2_running_mean.npy")
batch_norm4_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,16,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm2_running_var.npy")
batch_norm4_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,16,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm2_gamma.npy")
batch_norm4_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,16,1,1]), "mobilenetv20_features_linearbottleneck0_batchnorm2_beta.npy")
batch_norm4 = dml.batch_normalization(conv4, batch_norm4_mean, batch_norm4_variance, batch_norm4_scale, batch_norm4_bias, 1, 0.000009999999747378752)

# conv5
conv5_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [96,16,1,1]), "mobilenetv20_features_linearbottleneck1_conv0_weight.npy")
conv5_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,96,1,1]), "")
conv5 = dml.convolution(batch_norm4, conv5_filter, conv5_bias)

# batch_norm5
batch_norm5_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm0_running_mean.npy")
batch_norm5_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm0_running_var.npy")
batch_norm5_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm0_gamma.npy")
batch_norm5_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm0_beta.npy")
batch_norm5 = dml.batch_normalization(conv5, batch_norm5_mean, batch_norm5_variance, batch_norm5_scale, batch_norm5_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv6
conv6_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [96,1,3,3]), "mobilenetv20_features_linearbottleneck1_conv1_weight.npy")
conv6_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,96,1,1]), "")
conv6 = dml.convolution(batch_norm5, conv6_filter, conv6_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1], group_count = 96)

# batch_norm6
batch_norm6_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm1_running_mean.npy")
batch_norm6_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm1_running_var.npy")
batch_norm6_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm1_gamma.npy")
batch_norm6_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm1_beta.npy")
batch_norm6 = dml.batch_normalization(conv6, batch_norm6_mean, batch_norm6_variance, batch_norm6_scale, batch_norm6_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv7
conv7_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [24,96,1,1]), "mobilenetv20_features_linearbottleneck1_conv2_weight.npy")
conv7_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,24,1,1]), "")
conv7 = dml.convolution(batch_norm6, conv7_filter, conv7_bias)

# batch_norm7
batch_norm7_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,24,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm2_running_mean.npy")
batch_norm7_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,24,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm2_running_var.npy")
batch_norm7_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,24,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm2_gamma.npy")
batch_norm7_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,24,1,1]), "mobilenetv20_features_linearbottleneck1_batchnorm2_beta.npy")
batch_norm7 = dml.batch_normalization(conv7, batch_norm7_mean, batch_norm7_variance, batch_norm7_scale, batch_norm7_bias, 1, 0.000009999999747378752)

# conv8
conv8_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [144,24,1,1]), "mobilenetv20_features_linearbottleneck2_conv0_weight.npy")
conv8_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,144,1,1]), "")
conv8 = dml.convolution(batch_norm7, conv8_filter, conv8_bias)

# batch_norm8
batch_norm8_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm0_running_mean.npy")
batch_norm8_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm0_running_var.npy")
batch_norm8_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm0_gamma.npy")
batch_norm8_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm0_beta.npy")
batch_norm8 = dml.batch_normalization(conv8, batch_norm8_mean, batch_norm8_variance, batch_norm8_scale, batch_norm8_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv9
conv9_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [144,1,3,3]), "mobilenetv20_features_linearbottleneck2_conv1_weight.npy")
conv9_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,144,1,1]), "")
conv9 = dml.convolution(batch_norm8, conv9_filter, conv9_bias, start_padding = [1,1], end_padding = [1,1], group_count = 144)

# batch_norm9
batch_norm9_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm1_running_mean.npy")
batch_norm9_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm1_running_var.npy")
batch_norm9_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm1_gamma.npy")
batch_norm9_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm1_beta.npy")
batch_norm9 = dml.batch_normalization(conv9, batch_norm9_mean, batch_norm9_variance, batch_norm9_scale, batch_norm9_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv10
conv10_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [24,144,1,1]), "mobilenetv20_features_linearbottleneck2_conv2_weight.npy")
conv10_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,24,1,1]), "")
conv10 = dml.convolution(batch_norm9, conv10_filter, conv10_bias)

# batch_norm10
batch_norm10_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,24,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm2_running_mean.npy")
batch_norm10_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,24,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm2_running_var.npy")
batch_norm10_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,24,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm2_gamma.npy")
batch_norm10_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,24,1,1]), "mobilenetv20_features_linearbottleneck2_batchnorm2_beta.npy")
batch_norm10 = dml.batch_normalization(conv10, batch_norm10_mean, batch_norm10_variance, batch_norm10_scale, batch_norm10_bias, 1, 0.000009999999747378752)

# add1
add1 = dml.add(batch_norm7, batch_norm10)

# conv11
conv11_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [144,24,1,1]), "mobilenetv20_features_linearbottleneck3_conv0_weight.npy")
conv11_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,144,1,1]), "")
conv11 = dml.convolution(add1, conv11_filter, conv11_bias)

# batch_norm11
batch_norm11_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm0_running_mean.npy")
batch_norm11_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm0_running_var.npy")
batch_norm11_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm0_gamma.npy")
batch_norm11_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm0_beta.npy")
batch_norm11 = dml.batch_normalization(conv11, batch_norm11_mean, batch_norm11_variance, batch_norm11_scale, batch_norm11_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv12
conv12_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [144,1,3,3]), "mobilenetv20_features_linearbottleneck3_conv1_weight.npy")
conv12_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,144,1,1]), "")
conv12 = dml.convolution(batch_norm11, conv12_filter, conv12_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1], group_count = 144)

# batch_norm12
batch_norm12_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm1_running_mean.npy")
batch_norm12_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm1_running_var.npy")
batch_norm12_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm1_gamma.npy")
batch_norm12_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,144,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm1_beta.npy")
batch_norm12 = dml.batch_normalization(conv12, batch_norm12_mean, batch_norm12_variance, batch_norm12_scale, batch_norm12_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv13
conv13_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [32,144,1,1]), "mobilenetv20_features_linearbottleneck3_conv2_weight.npy")
conv13_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,32,1,1]), "")
conv13 = dml.convolution(batch_norm12, conv13_filter, conv13_bias)

# batch_norm13
batch_norm13_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm2_running_mean.npy")
batch_norm13_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm2_running_var.npy")
batch_norm13_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm2_gamma.npy")
batch_norm13_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck3_batchnorm2_beta.npy")
batch_norm13 = dml.batch_normalization(conv13, batch_norm13_mean, batch_norm13_variance, batch_norm13_scale, batch_norm13_bias, 1, 0.000009999999747378752)

# conv14
conv14_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [192,32,1,1]), "mobilenetv20_features_linearbottleneck4_conv0_weight.npy")
conv14_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,192,1,1]), "")
conv14 = dml.convolution(batch_norm13, conv14_filter, conv14_bias)

# batch_norm14
batch_norm14_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm0_running_mean.npy")
batch_norm14_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm0_running_var.npy")
batch_norm14_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm0_gamma.npy")
batch_norm14_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm0_beta.npy")
batch_norm14 = dml.batch_normalization(conv14, batch_norm14_mean, batch_norm14_variance, batch_norm14_scale, batch_norm14_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv15
conv15_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [192,1,3,3]), "mobilenetv20_features_linearbottleneck4_conv1_weight.npy")
conv15_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,192,1,1]), "")
conv15 = dml.convolution(batch_norm14, conv15_filter, conv15_bias, start_padding = [1,1], end_padding = [1,1], group_count = 192)

# batch_norm15
batch_norm15_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm1_running_mean.npy")
batch_norm15_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm1_running_var.npy")
batch_norm15_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm1_gamma.npy")
batch_norm15_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm1_beta.npy")
batch_norm15 = dml.batch_normalization(conv15, batch_norm15_mean, batch_norm15_variance, batch_norm15_scale, batch_norm15_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv16
conv16_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [32,192,1,1]), "mobilenetv20_features_linearbottleneck4_conv2_weight.npy")
conv16_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,32,1,1]), "")
conv16 = dml.convolution(batch_norm15, conv16_filter, conv16_bias)

# batch_norm16
batch_norm16_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm2_running_mean.npy")
batch_norm16_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm2_running_var.npy")
batch_norm16_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm2_gamma.npy")
batch_norm16_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck4_batchnorm2_beta.npy")
batch_norm16 = dml.batch_normalization(conv16, batch_norm16_mean, batch_norm16_variance, batch_norm16_scale, batch_norm16_bias, 1, 0.000009999999747378752)

# add2
add2 = dml.add(batch_norm13, batch_norm16)

# conv17
conv17_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [192,32,1,1]), "mobilenetv20_features_linearbottleneck5_conv0_weight.npy")
conv17_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,192,1,1]), "")
conv17 = dml.convolution(add2, conv17_filter, conv17_bias)

# batch_norm17
batch_norm17_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm0_running_mean.npy")
batch_norm17_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm0_running_var.npy")
batch_norm17_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm0_gamma.npy")
batch_norm17_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm0_beta.npy")
batch_norm17 = dml.batch_normalization(conv17, batch_norm17_mean, batch_norm17_variance, batch_norm17_scale, batch_norm17_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv18
conv18_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [192,1,3,3]), "mobilenetv20_features_linearbottleneck5_conv1_weight.npy")
conv18_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,192,1,1]), "")
conv18 = dml.convolution(batch_norm17, conv18_filter, conv18_bias, start_padding = [1,1], end_padding = [1,1], group_count = 192)

# batch_norm18
batch_norm18_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm1_running_mean.npy")
batch_norm18_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm1_running_var.npy")
batch_norm18_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm1_gamma.npy")
batch_norm18_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm1_beta.npy")
batch_norm18 = dml.batch_normalization(conv18, batch_norm18_mean, batch_norm18_variance, batch_norm18_scale, batch_norm18_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv19
conv19_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [32,192,1,1]), "mobilenetv20_features_linearbottleneck5_conv2_weight.npy")
conv19_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,32,1,1]), "")
conv19 = dml.convolution(batch_norm18, conv19_filter, conv19_bias)

# batch_norm19
batch_norm19_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm2_running_mean.npy")
batch_norm19_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm2_running_var.npy")
batch_norm19_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm2_gamma.npy")
batch_norm19_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,32,1,1]), "mobilenetv20_features_linearbottleneck5_batchnorm2_beta.npy")
batch_norm19 = dml.batch_normalization(conv19, batch_norm19_mean, batch_norm19_variance, batch_norm19_scale, batch_norm19_bias, 1, 0.000009999999747378752)

#add3
add3 = dml.add(add2, batch_norm19)

# conv20
conv20_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [192,32,1,1]), "mobilenetv20_features_linearbottleneck6_conv0_weight.npy")
conv20_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,192,1,1]), "")
conv20 = dml.convolution(add3, conv20_filter, conv20_bias)

# batch_norm20
batch_norm20_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm0_running_mean.npy")
batch_norm20_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm0_running_var.npy")
batch_norm20_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm0_gamma.npy")
batch_norm20_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm0_beta.npy")
batch_norm20 = dml.batch_normalization(conv20, batch_norm20_mean, batch_norm20_variance, batch_norm20_scale, batch_norm20_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv21
conv21_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [192,1,3,3]), "mobilenetv20_features_linearbottleneck6_conv1_weight.npy")
conv21_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,192,1,1]), "")
conv21 = dml.convolution(batch_norm20, conv21_filter, conv21_bias, start_padding = [1,1], end_padding = [1,1], group_count = 192)

# batch_norm21
batch_norm21_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm1_running_mean.npy")
batch_norm21_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm1_running_var.npy")
batch_norm21_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm1_gamma.npy")
batch_norm21_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,192,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm1_beta.npy")
batch_norm21 = dml.batch_normalization(conv21, batch_norm21_mean, batch_norm21_variance, batch_norm21_scale, batch_norm21_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv22
conv22_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [64,192,1,1]), "mobilenetv20_features_linearbottleneck6_conv2_weight.npy")
conv22_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,64,1,1]), "")
conv22 = dml.convolution(batch_norm21, conv22_filter, conv22_bias)

# batch_norm22
batch_norm22_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm2_running_mean.npy")
batch_norm22_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm2_running_var.npy")
batch_norm22_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm2_gamma.npy")
batch_norm22_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck6_batchnorm2_beta.npy")
batch_norm22 = dml.batch_normalization(conv22, batch_norm22_mean, batch_norm22_variance, batch_norm22_scale, batch_norm22_bias, 1, 0.000009999999747378752)

# conv23
conv23_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [384,64,1,1]), "mobilenetv20_features_linearbottleneck7_conv0_weight.npy")
conv23_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,384,1,1]), "")
conv23 = dml.convolution(batch_norm22, conv23_filter, conv23_bias)

# batch_norm23
batch_norm23_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm0_running_mean.npy")
batch_norm23_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm0_running_var.npy")
batch_norm23_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm0_gamma.npy")
batch_norm23_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm0_beta.npy")
batch_norm23 = dml.batch_normalization(conv23, batch_norm23_mean, batch_norm23_variance, batch_norm23_scale, batch_norm23_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv24
conv24_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [384,1,3,3]), "mobilenetv20_features_linearbottleneck7_conv1_weight.npy")
conv24_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,384,1,1]), "")
conv24 = dml.convolution(batch_norm23, conv24_filter, conv24_bias, start_padding = [1,1], end_padding = [1,1], group_count = 384)

# batch_norm24
batch_norm24_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm1_running_mean.npy")
batch_norm24_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm1_running_var.npy")
batch_norm24_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm1_gamma.npy")
batch_norm24_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm1_beta.npy")
batch_norm24 = dml.batch_normalization(conv24, batch_norm24_mean, batch_norm24_variance, batch_norm24_scale, batch_norm24_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv25
conv25_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [64,384,1,1]), "mobilenetv20_features_linearbottleneck7_conv2_weight.npy")
conv25_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,64,1,1]), "")
conv25 = dml.convolution(batch_norm24, conv25_filter, conv25_bias)

# batch_norm25
batch_norm25_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm2_running_mean.npy")
batch_norm25_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm2_running_var.npy")
batch_norm25_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm2_gamma.npy")
batch_norm25_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck7_batchnorm2_beta.npy")
batch_norm25 = dml.batch_normalization(conv25, batch_norm25_mean, batch_norm25_variance, batch_norm25_scale, batch_norm25_bias, 1, 0.000009999999747378752)

# add4
add4 = dml.add(batch_norm22, batch_norm25)

# conv26
conv26_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [384,64,1,1]), "mobilenetv20_features_linearbottleneck8_conv0_weight.npy")
conv26_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,384,1,1]), "")
conv26 = dml.convolution(add4, conv26_filter, conv26_bias)

# batch_norm26
batch_norm26_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm0_running_mean.npy")
batch_norm26_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm0_running_var.npy")
batch_norm26_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm0_gamma.npy")
batch_norm26_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm0_beta.npy")
batch_norm26 = dml.batch_normalization(conv26, batch_norm26_mean, batch_norm26_variance, batch_norm26_scale, batch_norm26_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv27
conv27_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [384,1,3,3]), "mobilenetv20_features_linearbottleneck8_conv1_weight.npy")
conv27_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,384,1,1]), "")
conv27 = dml.convolution(batch_norm26, conv27_filter, conv27_bias, start_padding = [1,1], end_padding = [1,1], group_count = 384)

# batch_norm27
batch_norm27_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm1_running_mean.npy")
batch_norm27_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm1_running_var.npy")
batch_norm27_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm1_gamma.npy")
batch_norm27_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm1_beta.npy")
batch_norm27 = dml.batch_normalization(conv27, batch_norm27_mean, batch_norm27_variance, batch_norm27_scale, batch_norm27_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv28
conv28_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [64,384,1,1]), "mobilenetv20_features_linearbottleneck8_conv2_weight.npy")
conv28_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,64,1,1]), "")
conv28 = dml.convolution(batch_norm27, conv28_filter, conv28_bias)

# batch_norm28
batch_norm28_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm2_running_mean.npy")
batch_norm28_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm2_running_var.npy")
batch_norm28_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm2_gamma.npy")
batch_norm28_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck8_batchnorm2_beta.npy")
batch_norm28 = dml.batch_normalization(conv28, batch_norm28_mean, batch_norm28_variance, batch_norm28_scale, batch_norm28_bias, 1, 0.000009999999747378752)

# add5
add5 = dml.add(add4, batch_norm28)

# conv29
conv29_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [384,64,1,1]), "mobilenetv20_features_linearbottleneck9_conv0_weight.npy")
conv29_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,384,1,1]), "")
conv29 = dml.convolution(add5, conv29_filter, conv29_bias)

# batch_norm29
batch_norm29_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm0_running_mean.npy")
batch_norm29_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm0_running_var.npy")
batch_norm29_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm0_gamma.npy")
batch_norm29_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm0_beta.npy")
batch_norm29 = dml.batch_normalization(conv29, batch_norm29_mean, batch_norm29_variance, batch_norm29_scale, batch_norm29_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv30
conv30_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [384,1,3,3]), "mobilenetv20_features_linearbottleneck9_conv1_weight.npy")
conv30_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,384,1,1]), "")
conv30 = dml.convolution(batch_norm29, conv30_filter, conv30_bias, start_padding = [1,1], end_padding = [1,1], group_count = 384)

# batch_norm30
batch_norm30_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm1_running_mean.npy")
batch_norm30_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm1_running_var.npy")
batch_norm30_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm1_gamma.npy")
batch_norm30_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm1_beta.npy")
batch_norm30 = dml.batch_normalization(conv30, batch_norm30_mean, batch_norm30_variance, batch_norm30_scale, batch_norm30_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv31
conv31_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [64,384,1,1]), "mobilenetv20_features_linearbottleneck9_conv2_weight.npy")
conv31_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,64,1,1]), "")
conv31 = dml.convolution(batch_norm30, conv31_filter, conv31_bias)

# batch_norm31
batch_norm31_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm2_running_mean.npy")
batch_norm31_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm2_running_var.npy")
batch_norm31_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm2_gamma.npy")
batch_norm31_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,64,1,1]), "mobilenetv20_features_linearbottleneck9_batchnorm2_beta.npy")
batch_norm31 = dml.batch_normalization(conv31, batch_norm31_mean, batch_norm31_variance, batch_norm31_scale, batch_norm31_bias, 1, 0.000009999999747378752)

# add6
add6 = dml.add(add5, batch_norm31)

# conv32
conv32_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [384,64,1,1]), "mobilenetv20_features_linearbottleneck10_conv0_weight.npy")
conv32_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,384,1,1]), "")
conv32 = dml.convolution(add6, conv32_filter, conv32_bias)

# batch_norm32
batch_norm32_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm0_running_mean.npy")
batch_norm32_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm0_running_var.npy")
batch_norm32_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm0_gamma.npy")
batch_norm32_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm0_beta.npy")
batch_norm32 = dml.batch_normalization(conv32, batch_norm32_mean, batch_norm32_variance, batch_norm32_scale, batch_norm32_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv33
conv33_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [384,1,3,3]), "mobilenetv20_features_linearbottleneck10_conv1_weight.npy")
conv33_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,384,1,1]), "")
conv33 = dml.convolution(batch_norm32, conv33_filter, conv33_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1], group_count = 384)

# batch_norm33
batch_norm33_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm1_running_mean.npy")
batch_norm33_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm1_running_var.npy")
batch_norm33_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm1_gamma.npy")
batch_norm33_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,384,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm1_beta.npy")
batch_norm33 = dml.batch_normalization(conv33, batch_norm33_mean, batch_norm33_variance, batch_norm33_scale, batch_norm33_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv34
conv34_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [96,384,1,1]), "mobilenetv20_features_linearbottleneck10_conv2_weight.npy")
conv34_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,96,1,1]), "")
conv34 = dml.convolution(batch_norm33, conv34_filter, conv34_bias)

# batch_norm34
batch_norm34_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm2_running_mean.npy")
batch_norm34_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm2_running_var.npy")
batch_norm34_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm2_gamma.npy")
batch_norm34_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck10_batchnorm2_beta.npy")
batch_norm34 = dml.batch_normalization(conv34, batch_norm34_mean, batch_norm34_variance, batch_norm34_scale, batch_norm34_bias, 1, 0.000009999999747378752)

# conv35
conv35_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [576,96,1,1]), "mobilenetv20_features_linearbottleneck11_conv0_weight.npy")
conv35_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,576,1,1]), "")
conv35 = dml.convolution(batch_norm34, conv35_filter, conv35_bias)

# batch_norm35
batch_norm35_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm0_running_mean.npy")
batch_norm35_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm0_running_var.npy")
batch_norm35_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm0_gamma.npy")
batch_norm35_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm0_beta.npy")
batch_norm35 = dml.batch_normalization(conv35, batch_norm35_mean, batch_norm35_variance, batch_norm35_scale, batch_norm35_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv36
conv36_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [576,1,3,3]), "mobilenetv20_features_linearbottleneck11_conv1_weight.npy")
conv36_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,576,1,1]), "")
conv36 = dml.convolution(batch_norm35, conv36_filter, conv36_bias, start_padding = [1,1], end_padding = [1,1], group_count = 576)

# batch_norm36
batch_norm36_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm1_running_mean.npy")
batch_norm36_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm1_running_var.npy")
batch_norm36_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm1_gamma.npy")
batch_norm36_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm1_beta.npy")
batch_norm36 = dml.batch_normalization(conv36, batch_norm36_mean, batch_norm36_variance, batch_norm36_scale, batch_norm36_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv37
conv37_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [96,576,1,1]), "mobilenetv20_features_linearbottleneck11_conv2_weight.npy")
conv37_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,96,1,1]), "")
conv37 = dml.convolution(batch_norm36, conv37_filter, conv37_bias)

# batch_norm37
batch_norm37_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm2_running_mean.npy")
batch_norm37_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm2_running_var.npy")
batch_norm37_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm2_gamma.npy")
batch_norm37_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck11_batchnorm2_beta.npy")
batch_norm37 = dml.batch_normalization(conv37, batch_norm37_mean, batch_norm37_variance, batch_norm37_scale, batch_norm37_bias, 1, 0.000009999999747378752)

# add7
add7 = dml.add(batch_norm34, batch_norm37)

# conv38
conv38_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [576,96,1,1]), "mobilenetv20_features_linearbottleneck12_conv0_weight.npy")
conv38_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,576,1,1]), "")
conv38 = dml.convolution(add7, conv38_filter, conv38_bias)

# batch_norm38
batch_norm38_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm0_running_mean.npy")
batch_norm38_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm0_running_var.npy")
batch_norm38_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm0_gamma.npy")
batch_norm38_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm0_beta.npy")
batch_norm38 = dml.batch_normalization(conv38, batch_norm38_mean, batch_norm38_variance, batch_norm38_scale, batch_norm38_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv39
conv39_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [576,1,3,3]), "mobilenetv20_features_linearbottleneck12_conv1_weight.npy")
conv39_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,576,1,1]), "")
conv39 = dml.convolution(batch_norm38, conv39_filter, conv39_bias, start_padding = [1,1], end_padding = [1,1], group_count = 576)

# batch_norm39
batch_norm39_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm1_running_mean.npy")
batch_norm39_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm1_running_var.npy")
batch_norm39_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm1_gamma.npy")
batch_norm39_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]),"mobilenetv20_features_linearbottleneck12_batchnorm1_beta.npy")
batch_norm39 = dml.batch_normalization(conv39, batch_norm39_mean, batch_norm39_variance, batch_norm39_scale, batch_norm39_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv40
conv40_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [96,576,1,1]), "mobilenetv20_features_linearbottleneck12_conv2_weight.npy")
conv40_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,96,1,1]), "")
conv40 = dml.convolution(batch_norm39, conv40_filter, conv40_bias)

# batch_norm40
batch_norm40_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm2_running_mean.npy")
batch_norm40_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm2_running_var.npy")
batch_norm40_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm2_gamma.npy")
batch_norm40_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,96,1,1]), "mobilenetv20_features_linearbottleneck12_batchnorm2_beta.npy")
batch_norm40 = dml.batch_normalization(conv40, batch_norm40_mean, batch_norm40_variance, batch_norm40_scale, batch_norm40_bias, 1, 0.000009999999747378752)

# add8
add8 = dml.add(add7, batch_norm40)

# conv41
conv41_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [576,96,1,1]), "mobilenetv20_features_linearbottleneck13_conv0_weight.npy")
conv41_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,576,1,1]), "")
conv41 = dml.convolution(add8, conv41_filter, conv41_bias)

# batch_norm41
batch_norm41_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm0_running_mean.npy")
batch_norm41_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm0_running_var.npy")
batch_norm41_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm0_gamma.npy")
batch_norm41_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm0_beta.npy")
batch_norm41 = dml.batch_normalization(conv41, batch_norm41_mean, batch_norm41_variance, batch_norm41_scale, batch_norm41_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv42
conv42_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [576,1,3,3]), "mobilenetv20_features_linearbottleneck13_conv1_weight.npy")
conv42_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,576,1,1]), "")
conv42 = dml.convolution(batch_norm41, conv42_filter, conv42_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1], group_count = 576)

# batch_norm42
batch_norm42_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm1_running_mean.npy")
batch_norm42_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm1_running_var.npy")
batch_norm42_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm1_gamma.npy")
batch_norm42_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,576,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm1_beta.npy")
batch_norm42 = dml.batch_normalization(conv42, batch_norm42_mean, batch_norm42_variance, batch_norm42_scale, batch_norm42_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv43
conv43_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [160,576,1,1]), "mobilenetv20_features_linearbottleneck13_conv2_weight.npy")
conv43_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,160,1,1]), "")
conv43 = dml.convolution(batch_norm42, conv43_filter, conv43_bias)

# batch_norm43
batch_norm43_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm2_running_mean.npy")
batch_norm43_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm2_running_var.npy")
batch_norm43_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm2_gamma.npy")
batch_norm43_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck13_batchnorm2_beta.npy")
batch_norm43 = dml.batch_normalization(conv43, batch_norm43_mean, batch_norm43_variance, batch_norm43_scale, batch_norm43_bias, 1, 0.000009999999747378752)

# conv44
conv44_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [960,160,1,1]), "mobilenetv20_features_linearbottleneck14_conv0_weight.npy")
conv44_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,960,1,1]), "")
conv44 = dml.convolution(batch_norm43, conv44_filter, conv44_bias)

# batch_norm44
batch_norm44_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm0_running_mean.npy")
batch_norm44_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm0_running_var.npy")
batch_norm44_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm0_gamma.npy")
batch_norm44_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm0_beta.npy")
batch_norm44 = dml.batch_normalization(conv44, batch_norm44_mean, batch_norm44_variance, batch_norm44_scale, batch_norm44_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv45
conv45_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [960,1,3,3]), "mobilenetv20_features_linearbottleneck14_conv1_weight.npy")
conv45_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,960,1,1]), "")
conv45 = dml.convolution(batch_norm44, conv45_filter, conv45_bias, start_padding = [1,1], end_padding = [1,1], group_count = 960)

# batch_norm45
batch_norm45_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm1_running_mean.npy")
batch_norm45_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm1_running_var.npy")
batch_norm45_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm1_gamma.npy")
batch_norm45_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm1_beta.npy")
batch_norm45 = dml.batch_normalization(conv45, batch_norm45_mean, batch_norm45_variance, batch_norm45_scale, batch_norm45_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv46
conv46_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [160,960,1,1]), "mobilenetv20_features_linearbottleneck14_conv2_weight.npy")
conv46_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,160,1,1]), "")
conv46 = dml.convolution(batch_norm45, conv46_filter, conv46_bias)

# batch_norm46
batch_norm46_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm2_running_mean.npy")
batch_norm46_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm2_running_var.npy")
batch_norm46_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm2_gamma.npy")
batch_norm46_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck14_batchnorm2_beta.npy")
batch_norm46 = dml.batch_normalization(conv46, batch_norm46_mean, batch_norm46_variance, batch_norm46_scale, batch_norm46_bias, 1, 0.000009999999747378752)

# add9
add9 = dml.add(batch_norm43, batch_norm46)

# conv47
conv47_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [960,160,1,1]), "mobilenetv20_features_linearbottleneck15_conv0_weight.npy")
conv47_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,960,1,1]), "")
conv47 = dml.convolution(add9, conv47_filter, conv47_bias)

# batch_norm47
batch_norm47_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm0_running_mean.npy")
batch_norm47_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm0_running_var.npy")
batch_norm47_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm0_gamma.npy")
batch_norm47_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm0_beta.npy")
batch_norm47 = dml.batch_normalization(conv47, batch_norm47_mean, batch_norm47_variance, batch_norm47_scale, batch_norm47_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv48
conv48_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [960,1,3,3]), "mobilenetv20_features_linearbottleneck15_conv1_weight.npy")
conv48_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,960,1,1]), "")
conv48 = dml.convolution(batch_norm47, conv48_filter, conv48_bias, start_padding = [1,1], end_padding = [1,1], group_count = 960)

# batch_norm48
batch_norm48_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm1_running_mean.npy")
batch_norm48_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm1_running_var.npy")
batch_norm48_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm1_gamma.npy")
batch_norm48_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm1_beta.npy")
batch_norm48 = dml.batch_normalization(conv48, batch_norm48_mean, batch_norm48_variance, batch_norm48_scale, batch_norm48_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv49
conv49_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [160,960,1,1]), "mobilenetv20_features_linearbottleneck15_conv2_weight.npy")
conv49_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,160,1,1]), "")
conv49 = dml.convolution(batch_norm48, conv49_filter, conv49_bias)

# batch_norm49
batch_norm49_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm2_running_mean.npy")
batch_norm49_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm2_running_var.npy")
batch_norm49_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm2_gamma.npy")
batch_norm49_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,160,1,1]), "mobilenetv20_features_linearbottleneck15_batchnorm2_beta.npy")
batch_norm49 = dml.batch_normalization(conv49, batch_norm49_mean, batch_norm49_variance, batch_norm49_scale, batch_norm49_bias, 1, 0.000009999999747378752)

# add10
add10 = dml.add(add9, batch_norm49)

# conv50
conv50_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [960,160,1,1]), "mobilenetv20_features_linearbottleneck16_conv0_weight.npy")
conv50_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,960,1,1]), "")
conv50 = dml.convolution(add10, conv50_filter, conv50_bias)

# batch_norm50
batch_norm50_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm0_running_mean.npy")
batch_norm50_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm0_running_var.npy")
batch_norm50_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm0_gamma.npy")
batch_norm50_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm0_beta.npy")
batch_norm50 = dml.batch_normalization(conv50, batch_norm50_mean, batch_norm50_variance, batch_norm50_scale, batch_norm50_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv51
conv51_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [960,1,3,3]), "mobilenetv20_features_linearbottleneck16_conv1_weight.npy")
conv51_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,960,1,1]), "")
conv51 = dml.convolution(batch_norm50, conv51_filter, conv51_bias, start_padding = [1,1], end_padding = [1,1], group_count = 960)

# batch_norm51
batch_norm51_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm1_running_mean.npy")
batch_norm51_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm1_running_var.npy")
batch_norm51_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm1_gamma.npy")
batch_norm51_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,960,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm1_beta.npy")
batch_norm51 = dml.batch_normalization(conv51, batch_norm51_mean, batch_norm51_variance, batch_norm51_scale, batch_norm51_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# conv52
conv52_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [320,960,1,1]), "mobilenetv20_features_linearbottleneck16_conv2_weight.npy")
conv52_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,320,1,1]), "")
conv52 = dml.convolution(batch_norm51, conv52_filter, conv52_bias)

# batch_norm52
batch_norm52_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,320,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm2_running_mean.npy")
batch_norm52_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,320,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm2_running_var.npy")
batch_norm52_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,320,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm2_gamma.npy")
batch_norm52_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,320,1,1]), "mobilenetv20_features_linearbottleneck16_batchnorm2_beta.npy")
batch_norm52 = dml.batch_normalization(conv52, batch_norm52_mean, batch_norm52_variance, batch_norm52_scale, batch_norm52_bias, 1, 0.000009999999747378752)

# conv53
conv53_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1280,320,1,1]), "mobilenetv20_features_conv1_weight.npy")
conv53_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1,1280,1,1]), "")
conv53 = dml.convolution(batch_norm52, conv53_filter, conv53_bias)

# batch_norm52
batch_norm53_mean = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,1280,1,1]), "mobilenetv20_features_batchnorm1_running_mean.npy")
batch_norm53_variance = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,1280,1,1]), "mobilenetv20_features_batchnorm1_running_var.npy")
batch_norm53_scale = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,1280,1,1]), "mobilenetv20_features_batchnorm1_gamma.npy")
batch_norm53_bias = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, [1,1280,1,1]), "mobilenetv20_features_batchnorm1_beta.npy")
batch_norm53 = dml.batch_normalization(conv53, batch_norm53_mean, batch_norm53_variance, batch_norm53_scale, batch_norm53_bias, 1, 0.000009999999747378752, dml.FusedActivation(dml.OperatorType.ACTIVATION_RELU))

# avg_pool1
avg_pool1 = dml.average_pooling(batch_norm53, [1,1], [7,7], [0,0], [0,0], [0,0], 0)

# conv54
conv54_filter = append_input_tensor(builder, input_bindings, dml.TensorDesc(data_type, flags, [1000,1280,1,1]), "mobilenetv20_output_pred_weight.npy")
conv54 = dml.convolution(avg_pool1, conv54_filter)

# reshape
reshape = dml.reinterpret(conv54, dml.TensorDataType.FLOAT32, [1,1,1,1000], [1000,1000,1000,1])

# softmax
soft_max = dml.activation_soft_max(reshape)

# Compile the expression graph into a compiled operator
op = builder.build(dml.ExecutionFlags.NONE, [soft_max])

# Compute the result
output_data = device.compute(op, input_bindings, [soft_max])
output_tensor = np.array(output_data[0], np.float32)

# Opens text file of categories to collect the correct image category
label_file = open("imagenet_categories.txt","r")
label_lines = label_file.readlines()
prediction_index = np.argmax(output_tensor)

# Print the program confidence and the category from locally stored ImageNet text file
print("\nCategory: {}".format(label_lines[prediction_index], end=''))
print("Confidence: {:2.2f}%".format(np.amax(output_tensor) * 100))
