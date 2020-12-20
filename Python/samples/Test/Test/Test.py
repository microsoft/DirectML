import pydirectml as dml
import numpy as np

device = dml.Device(use_debug_layer = True)
builder = dml.GraphBuilder(device)
data_type = dml.TensorDataType.FLOAT32
flags = dml.TensorFlags.OWNED_BY_DML
input_bindings = []
a = dml.input_tensor(builder, 0, dml.TensorDesc(data_type, [1, 1, 3, 4]))
b = dml.input_tensor(builder, 1, dml.TensorDesc(data_type, flags, [1, 1, 4, 3]))
c = dml.gemm(a, b)
#c = dml.activation_identity(c)
input_bindings.append(dml.Binding(a, np.ones([1, 1, 3, 4], dtype=np.float32)))
input_bindings.append(dml.Binding(b, np.ones([1, 1, 4, 3], dtype=np.float32)))
op = builder.build(dml.ExecutionFlags.NONE, [c])
output_data = device.compute(op, input_bindings, [c])
output_tensor = np.array(output_data[0], np.float32)
print(output_tensor)
