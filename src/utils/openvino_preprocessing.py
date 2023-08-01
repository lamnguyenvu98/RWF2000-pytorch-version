import numpy as np
import openvino.runtime.opset11 as ov
from openvino.runtime import Output
from openvino.runtime.utils.decorators import custom_preprocess_function

INT_MAX = np.iinfo(np.int32).max

def mean(node, axes: list, keep_dims: bool = True):
    return ov.reduce_mean(node, reduction_axes=axes, keep_dims=keep_dims)

def slicing(node, start: list, stop: list, step: list):
    start_ind = ov.constant(start, dtype=np.int32)
    stop_ind = ov.constant(stop, dtype=np.int32)
    step_ind = ov.constant(step, dtype=np.int32)
    return ov.slice(node, start=start_ind, stop=stop_ind, step=step_ind)

def std(node, mean, axes):
    square_diff = ov.squared_difference(node, mean)
    # ax = ov.constant(axes, dtype=np.int32)
    return ov.sqrt(ov.reduce_mean(square_diff, reduction_axes=axes, keep_dims=True))

def normalize(node, mean, std):
    return ov.divide(ov.subtract(node, mean), std)

@custom_preprocess_function
def custom_normalizer(output: Output):
    rgb_slice = slicing(output, start=[0,0,0,0,0], stop=[INT_MAX,3,64,224,224], step=[1,1,1,1,1])
    flow_slice = slicing(output, start=[0,3,0,0,0], stop=[INT_MAX,5,64,224,224], step=[1,1,1,1,1])

    mean_rgb = mean(rgb_slice, axes=[1, 2, 3, 4], keep_dims=True)
    mean_flow = mean(flow_slice, axes=[1, 2, 3, 4], keep_dims=True)
    
    std_rgb = std(rgb_slice, mean_rgb, axes=[1, 2, 3, 4])
    std_flow = std(flow_slice, mean_flow, axes=[1, 2, 3, 4])


    norm_rgb = normalize(rgb_slice, mean_rgb, std_rgb)
    norm_flow = normalize(flow_slice, mean_flow, std_flow)

    return ov.concat([norm_rgb, norm_flow], axis=1)

@custom_preprocess_function
def custom_softmax(output: Output):
    return ov.softmax(output, axis=1)