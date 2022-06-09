import torch
import time
from typing import List
# TorchScript often needs some help with type information, let's provide it with the type of inputs it needs to expect
@torch.jit.script
def composite_definition(input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor, bias1: torch.Tensor, bias2: torch.Tensor, normalization_axis: int, dropout_prob: float, keepdim: bool):
    __constants__ = ['normalization_axis', 'dropout_prob', 'keepdim']
    bias1_out = input1 + bias1
    dropout_out = torch.nn.functional.dropout(bias1_out, dropout_prob)
    norm_input = dropout_out + input2
    norm_output = torch.nn.functional.layer_norm(norm_input, (input1.size(normalization_axis),), weight, bias2)
    return norm_output

# Setup initial tensors and parameters
input_size = [64, 128, 1024]
device = "cuda"
dtype = torch.float32

weight = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias1 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias2 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))

inputs1 = []
inputs2 = []
grad_outputs = []

import random
random.seed(0)
# Create 20 random shapes to run through the network
shape_count = 20
for _ in range(shape_count):
    input_size[0] = input_size[0] + random.randrange(-2, 3)
    input_size[1] = input_size[1] + random.randrange(-2, 3)
    input = torch.randn(*input_size, device=device, dtype=dtype, requires_grad=True)
    inputs1.append(input)
    inputs2.append(torch.rand_like(input))
    grad_outputs.append(torch.rand_like(input))

# Perform warm-up iterations
for _ in range(3):
    input1 = inputs1[0]
    input2 = inputs2[0]
    grad_output = grad_outputs[0]
    # Run model, forward and backward
    output = composite_definition(input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1, keepdim=True)
    output.backward(grad_output)

iteration_count = 100
# Synchronize the GPU before starting the timer
torch.cuda.synchronize()
start = time.perf_counter()
for i in range(iteration_count):
    input1 = inputs1[i % shape_count]
    input2 = inputs2[i % shape_count]
    grad_output = grad_outputs[i % shape_count]

    # Run model, forward and backward
    output = composite_definition(input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1, keepdim=True)
    output.backward(grad_output)

# Synchronize the GPU before stopping the timer
torch.cuda.synchronize()
stop = time.perf_counter()
print("Average iteration time: {:.2f} ms".format((stop - start) * 1000 / iteration_count))

# Average iteration time: 0.78 ms
