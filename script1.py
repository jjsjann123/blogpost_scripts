import torch
import time

from typing import List
@torch.jit.script
def composite_definition(input1, input2, weight, bias1, bias2, normalization_axis, dropout_prob):
    bias1_out = input1 + bias1
    dropout_out = torch.nn.functional.dropout(bias1_out, dropout_prob)
    norm_input = dropout_out + input2
    norm_output = torch.nn.functional.layer_norm(norm_input, (input1.size(normalization_axis),), weight, bias2)
    return norm_output

# Setup initial tensors and parameters
input_size = [64, 128, 1024]
device = "cuda"
dtype = torch.float32

# Create sample inputs
input1 = torch.randn(*input_size, device=device, dtype=dtype, requires_grad=True)
input2 = torch.rand_like(input1)

# Precompute a grad output tensor, for this example it's the same size as the inputs
grad_output = torch.rand_like(input1)

weight = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias1 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))
bias2 = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))

# Perform warm-up iterations
for _ in range(3):
    # Run model, forward and backward
    output = composite_definition(input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1)
    output.backward(grad_output)

iteration_count = 100
# Synchronize the GPU before starting the timer
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(iteration_count):
    # Run model, forward and backward
    output = composite_definition(input1, input2, weight, bias1, bias2, normalization_axis=2, dropout_prob=0.1)
    output.backward(grad_output)

# Synchronize the GPU before stopping the timer
torch.cuda.synchronize()
stop = time.perf_counter()
print("Average iteration time: {:.2f} ms".format((stop - start) * 1000 / iteration_count))

# Average iteration time: 0.65 ms
