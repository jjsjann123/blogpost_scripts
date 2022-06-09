import torch
import time
from typing import List

@torch.jit.script
def with_rms_norm(input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, normalization_axis: int, dropout_prob: float, keepdim: bool):
    __constants__ = ['normalization_axis', 'dropout_prob', 'keepdim']
    bias_out = input1 + bias
    dropout_out = torch.nn.functional.dropout(bias_out, dropout_prob)
    norm_input = dropout_out + input2
    var = norm_input.mul(norm_input).mean(normalization_axis, keepdim=keepdim)
    pre_shift_scale_norm_output = norm_input / torch.sqrt(var + 1e-12)
    norm_output = weight * pre_shift_scale_norm_output
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
bias = torch.nn.Parameter(torch.randn(input_size[2], dtype=dtype, device=device))

# Perform warm-up iterations
for _ in range(3):
    # Run model, forward and backward
    output = with_rms_norm(input1, input2, weight, bias, normalization_axis=2, dropout_prob=0.1, keepdim=True)
    output.backward(grad_output)

iteration_count = 100
# Synchronize the GPU before starting the timer
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(iteration_count):
    # Run model, forward and backward
    output = with_rms_norm(input1, input2, weight, bias, normalization_axis=2, dropout_prob=0.1, keepdim=True)
    output.backward(grad_output)

# Synchronize the GPU before stoping the timer
torch.cuda.synchronize()
stop = time.perf_counter()
print("Average iteration time: {:.2f} ms".format((stop - start) * 1000 / iteration_count))

# Average iteration time: 0.97 ms
