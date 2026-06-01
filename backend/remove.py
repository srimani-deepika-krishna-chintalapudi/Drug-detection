import paddle

print("Paddle:", paddle.__version__)
print("CUDA:", paddle.device.is_compiled_with_cuda())
print("Device:", paddle.device.get_device())