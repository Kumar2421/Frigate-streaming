# Inference Benchmark Report

## Model Performance Comparison

| Model Format | Inference Time |
|--------------|----------------|
| yolol.pt     | 2.956 s        |
| yolol.onnx   | 1.481 s        |
| yolol.engine | 0.731 s        |

## Summary

- **PyTorch (.pt)**: 2.956 seconds - Baseline PyTorch model inference
- **ONNX (.onnx)**: 1.481 seconds - ~50% faster than PyTorch
- **TensorRT (.engine)**: 0.731 seconds - ~75% faster than PyTorch, ~50% faster than ONNX

## Performance Gains

- **ONNX vs PyTorch**: 50% reduction in inference time
- **TensorRT vs PyTorch**: 75% reduction in inference time
- **TensorRT vs ONNX**: 50% reduction in inference time

## Environment

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- CUDA: 12.1
- PyTorch: 2.5.1+cu121
- TensorRT: 10.11.0.33

## Notes

TensorRT provides the best inference performance with significant speedup over both PyTorch and ONNX formats, making it ideal for production deployments where inference latency is critical.
