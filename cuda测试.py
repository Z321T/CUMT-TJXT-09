import torch
import time
import numpy as np


def comprehensive_gpu_test():
    """ GPU 综合性能测试"""
    print("=== GPU 性能测试 ===\n")

    device = torch.device("cuda")
    print(f"测试设备: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}\n")

    # 1. 基础矩阵运算性能
    print("1. 矩阵运算性能测试:")
    sizes = [1024, 2048, 4096, 8192]

    for size in sizes:
        # GPU 测试
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # 预热
        for _ in range(3):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

        # 正式测试
        start_time = time.time()
        for _ in range(10):
            result = torch.mm(a, b)
        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / 10 * 1000  # ms
        flops = 2 * size ** 3 * 10  # 10次操作的总FLOPS
        tflops = flops / (end_time - start_time) / 1e12

        print(f"   {size}x{size}: {avg_time:.2f} ms/次, {tflops:.2f} TFLOPS")

    # 2. 混合精度测试
    print("\n2. 混合精度 (FP16) 测试:")
    size = 4096
    a_fp32 = torch.randn(size, size, device=device)
    b_fp32 = torch.randn(size, size, device=device)
    a_fp16 = a_fp32.half()
    b_fp16 = b_fp32.half()

    # FP32
    torch.cuda.synchronize()
    start = time.time()
    result_fp32 = torch.mm(a_fp32, b_fp32)
    torch.cuda.synchronize()
    fp32_time = time.time() - start

    # FP16
    torch.cuda.synchronize()
    start = time.time()
    result_fp16 = torch.mm(a_fp16, b_fp16)
    torch.cuda.synchronize()
    fp16_time = time.time() - start

    speedup = fp32_time / fp16_time
    print(f"   FP32: {fp32_time * 1000:.2f} ms")
    print(f"   FP16: {fp16_time * 1000:.2f} ms")
    print(f"   FP16 加速比: {speedup:.2f}x")

    # 3. 深度学习模拟测试
    print("\n3. 深度学习模拟测试:")

    # 模拟卷积网络
    conv_layers = [
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.Conv2d(128, 256, 3, padding=1),
        torch.nn.Conv2d(256, 512, 3, padding=1),
    ]

    for layer in conv_layers:
        layer = layer.to(device)

    batch_sizes = [1, 8, 16, 32]
    input_size = 224

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, input_size, input_size, device=device)

        torch.cuda.synchronize()
        start = time.time()

        # 前向传播
        for layer in conv_layers:
            x = layer(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)

        torch.cuda.synchronize()
        end = time.time()

        throughput = batch_size / (end - start)
        print(f"   Batch size {batch_size:2d}: {(end - start) * 1000:.2f} ms, {throughput:.1f} imgs/sec")

    # 4. 内存带宽测试
    print("\n4. 显存带宽测试:")
    sizes_mb = [100, 500, 1000, 2000]

    for size_mb in sizes_mb:
        elements = size_mb * 1024 * 1024 // 4  # float32
        data = torch.randn(elements, device=device)

        torch.cuda.synchronize()
        start = time.time()

        # 内存复制测试
        for _ in range(10):
            data_copy = data.clone()

        torch.cuda.synchronize()
        end = time.time()

        bandwidth = (size_mb * 10 * 2) / (end - start) / 1024  # GB/s (读+写)
        print(f"   {size_mb} MB: {bandwidth:.1f} GB/s")

    # 5. Transformer 模拟测试
    print("\n5. Transformer 注意力机制测试:")

    seq_lengths = [512, 1024, 2048]
    d_model = 768
    num_heads = 12

    for seq_len in seq_lengths:
        # 模拟多头注意力
        batch_size = 8
        q = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads, device=device)

        torch.cuda.synchronize()
        start = time.time()

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_model // num_heads)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        torch.cuda.synchronize()
        end = time.time()

        print(f"   序列长度 {seq_len}: {(end - start) * 1000:.2f} ms")


def memory_stress_test():
    """显存压力测试"""
    print("\n=== 显存压力测试 ===")

    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"显卡总显存: {total_memory:.1f} GB")

    # 逐步增加显存使用
    tensors = []
    allocated_gb = 0

    try:
        while allocated_gb < total_memory * 0.9:  # 使用90%显存
            # 每次分配 100MB
            tensor = torch.randn(100 * 1024 * 1024 // 4, device=device)
            tensors.append(tensor)
            allocated_gb += 0.1

            current_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"\r已分配显存: {current_allocated:.1f} GB", end="", flush=True)

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n显存不足，最大可用: {torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")
        else:
            print(f"\n错误: {e}")

    # 清理显存
    del tensors
    torch.cuda.empty_cache()
    print(f"\n显存已清理")


if __name__ == "__main__":
    if torch.cuda.is_available():
        comprehensive_gpu_test()
        memory_stress_test()
        print(f"\n🎉 性能测试完成！")
    else:
        print("❌ CUDA 不可用")