#!/usr/bin/env python3
"""
Model Profiling Script for SincNet vs SSL Comparison
Measures computational complexity, memory usage, and inference speed.
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import importlib
import pandas as pd
import psutil
from args_config import get_args

__author__ = "Model Profiling for Anti-Spoofing"
__email__ = "comparison@antispoofing.ai"


def profile_model_complexity(model, device, input_shape=(1, 64000)):
    """Profile model computational complexity"""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    print("=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)

    # 1. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    print(f"💾 Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")

    # 2. Memory Usage
    torch.cuda.empty_cache()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output = model(dummy_input)

    memory_used = 0
    if device == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"🔥 GPU Memory Used: {memory_used:.2f} MB")
    else:
        # Estimate CPU memory usage
        process = psutil.Process(os.getpid())
        memory_used = process.memory_info().rss / 1024 / 1024
        print(f"💻 CPU Memory Used: {memory_used:.2f} MB")

    # 3. Inference Time
    model.eval()
    times = []

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Actual timing
    if device == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(100):
            start_time = time.time()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

    avg_inference_time = np.mean(times) * 1000  # Convert to ms
    std_inference_time = np.std(times) * 1000

    print(
        f"⚡ Avg Inference Time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms"
    )
    print(f"🚀 Throughput: {1000/avg_inference_time:.1f} samples/sec")

    # 4. FLOPs Estimation (optional)
    try:
        from torchprofile import profile_macs

        macs = profile_macs(model, dummy_input)
        print(f"🔢 FLOPs: {macs:,}")
        print(f"📈 GFLOPs: {macs / 1e9:.2f}")
    except ImportError:
        print("⚠️ Install torchprofile for FLOP analysis: pip install torchprofile")
        macs = 0

    print("=" * 60)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / 1024 / 1024,
        "memory_mb": memory_used,
        "inference_time_ms": avg_inference_time,
        "inference_std_ms": std_inference_time,
        "throughput": 1000 / avg_inference_time,
        "flops": macs if "macs" in locals() else 0,
    }


def compare_models_complexity():
    """Compare SincNet vs SSL model complexity"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    # Test both models
    models_to_test = [
        ("SincNet + SA", "sincnet_model"),
        ("SincNet (No SA)", "sincnet_model_without_sa"),
        ("SSL + SA", "model"),
        ("SSL (No SA)", "model_without_sa"),
    ]

    results = {}

    for model_name, module_name in models_to_test:
        try:
            print(f"\n🔍 Testing: {model_name}")
            print("-" * 40)

            # Import model
            model_module = importlib.import_module(module_name)
            Model = getattr(model_module, "Model")

            # Create dummy args
            class DummyArgs:
                def __init__(self):
                    self.algo = 0
                    self.seed = 42
                    # Add other required args as needed
                    pass

            args = DummyArgs()
            model = Model(args, device).to(device)

            # Profile the model
            results[model_name] = profile_model_complexity(model, device)

            # Cleanup
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            print(f"   Module: {module_name}")
            continue

    if not results:
        print("❌ No models could be profiled successfully!")
        return {}

    # Create comparison table
    print("\n" + "=" * 80)
    print("📋 MODEL COMPARISON TABLE")
    print("=" * 80)

    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append(
            {
                "Model": model_name,
                "Parameters (M)": f"{metrics['total_params']/1e6:.2f}M",
                "Size (MB)": f"{metrics['model_size_mb']:.2f}",
                "Memory (MB)": f"{metrics['memory_mb']:.2f}",
                "Inference (ms)": f"{metrics['inference_time_ms']:.2f} ± {metrics['inference_std_ms']:.2f}",
                "Throughput (sps)": f"{metrics['throughput']:.1f}",
                "GFLOPs": (
                    f"{metrics['flops']/1e9:.2f}" if metrics["flops"] > 0 else "N/A"
                ),
            }
        )

    # Save to Excel
    df = pd.DataFrame(comparison_data)
    excel_path = "model_complexity_comparison.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"📊 Comparison saved to: {excel_path}")

    # Print table
    print(df.to_string(index=False))

    # Calculate efficiency ratios
    print("\n" + "=" * 80)
    print("⚖️ EFFICIENCY ANALYSIS")
    print("=" * 80)

    if len(results) >= 2:
        model_names = list(results.keys())
        if "SincNet + SA" in results and "SSL + SA" in results:
            sincnet = results["SincNet + SA"]
            ssl = results["SSL + SA"]

            param_ratio = ssl["total_params"] / sincnet["total_params"]
            size_ratio = ssl["model_size_mb"] / sincnet["model_size_mb"]
            memory_ratio = (
                ssl["memory_mb"] / sincnet["memory_mb"]
                if sincnet["memory_mb"] > 0
                else 1
            )
            speed_ratio = ssl["inference_time_ms"] / sincnet["inference_time_ms"]

            print(f"📊 SSL vs SincNet Ratios:")
            print(f"   Parameters: {param_ratio:.1f}x larger")
            print(f"   Model Size: {size_ratio:.1f}x larger")
            print(f"   Memory Usage: {memory_ratio:.1f}x more")
            print(f"   Inference Time: {speed_ratio:.1f}x slower")
            print(f"\n🏆 SincNet is {param_ratio:.1f}x more parameter-efficient!")
            print(f"🚀 SincNet is {speed_ratio:.1f}x faster!")

    return results


def benchmark_training_step(model, device, batch_size=32):
    """Benchmark a single training step"""
    model.train()
    dummy_input = torch.randn(batch_size, 64000).to(device)
    dummy_target = torch.randint(0, 2, (batch_size,)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()

    # Benchmark
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()

    for _ in range(10):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()

    avg_step_time = (end_time - start_time) / 10 * 1000  # ms per step

    return avg_step_time


def full_benchmark():
    """Run comprehensive benchmark comparing all models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🚀 STARTING COMPREHENSIVE MODEL BENCHMARK")
    print("=" * 80)

    # Basic complexity comparison
    complexity_results = compare_models_complexity()

    # Training step benchmark
    print("\n📈 TRAINING STEP BENCHMARK")
    print("=" * 60)

    training_results = {}

    for model_name, module_name in [
        ("SincNet + SA", "sincnet_model"),
        ("SSL + SA", "model"),
    ]:
        try:
            print(f"\n🔄 Benchmarking training step: {model_name}")

            model_module = importlib.import_module(module_name)
            Model = getattr(model_module, "Model")

            class DummyArgs:
                def __init__(self):
                    self.algo = 0
                    self.seed = 42

            args = DummyArgs()
            model = Model(args, device).to(device)

            step_time = benchmark_training_step(model, device)
            training_results[model_name] = step_time

            print(f"   Training step time: {step_time:.2f} ms")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error benchmarking {model_name}: {e}")

    # Summary report
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY REPORT")
    print("=" * 80)

    summary_data = []

    for model_name in complexity_results.keys():
        if model_name in complexity_results:
            metrics = complexity_results[model_name]
            train_time = training_results.get(model_name, 0)

            summary_data.append(
                {
                    "Model": model_name,
                    "Parameters": f"{metrics['total_params']:,}",
                    "Size (MB)": f"{metrics['model_size_mb']:.1f}",
                    "Inference (ms)": f"{metrics['inference_time_ms']:.2f}",
                    "Training Step (ms)": (
                        f"{train_time:.2f}" if train_time > 0 else "N/A"
                    ),
                    "Throughput (sps)": f"{metrics['throughput']:.1f}",
                }
            )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Save detailed report
        report_path = "detailed_benchmark_report.xlsx"

        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Add raw metrics
            if complexity_results:
                raw_df = pd.DataFrame(complexity_results).T
                raw_df.to_excel(writer, sheet_name="Raw_Metrics")

        print(f"📁 Detailed report saved to: {report_path}")
        print("\n" + summary_df.to_string(index=False))

    return complexity_results, training_results


if __name__ == "__main__":
    print("🔬 MODEL PROFILING AND COMPARISON TOOL")
    print("=" * 80)

    try:
        # Run full benchmark
        complexity_results, training_results = full_benchmark()

        print("\n✅ Benchmark completed successfully!")
        print("📋 Check the generated Excel files for detailed results.")

    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted by user.")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n🏁 Done!")
