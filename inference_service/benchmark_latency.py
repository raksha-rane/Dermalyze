#!/usr/bin/env python3
"""Benchmark single-image inference latency on CPU vs GPU backends.

This measures end-to-end `SkinLesionPredictor.predict` latency for one image,
including preprocessing and model forward pass.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_service.predictor import SkinLesionPredictor


def _resolve_default_checkpoint(service_dir: Path) -> Path:
    primary = service_dir / "models" / "checkpoint_best.pt"
    legacy = service_dir / "model" / "checkpoint_best.pt"
    if primary.exists():
        return primary
    return legacy


def _sync_device(device_name: str) -> None:
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()


def _available_gpu_devices() -> List[str]:
    devices: List[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _resolve_devices(compare_mode: str) -> List[str]:
    gpu_devices = _available_gpu_devices()

    if compare_mode == "cpu":
        return ["cpu"]

    if compare_mode == "gpu":
        if not gpu_devices:
            raise RuntimeError("No GPU backend is available (CUDA/MPS).")
        return [gpu_devices[0]]

    if compare_mode == "both":
        devices = ["cpu"]
        if gpu_devices:
            devices.append(gpu_devices[0])
        return devices

    # compare_mode == "all"
    return ["cpu", *gpu_devices]


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0

    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]

    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _parse_metadata(metadata_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if metadata_json is None:
        return None

    parsed = json.loads(metadata_json)
    if not isinstance(parsed, dict):
        raise ValueError("--metadata-json must decode to a JSON object.")
    return parsed


def _benchmark_device(
    checkpoint_path: Path,
    image_bytes: bytes,
    device_name: str,
    image_size: int,
    runs: int,
    warmup: int,
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    predictor = SkinLesionPredictor(
        checkpoint_path=checkpoint_path,
        device=device_name,
        image_size=image_size,
    )

    # Warmup avoids one-time effects (kernel compilation, caching, etc.).
    for _ in range(warmup):
        predictor.predict(
            image=image_bytes,
            metadata=metadata,
            include_disclaimer=False,
            include_gradcam=False,
        )
    _sync_device(device_name)

    latencies_ms: List[float] = []
    for _ in range(runs):
        _sync_device(device_name)
        start = time.perf_counter()
        predictor.predict(
            image=image_bytes,
            metadata=metadata,
            include_disclaimer=False,
            include_gradcam=False,
        )
        _sync_device(device_name)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    latencies_ms.sort()
    mean_ms = statistics.fmean(latencies_ms)
    median_ms = statistics.median(latencies_ms)

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": _percentile(latencies_ms, 0.95),
        "p99_ms": _percentile(latencies_ms, 0.99),
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
        "throughput_img_per_s": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def _save_comparison_plot(
    results: Dict[str, Dict[str, float]], output_path: Path
) -> None:
    if len(results) < 2:
        raise ValueError("Need at least two devices to generate a comparison graph.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for graph output. Install it in your environment."
        ) from exc

    latency_metrics = [
        ("mean_ms", "Mean"),
        ("median_ms", "Median"),
        ("p95_ms", "P95"),
        ("p99_ms", "P99"),
    ]
    throughput_metric = ("throughput_img_per_s", "Throughput")

    devices = list(results.keys())
    num_devices = len(devices)
    gpu_devices = [device_name for device_name in devices if device_name != "cpu"]
    display_label_by_device: Dict[str, str] = {"cpu": "CPU"}
    if len(gpu_devices) == 1:
        display_label_by_device[gpu_devices[0]] = "GPU"
    else:
        for index, device_name in enumerate(gpu_devices, start=1):
            display_label_by_device[device_name] = f"GPU {index}"

    fig, (ax_latency, ax_throughput) = plt.subplots(1, 2, figsize=(13, 5))

    x_positions = list(range(len(latency_metrics)))
    width = 0.8 / num_devices
    max_latency_value = max(
        results[device_name][metric_key]
        for device_name in devices
        for metric_key, _ in latency_metrics
    )
    for i, device_name in enumerate(devices):
        bar_positions = [x - 0.4 + (i + 0.5) * width for x in x_positions]
        values = [results[device_name][metric_key] for metric_key, _ in latency_metrics]
        bars = ax_latency.bar(
            bar_positions,
            values,
            width=width,
            label=display_label_by_device.get(device_name, device_name.upper()),
        )
        for bar, value in zip(bars, values):
            ax_latency.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_latency_value * 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax_latency.set_xticks(x_positions)
    ax_latency.set_xticklabels([label for _, label in latency_metrics])
    ax_latency.set_ylabel("Latency (ms)")
    ax_latency.set_title("Latency metrics")
    ax_latency.set_ylim(0, max_latency_value * 1.25)
    ax_latency.grid(axis="y", linestyle="--", alpha=0.3)
    ax_latency.legend()

    throughput_values = [
        results[device_name][throughput_metric[0]] for device_name in devices
    ]
    throughput_positions = list(range(num_devices))
    max_throughput_value = max(throughput_values) if throughput_values else 0.0
    throughput_bars = ax_throughput.bar(
        throughput_positions, throughput_values, width=0.6
    )
    for bar, value in zip(throughput_bars, throughput_values):
        ax_throughput.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max_throughput_value * 0.01,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_throughput.set_xticks(throughput_positions)
    ax_throughput.set_xticklabels(
        [
            display_label_by_device.get(device_name, device_name.upper())
            for device_name in devices
        ]
    )
    ax_throughput.set_ylabel("Images / second")
    ax_throughput.set_title(throughput_metric[1])
    ax_throughput.set_ylim(
        0, max_throughput_value * 1.35 if max_throughput_value > 0 else 1.0
    )
    ax_throughput.grid(axis="y", linestyle="--", alpha=0.3)

    if "cpu" in results and num_devices > 1:
        cpu_mean = results["cpu"]["mean_ms"]
        speedup_lines = []
        for device_name in devices:
            if device_name == "cpu":
                continue
            device_mean = results[device_name]["mean_ms"]
            speedup = cpu_mean / device_mean if device_mean > 0 else 0.0
            speedup_lines.append(
                f"{display_label_by_device.get(device_name, device_name.upper())}: {speedup:.2f}x"
            )

        if speedup_lines:
            ax_throughput.text(
                0.02,
                0.98,
                "Speedup vs CPU (mean latency)\n" + "\n".join(speedup_lines),
                transform=ax_throughput.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
            )

    fig.suptitle("Single-image inference benchmark (CPU vs GPU)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_results_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    service_dir = Path(__file__).resolve().parent
    default_output_dir = service_dir / "benchmark_results"

    parser = argparse.ArgumentParser(
        description="Benchmark single-image inference latency on CPU vs GPU"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_resolve_default_checkpoint(service_dir),
        help=(
            "Path to .pt checkpoint "
            "(default: inference_service/models/checkpoint_best.pt, "
            "fallback: inference_service/model/checkpoint_best.pt)"
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=300,
        help="Model input image size (must match checkpoint training setup)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of timed inference runs per device",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of warmup runs per device",
    )
    parser.add_argument(
        "--compare",
        choices=["both", "cpu", "gpu", "all"],
        default="both",
        help=(
            "Device set to benchmark: "
            "both=CPU + one preferred GPU, gpu=preferred GPU only, "
            "all=CPU + all available GPU backends"
        ),
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object for metadata-fusion checkpoints, e.g. "
            '\'{"age_approx": 52, "sex": "male", "anatom_site": "back"}\''
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to store benchmark artifacts",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=default_output_dir / "benchmark_summary.json",
        help="Path to save benchmark summary JSON",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=default_output_dir / "benchmark_cpu_gpu_comparison.png",
        help=(
            "Path to save the comparative graph "
            "(mean/median/p95/p99 latency + throughput)"
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib graph generation",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    image_path = args.image.expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    metadata = _parse_metadata(args.metadata_json)
    devices = _resolve_devices(args.compare)
    image_bytes = image_path.read_bytes()

    print("Single-image inference latency benchmark")
    print(f"checkpoint: {checkpoint_path}")
    print(f"image:      {image_path}")
    print(f"image_size: {args.image_size}")
    print(f"runs:       {args.runs} (warmup: {args.warmup})")
    print(f"devices:    {', '.join(devices)}")
    if metadata is not None:
        print("metadata:   provided")

    results: Dict[str, Dict[str, float]] = {}
    for device_name in devices:
        print(f"\nBenchmarking on {device_name}...")
        results[device_name] = _benchmark_device(
            checkpoint_path=checkpoint_path,
            image_bytes=image_bytes,
            device_name=device_name,
            image_size=args.image_size,
            runs=args.runs,
            warmup=args.warmup,
            metadata=metadata,
        )

        r = results[device_name]
        print(
            f"  mean={r['mean_ms']:.2f} ms | median={r['median_ms']:.2f} ms | "
            f"p95={r['p95_ms']:.2f} ms | p99={r['p99_ms']:.2f} ms | "
            f"throughput={r['throughput_img_per_s']:.2f} img/s"
        )

    if "cpu" in results and len(results) > 1:
        cpu_mean = results["cpu"]["mean_ms"]
        print("\nSpeedup vs CPU (mean latency):")
        for device_name, metrics in results.items():
            if device_name == "cpu":
                continue
            speedup = cpu_mean / metrics["mean_ms"] if metrics["mean_ms"] > 0 else 0.0
            print(f"  {device_name}: {speedup:.2f}x")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_output = args.results_output.expanduser()
    if not results_output.is_absolute():
        results_output = output_dir / results_output
    results_output = results_output.resolve()
    summary_payload: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "image": str(image_path),
        "image_size": args.image_size,
        "runs": args.runs,
        "warmup": args.warmup,
        "devices": devices,
        "results": results,
    }
    _save_results_json(summary_payload, results_output)
    print(f"\nSaved benchmark summary: {results_output}")

    if not args.no_plot:
        if len(results) < 2:
            print(
                "\nSkipping comparison graph (requires at least two benchmarked devices)."
            )
        else:
            plot_output = args.plot_output.expanduser()
            if not plot_output.is_absolute():
                plot_output = output_dir / plot_output
            plot_output = plot_output.resolve()
            _save_comparison_plot(results, plot_output)
            print(f"\nSaved comparison graph: {plot_output}")


if __name__ == "__main__":
    main()
