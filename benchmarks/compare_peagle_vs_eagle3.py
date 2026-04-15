#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_SCRIPT = REPO_ROOT / "benchmarks" / "bench_eagle3.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the same SpecForge benchmark suite for EAGLE3 and P-EAGLE, then summarize the speedup."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eagle3-draft-model-path", type=str, required=True)
    parser.add_argument("--peagle-draft-model-path", type=str, required=True)
    parser.add_argument(
        "--eagle3-config-list",
        nargs="+",
        default=["1,3,1,4"],
        help="Config list for the EAGLE3 baseline.",
    )
    parser.add_argument(
        "--peagle-config-list",
        nargs="+",
        default=["1,7,1,8"],
        help="Config list for the P-EAGLE run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./results"),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="gptoss20b_peagle_vs_eagle3",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
    )
    return parser.parse_known_args()


def _latest_result_file(output_dir: Path, name_prefix: str) -> Path:
    matches = sorted(output_dir.glob(f"{name_prefix}_results_*.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find benchmark result files for prefix {name_prefix!r} in {output_dir}"
        )
    return matches[-1]


def _run_case(
    *,
    label: str,
    model_path: str,
    draft_model_path: str,
    config_list: list[str],
    output_dir: Path,
    port: int,
    passthrough_args: list[str],
) -> dict:
    run_name = label
    command = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--model-path",
        model_path,
        "--speculative-draft-model-path",
        draft_model_path,
        "--config-list",
        *config_list,
        "--output-dir",
        str(output_dir),
        "--name",
        run_name,
        "--port",
        str(port),
        "--disable-overlap-schedule",
        *passthrough_args,
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    result_path = _latest_result_file(output_dir, run_name)
    return json.loads(result_path.read_text())


def _best_by_benchmark(results: dict) -> dict[str, dict]:
    summary = {}
    for benchmark_name, rows in results.items():
        if benchmark_name in {"model", "bita_model"}:
            continue
        best_row = None
        best_tps = -1.0
        for row in rows:
            metrics = row.get("metrics", [])
            if not metrics:
                continue
            avg_tps = sum(metric["output_throughput"] for metric in metrics) / len(metrics)
            avg_accept = sum(metric["accept_length"] for metric in metrics) / len(metrics)
            candidate = {
                "output_throughput": avg_tps,
                "accept_length": avg_accept,
                "steps": row["steps"],
                "topk": row["topk"],
                "num_draft_tokens": row["num_draft_tokens"],
            }
            if avg_tps > best_tps:
                best_tps = avg_tps
                best_row = candidate
        if best_row is not None:
            summary[benchmark_name] = best_row
    return summary


def main():
    args, passthrough_args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eagle3_results = _run_case(
        label=f"{args.name}_eagle3",
        model_path=args.model_path,
        draft_model_path=args.eagle3_draft_model_path,
        config_list=args.eagle3_config_list,
        output_dir=output_dir,
        port=args.port,
        passthrough_args=passthrough_args,
    )
    peagle_results = _run_case(
        label=f"{args.name}_peagle",
        model_path=args.model_path,
        draft_model_path=args.peagle_draft_model_path,
        config_list=args.peagle_config_list,
        output_dir=output_dir,
        port=args.port,
        passthrough_args=passthrough_args,
    )

    eagle3_best = _best_by_benchmark(eagle3_results)
    peagle_best = _best_by_benchmark(peagle_results)

    comparison = {
        "model_path": args.model_path,
        "eagle3_draft_model_path": args.eagle3_draft_model_path,
        "peagle_draft_model_path": args.peagle_draft_model_path,
        "benchmarks": {},
    }
    for benchmark_name in sorted(set(eagle3_best) & set(peagle_best)):
        base = eagle3_best[benchmark_name]
        peagle = peagle_best[benchmark_name]
        speedup = peagle["output_throughput"] / base["output_throughput"]
        comparison["benchmarks"][benchmark_name] = {
            "eagle3": base,
            "peagle": peagle,
            "speedup": speedup,
        }

    result_path = output_dir / f"{args.name}_comparison.json"
    result_path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    print(f"Comparison saved to {result_path}")


if __name__ == "__main__":
    main()
