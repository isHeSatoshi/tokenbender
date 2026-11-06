from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Frozen-weight ten-percent Banking77 experiment")
    result.add_argument("--root", type=Path, default=Path.cwd(), help="Execution project root")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Record the actual Python/CUDA environment")
    download = commands.add_parser("download", help="Download and verify immutable model files")
    download.add_argument("--metadata-only", action="store_true")
    prepare = commands.add_parser("prepare-splits", help="One-shot data preparation and sealing")
    prepare.add_argument("--banking-only", action="store_true")
    baseline = commands.add_parser("baseline", help="Reproduce both base modes on development data")
    baseline.add_argument("--split", choices=["train", "working"], default="working")
    baseline.add_argument("--limit", type=int)
    baseline.add_argument("--device", default="cuda")
    baseline.add_argument(
        "--vocabulary-only",
        action="store_true",
        help="Development ablation, not the original-base comparator",
    )
    commands.add_parser("build-vocabulary", help="Freeze permitted-token vocabulary and fallback")
    identity = commands.add_parser("check-identity", help="Verify original-vocabulary all-one mask")
    identity.add_argument("--limit", type=int, default=8)
    identity.add_argument("--device", default="cuda")
    adaptation = commands.add_parser(
        "prepare-adaptation", help="Hash train-only EDA and measure maximum expanded length"
    )
    adaptation.add_argument("--variants", type=int, default=40000)
    cache = commands.add_parser("cache-teacher", help="Train-only teacher cache, never exam data")
    cache.add_argument("--variants", type=int, default=40000)
    cache.add_argument("--limit", type=int)
    cache.add_argument("--device", default="cuda")
    search = commands.add_parser("search-mask", help="Registered frozen-weight mask optimization")
    search.add_argument("--steps", type=int, default=2000)
    search.add_argument("--learning-rate", type=float, default=0.003)
    search.add_argument("--accumulation", type=int, default=32)
    search.add_argument("--source-target", type=int, default=152800000)
    search.add_argument("--group-size", type=int, choices=[2, 4], default=4)
    search.add_argument("--device", default="cuda")
    search.add_argument(
        "--diagnostic-oversize",
        action="store_true",
        help="Permit a 30%% diagnostic budget, never a qualifying bundle",
    )
    search.add_argument("--resume", action="store_true")
    pack = commands.add_parser("pack", help="Pack an eligible selected candidate as immutable A")
    pack.add_argument("--candidate", required=True)
    pack.add_argument("--device", default="cuda")
    verify = commands.add_parser("verify", help="Exhaustively compare every retained source word")
    verify.add_argument("--bundle", required=True, type=Path)
    feedback = commands.add_parser(
        "replay-feedback", help="Replay counted correction/new-intent memory"
    )
    feedback.add_argument("--state-a", type=Path, required=True)
    feedback.add_argument("--config", type=Path, required=True)
    feedback.add_argument("--device", default="cuda")
    commands.add_parser("preregister", help="Generate measured local preregistration inputs")
    commands.add_parser("freeze", help="Write the final manifest, before commit/push/attestation")
    final = commands.add_parser("final-evaluate", help="Single guarded frozen exam session")
    final.add_argument("--attestation", type=Path, required=True)
    final.add_argument("--resume", action="store_true")
    final.add_argument("--device", default="cuda")
    commands.add_parser("report", help="Read existing final results without reopening the exam")
    return result


def dispatch(args):
    root = args.root.resolve()
    match args.command:
        case "doctor":
            from .io import write_json
            from .sources import environment_report

            report = environment_report()
            write_json(root / "runs" / "environment.json", report)
            return report
        case "download":
            from .sources import download_model

            return download_model(root, weights=not args.metadata_only)
        case "prepare-splits":
            from .data import prepare

            report = prepare(root, include_clinc=not args.banking_only)
            return {"report": report["report"], "clinc": report["clinc_report"]}
        case "baseline":
            from .evaluation import baseline

            return baseline(
                root,
                split=args.split,
                limit=args.limit,
                device=args.device,
                vocabulary_only=args.vocabulary_only,
            )
        case "build-vocabulary":
            from .vocabulary import build

            return build(root)
        case "check-identity":
            from .identity import check_identity

            return check_identity(root, limit=args.limit, device=args.device)
        case "prepare-adaptation":
            from .teacher import prepare_adaptation

            return prepare_adaptation(root, variants=args.variants)
        case "cache-teacher":
            from .teacher import cache_teacher

            return cache_teacher(root, variants=args.variants, limit=args.limit, device=args.device)
        case "search-mask":
            from .training import run_search

            return run_search(
                root,
                steps=args.steps,
                learning_rate=args.learning_rate,
                accumulation=args.accumulation,
                source_target=args.source_target,
                group_size=args.group_size,
                device=args.device,
                diagnostic_oversize=args.diagnostic_oversize,
                resume=args.resume,
            )
        case "pack":
            from .workflow import pack_candidate

            return pack_candidate(root, args.candidate, device=args.device)
        case "verify":
            from .bundle import verify_lineage

            return verify_lineage(args.bundle.resolve(), root / "assets/base/model.safetensors")
        case "replay-feedback":
            from .workflow import replay_feedback

            return replay_feedback(
                root, args.state_a.resolve(), args.config.resolve(), device=args.device
            )
        case "preregister":
            from .workflow import create_preregistration

            return create_preregistration(root)
        case "freeze":
            from .workflow import make_freeze_manifest

            return make_freeze_manifest(root)
        case "final-evaluate":
            from .workflow import final_evaluate

            return final_evaluate(
                root, args.attestation.resolve(), resume=args.resume, device=args.device
            )
        case "report":
            from .io import read_json

            return read_json(root / "runs/final/results.json")
    raise ValueError("Unknown command")


def main():
    args = parser().parse_args()
    try:
        report = dispatch(args)
    except (ValueError, PermissionError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc), "command": args.command}), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(report, sort_keys=True, indent=2))
