import argparse
import json
from pathlib import Path
import subprocess


def save_report(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Starfall local visual Android agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    predict = sub.add_parser(
        "predict-image", help="Save annotated predictions from the trained detector"
    )
    predict.add_argument("image")
    predict.add_argument("--model", default="weights/starfall-detector-1988/best.pt")
    predict.add_argument("--device", default="auto")
    predict.add_argument("--confidence", type=float, default=0.35)
    predict.add_argument("--imgsz", type=int, default=416)
    predict.add_argument("--output", default="reports/trained-model-test.png")
    capture = sub.add_parser("capture-benchmark")
    capture.add_argument("--seconds", type=float, default=15)
    capture.add_argument("--size", type=int, default=960)
    capture.add_argument("--serial")
    live = sub.add_parser("live")
    live.add_argument("--config", default="configs/phone.yaml")
    live.add_argument("--serial")
    live.add_argument("--seconds", type=float, default=60)
    live.add_argument("--control", action="store_true")
    live.add_argument(
        "--guard",
        action="store_true",
        help="Watch health and exit at one heart without autonomous movement",
    )
    live.add_argument("--collect", action="store_true")
    live.add_argument("--no-preview", action="store_true")
    live.add_argument("--model")
    live.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    loop = sub.add_parser(
        "learning-loop",
        help="Play, record attempts, and restart after the user reopens Starfall; no heart scanning",
    )
    loop.add_argument("--config", default="configs/phone.yaml")
    loop.add_argument("--serial")
    loop.add_argument(
        "--seconds", type=float, default=0, help="0 continues until interrupted"
    )
    loop.add_argument(
        "--episodes", type=int, help="Stop after this many manually ended attempts"
    )
    loop.add_argument("--no-preview", action="store_true")
    response = sub.add_parser(
        "train-real-response",
        help="Fit and evaluate motion response from saved real attempts",
    )
    response.add_argument("--config", default="configs/phone.yaml")
    response.add_argument("--runs", default="runs")
    for name in ("calibrate-geometry", "calibrate-motion"):
        cal = sub.add_parser(name)
        cal.add_argument("--config", default="configs/phone.yaml")
        cal.add_argument("--serial")
    for name in ("annotate", "validate-data", "split-data"):
        ds = sub.add_parser(name)
        ds.add_argument("--root", default="data/game")
        ds.add_argument(
            "--classes",
            required=True,
            help="comma-separated manually confirmed class names",
        )
    sim = sub.add_parser("evaluate")
    sim.add_argument(
        "--policy",
        choices=["mpc", "nearest", "random", "stay", "learned", "shielded"],
        default="mpc",
    )
    sim.add_argument("--episodes", type=int, default=30)
    sim.add_argument("--seed", type=int, default=10000)
    sim.add_argument("--model")
    sim.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    sim.add_argument("--output", default="reports/evaluation.json")
    sim.add_argument("--duration", type=float, default=30)
    train = sub.add_parser("train-policy")
    train.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    train.add_argument("--steps", type=int, default=20000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--device", default="cpu", help="cpu, auto, cuda or CUDA GPU index"
    )
    train.add_argument("--n-envs", type=int, default=1)
    train.add_argument("--vec-backend", choices=["dummy", "subproc"], default="dummy")
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--rollout-steps", type=int, default=512)
    gate = sub.add_parser("compare")
    gate.add_argument("candidate")
    gate.add_argument("baseline")
    bench = sub.add_parser("benchmark-detectors")
    bench.add_argument("images", nargs="+")
    bench.add_argument("--openvino", action="store_true")
    bench.add_argument("--repeats", type=int, default=10)
    det = sub.add_parser("train-detector")
    det.add_argument("dataset")
    det.add_argument("--model", default="yolo26n.pt")
    det.add_argument("--epochs", type=int, default=40)
    det.add_argument(
        "--device",
        default="auto",
        help="auto prefers NVIDIA CUDA; cpu or 0 also supported",
    )
    det.add_argument("--imgsz", type=int, default=416)
    det.add_argument("--batch", type=int, default=4)
    det.add_argument("--workers", type=int, default=0)
    det.add_argument("--threads", type=int, default=2)
    val = sub.add_parser("evaluate-detector")
    val.add_argument("model")
    val.add_argument("dataset")
    val.add_argument("--split", choices=["val", "test"], default="test")
    val.add_argument("--device", default="auto")
    val.add_argument("--imgsz", type=int, default=416)
    summary = sub.add_parser("summarize-runs")
    summary.add_argument("--root", default="runs")
    args = parser.parse_args()
    if args.command == "doctor":
        import importlib.metadata as metadata
        import platform
        import psutil
        from .android import Adb, BUNDLE

        report = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ram_gb": psutil.virtual_memory().total / 2**30,
            "logical_cpus": psutil.cpu_count(),
            "scrcpy_bundle": str(BUNDLE),
        }
        try:
            import torch

            report["cuda_available"] = torch.cuda.is_available()
            report["torch_version"] = torch.__version__
            report["torch_cuda_version"] = torch.version.cuda
            report["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "dedicated_gb": torch.cuda.get_device_properties(i).total_memory
                    / 2**30,
                }
                for i in range(torch.cuda.device_count())
            ]
        except ImportError:
            report["cuda_available"] = None
        try:
            from openvino import Core

            report["openvino_devices"] = Core().available_devices
        except ImportError:
            report["openvino_devices"] = None
        report["packages"] = {
            d.metadata["Name"]: d.version for d in metadata.distributions()
        }
        try:
            report["android_devices"] = Adb().devices()
        except (OSError, subprocess.SubprocessError) as exc:
            report["adb_error"] = str(exc)
        save_report("reports/environment.json", report)
    elif args.command == "predict-image":
        from .prediction import predict_image

        print(
            json.dumps(
                predict_image(
                    args.image,
                    args.model,
                    args.device,
                    args.output,
                    args.confidence,
                    args.imgsz,
                ),
                indent=2,
            )
        )
    elif args.command == "capture-benchmark":
        from .runtime import capture_benchmark

        result = capture_benchmark(args.seconds, args.serial, args.size)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    elif args.command == "live":
        from .runtime import run_live

        path, report = run_live(
            args.config,
            args.serial,
            args.seconds,
            args.control,
            args.collect,
            not args.no_preview,
            args.model,
            args.algorithm,
            args.guard,
        )
        print(path)
        print(json.dumps(report, indent=2))
    elif args.command == "learning-loop":
        from .runtime import run_live

        path, report = run_live(
            args.config,
            serial=args.serial,
            seconds=args.seconds,
            control=True,
            collect=True,
            preview=not args.no_preview,
            repeat=True,
            max_episodes=args.episodes,
        )
        print(path)
        print(json.dumps(report, indent=2))
    elif args.command == "train-real-response":
        from .runtime import load_config
        from .real_learning import train_real_response

        print(
            json.dumps(
                train_real_response(load_config(args.config), args.runs), indent=2
            )
        )
    elif args.command.startswith("calibrate"):
        from .calibration import select_geometry, measure_motion

        result = (
            select_geometry if args.command == "calibrate-geometry" else measure_motion
        )(args.config, args.serial)
        print(json.dumps(result, indent=2))
    elif args.command in ("annotate", "validate-data", "split-data"):
        from .dataset import annotate, validate, split

        classes = args.classes.split(",")
        if args.command == "annotate":
            annotate(args.root, classes)
        elif args.command == "validate-data":
            print(json.dumps(validate(args.root, classes), indent=2))
        else:
            print(split(args.root, classes))
    elif args.command == "evaluate":
        from .experiments import evaluate

        if args.policy in ("learned", "shielded") and not args.model:
            parser.error("Learned policy requires --model")
        report = evaluate(
            args.policy,
            args.episodes,
            args.seed,
            args.model,
            args.algorithm,
            args.duration,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
    elif args.command == "train-policy":
        from .experiments import train_policy

        print(
            train_policy(
                args.algorithm,
                args.steps,
                args.seed,
                device=args.device,
                n_envs=args.n_envs,
                vec_backend=args.vec_backend,
                threads=args.threads,
                rollout_steps=args.rollout_steps,
            )
        )
    elif args.command == "compare":
        from .experiments import compare

        print(
            json.dumps(
                compare(
                    json.loads(Path(args.candidate).read_text()),
                    json.loads(Path(args.baseline).read_text()),
                ),
                indent=2,
            )
        )
    elif args.command == "benchmark-detectors":
        from .experiments import benchmark_detectors

        print(
            json.dumps(
                benchmark_detectors(
                    args.images, repeats=args.repeats, export_openvino=args.openvino
                ),
                indent=2,
            )
        )
    elif args.command == "train-detector":
        from .experiments import train_detector

        train_detector(
            args.dataset,
            args.model,
            args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            threads=args.threads,
        )
    elif args.command == "evaluate-detector":
        from ultralytics import YOLO
        from .training import select_device, prepare_dataset

        data = prepare_dataset(args.dataset, "reports/detector-dataset.yaml")
        metrics = YOLO(args.model).val(
            data=str(data),
            split=args.split,
            imgsz=args.imgsz,
            device=select_device(args.device),
            plots=True,
        )
        save_report(
            "reports/detector-evaluation.json",
            {
                "model": args.model,
                "dataset": args.dataset,
                "split": args.split,
                "overall": metrics.results_dict,
                "per_class_map": metrics.box.maps.tolist(),
                "class_names": metrics.names,
                "speed": metrics.speed,
                "precision": metrics.box.p.tolist(),
                "recall": metrics.box.r.tolist(),
            },
        )
    elif args.command == "summarize-runs":
        import numpy as np

        reports = [
            json.loads(p.read_text()) for p in Path(args.root).glob("*/summary.json")
        ]
        scores = [r["score"] for r in reports if r.get("score") is not None]
        save_report(
            "reports/run-summary.json",
            {
                "runs": len(reports),
                "scored_runs": len(scores),
                "highest_score": max(scores) if scores else None,
                "mean_score": float(np.mean(scores)) if scores else None,
                "median_score": float(np.median(scores)) if scores else None,
                "summaries": reports,
            },
        )


if __name__ == "__main__":
    main()
