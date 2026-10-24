import argparse
import base64
import sys
import time

import cv2
import httpx
import numpy as np


def load_frame(path: str, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read a frame from {path}")
    height, original_width = frame.shape[:2]
    if original_width > width:
        frame = cv2.resize(
            frame,
            (width, max(1, round(height * width / original_width))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def percentile(sorted_samples: list[float], q: float) -> float:
    index = min(len(sorted_samples) - 1, max(0, round(q * (len(sorted_samples) - 1))))
    return sorted_samples[index]


def measure(send, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        send()
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        send()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {
        "p50": percentile(samples, 0.5),
        "p95": percentile(samples, 0.95),
        "mean": sum(samples) / len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--video", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    frame = load_frame(args.video, args.width)
    height, width = frame.shape[:2]
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    if not ok:
        sys.exit("jpeg encoding failed")
    jpeg = buffer.tobytes()
    b64 = base64.b64encode(jpeg).decode()
    raw = frame.tobytes()
    raw_headers = {
        "X-Frame-Height": str(height),
        "X-Frame-Width": str(width),
        "Content-Type": "application/octet-stream",
    }

    with httpx.Client(base_url=args.url, timeout=30.0) as client:
        def post(path: str, **kwargs):
            def send():
                client.post(path, **kwargs).raise_for_status()
            return send

        cases = [
            ("raw ndarray", len(raw), "/echo/raw", "/detect/raw", {"content": raw, "headers": raw_headers}),
            ("jpeg binary", len(jpeg), "/echo/binary", "/detect/binary", {"files": {"file": ("f.jpg", jpeg, "image/jpeg")}}),
            ("jpeg base64", len(b64), "/echo/base64", "/detect/base64", {"json": {"image": b64}}),
        ]

        print(f"frame {width}x{height}, runs={args.runs}, warmup={args.warmup}")
        print()
        header = f"{'mode':<14}{'payload KB':>12}{'echo p50':>11}{'echo p95':>11}{'detect p50':>12}{'detect p95':>12}{'detect mean':>13}"
        print(header)
        print("-" * len(header))

        results = {}
        for label, size, echo_path, detect_path, kwargs in cases:
            echo = measure(post(echo_path, **kwargs), args.runs, args.warmup)
            detect = measure(post(detect_path, **kwargs), args.runs, args.warmup)
            results[label] = (echo, detect)
            print(
                f"{label:<14}{size / 1024:>12.1f}"
                f"{echo['p50']:>11.1f}{echo['p95']:>11.1f}"
                f"{detect['p50']:>12.1f}{detect['p95']:>12.1f}{detect['mean']:>13.1f}"
            )

        print()
        base = results["jpeg binary"][0]["p50"]
        print(f"transport+decode p50, base64 vs binary: {results['jpeg base64'][0]['p50'] - base:+.1f} ms")
        print(f"transport+decode p50, raw vs binary:    {results['raw ndarray'][0]['p50'] - base:+.1f} ms")
        print(f"payload, base64 vs binary:              {len(b64) / len(jpeg) - 1:+.1%}")
        print(f"payload, raw vs binary:                 {len(raw) / len(jpeg):.1f}x")


if __name__ == "__main__":
    main()
