import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze binaural spatial cues in generated wav files.")
    parser.add_argument("wav", nargs="+", help="One or more stereo wav files.")
    parser.add_argument("--csv", default=None, help="Optional path to save metrics as CSV.")
    return parser.parse_args()


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + EPS))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = np.sqrt(np.sum(left * left) * np.sum(right * right)) + EPS
    return float(np.sum(left * right) / denom)


def analyze(path: Path):
    audio, sr = sf.read(str(path), always_2d=True)
    if audio.shape[1] < 2:
        raise ValueError(f"{path} is mono; expected a stereo/binaural file.")
    left = audio[:, 0]
    right = audio[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    left_rms = rms(left)
    right_rms = rms(right)
    mid_rms = rms(mid)
    side_rms = rms(side)
    ild_db = 20.0 * np.log10((left_rms + EPS) / (right_rms + EPS))
    side_mid_db = 20.0 * np.log10((side_rms + EPS) / (mid_rms + EPS))
    diff_rms = rms(left - right)
    return {
        "file": path.name,
        "path": str(path),
        "sr": sr,
        "duration_sec": audio.shape[0] / sr,
        "samples": audio.shape[0],
        "left_rms": left_rms,
        "right_rms": right_rms,
        "ild_db": float(ild_db),
        "diff_rms": diff_rms,
        "lr_corr": safe_corr(left, right),
        "mid_rms": mid_rms,
        "side_rms": side_rms,
        "side_mid_db": float(side_mid_db),
        "peak_abs": float(np.max(np.abs(audio))),
    }


def format_table(rows):
    columns = [
        ("file", "file", 28),
        ("duration_sec", "dur(s)", 8),
        ("ild_db", "ILD(dB)", 9),
        ("diff_rms", "diff_rms", 10),
        ("lr_corr", "lr_corr", 9),
        ("side_mid_db", "side/mid", 9),
        ("peak_abs", "peak", 8),
    ]
    header = " ".join(label.ljust(width) for _, label, width in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        values = []
        for key, _, width in columns:
            value = row[key]
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            values.append(text.ljust(width))
        lines.append(" ".join(values))
    return "\n".join(lines)


def print_pairwise(rows):
    if len(rows) < 2:
        return
    print("\nPairwise cue differences:")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a = rows[i]
            b = rows[j]
            print(
                f"{a['file']} vs {b['file']}: "
                f"delta_ILD={abs(a['ild_db'] - b['ild_db']):.4f} dB, "
                f"delta_diff_rms={abs(a['diff_rms'] - b['diff_rms']):.6f}, "
                f"delta_corr={abs(a['lr_corr'] - b['lr_corr']):.4f}"
            )


def save_csv(rows, path: Path):
    keys = [
        "file",
        "path",
        "sr",
        "duration_sec",
        "samples",
        "left_rms",
        "right_rms",
        "ild_db",
        "diff_rms",
        "lr_corr",
        "mid_rms",
        "side_rms",
        "side_mid_db",
        "peak_abs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in keys) + "\n")


def main():
    args = parse_args()
    rows = [analyze(Path(wav)) for wav in args.wav]
    print(format_table(rows))
    print_pairwise(rows)
    if args.csv:
        save_csv(rows, Path(args.csv))
        print(f"\nsaved csv: {args.csv}")


if __name__ == "__main__":
    main()
