import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from analyze_audio import analyze
from dataset import SAMPLE_RATE, discover_ids, load_prompts, make_splits
from infer import crop_mono_clip, infer_clip, infer_full, load_model_and_tokenizer, load_mono


PROMPTS = {
    "left": "The sound source is on the left.",
    "center": "The sound source is in the center.",
    "right": "The sound source is on the right.",
}


def direction_prompts_for_object(object_phrase: str, verb: str):
    return {
        "left": f"The {object_phrase} {verb} on the left.",
        "center": f"The {object_phrase} {verb} in the center.",
        "right": f"The {object_phrase} {verb} on the right.",
    }


def extract_first_object(prompt: str):
    match = re.search(r"^The\s+(.+?)\s+(is|are)\s+(?:on|in)\s+the\s+(?:left|right|center)", prompt)
    if not match:
        return "sound source", "is"
    object_phrase = match.group(1).strip()
    verb = match.group(2).strip()
    if " and the " in object_phrase:
        object_phrase = object_phrase.split(" and the ", 1)[0].strip()
    return object_phrase, verb


def is_single_source_prompt(prompt: str):
    matches = re.findall(r"\b(?:is|are)\s+(?:on|in)\s+the\s+(?:left|right|center)\b", prompt)
    object_phrase, _ = extract_first_object(prompt)
    has_multiple_objects = bool(re.search(r"\band\b", object_phrase, flags=re.IGNORECASE))
    return len(matches) == 1 and not has_multiple_objects


def prompts_for_audio(data_root: Path, audio_id: str, prompt_template: str):
    if prompt_template == "generic":
        return PROMPTS, "sound source", "The sound source is on the left."
    prompts = load_prompts(data_root / "text_prompts" / f"{audio_id}.csv")
    base_prompt = prompts[0]
    object_phrase, verb = extract_first_object(base_prompt)
    return direction_prompts_for_object(object_phrase, verb), object_phrase, base_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-evaluate prompt-controlled binaural spatial cues.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data/TASBench")
    parser.add_argument("--out_dir", default="code/TAS_repro/runs/batch_prompt_eval")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num_items", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mode", choices=["clip", "full"], default="clip")
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--hop_sec", type=float, default=0.25)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--prompt_template", choices=["object", "generic"], default="object")
    parser.add_argument("--single_source_only", action="store_true")
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow_model_download", action="store_true")
    parser.add_argument("--save_audio", action="store_true")
    return parser.parse_args()


def write_wav(path: Path, binaural: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = binaural.squeeze(0).detach().cpu().transpose(0, 1).numpy()
    peak = abs(wav).max()
    clipped = False
    if peak > 1.0:
        wav = wav.clip(-1.0, 1.0)
        clipped = True
    sf.write(str(path), wav, SAMPLE_RATE)
    return clipped, float(peak)


def analyze_tensor(binaural: torch.Tensor, out_path: Path):
    clipped, peak = write_wav(out_path, binaural)
    row = analyze(out_path)
    row["was_clipped_on_write"] = clipped
    row["preclip_peak_abs"] = peak
    return row


def direction_order_ok(rows_by_direction):
    left_ild = rows_by_direction["left"]["ild_db"]
    center_ild = rows_by_direction["center"]["ild_db"]
    right_ild = rows_by_direction["right"]["ild_db"]
    return left_ild > center_ild > right_ild


def center_between_ok(rows_by_direction):
    left_ild = rows_by_direction["left"]["ild_db"]
    center_ild = rows_by_direction["center"]["ild_db"]
    right_ild = rows_by_direction["right"]["ild_db"]
    low, high = sorted([left_ild, right_ild])
    return low <= center_ild <= high


def left_right_ok(rows_by_direction):
    return rows_by_direction["left"]["ild_db"] > rows_by_direction["right"]["ild_db"]


def main():
    args = parse_args()
    if args.hop_sec <= 0:
        raise ValueError("--hop_sec must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = discover_ids(Path(args.data_root))
    split_ids = make_splits(ids, seed=args.seed)[args.split]
    selected_ids = list(split_ids)
    skipped_multi_source = 0
    if args.single_source_only:
        filtered_ids = []
        for audio_id in selected_ids:
            prompts = load_prompts(Path(args.data_root) / "text_prompts" / f"{audio_id}.csv")
            if is_single_source_prompt(prompts[0]):
                filtered_ids.append(audio_id)
            else:
                skipped_multi_source += 1
        selected_ids = filtered_ids
        if not selected_ids:
            raise RuntimeError(f"No single-source items found in split {args.split}.")
    rng.shuffle(selected_ids)
    selected_ids = selected_ids[: args.num_items]

    diffusion, tokenizer = load_model_and_tokenizer(args)
    diffusion = diffusion.to(device)
    diffusion.eval()

    rows = []
    correct_order = 0
    center_between = 0
    left_right_correct = 0
    left_right_deltas = []
    center_abs_ild = []
    for item_index, audio_id in enumerate(selected_ids, start=1):
        wav_path = Path(args.data_root) / "binaural_audios" / f"{audio_id}.wav"
        mono = load_mono(wav_path)
        rows_by_direction = {}
        prompts, object_phrase, base_prompt = prompts_for_audio(Path(args.data_root), audio_id, args.prompt_template)
        print(f"[{item_index}/{len(selected_ids)}] audio_id={audio_id} object={object_phrase} base_prompt=\"{base_prompt}\"")

        for direction, prompt in prompts.items():
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            tokens = tokenizer.batch_encode([prompt])
            tokens = {key: value.to(device) for key, value in tokens.items()}
            if args.mode == "clip":
                mono_clip = crop_mono_clip(mono, args.start_sec).to(device)
                binaural = infer_clip(
                    diffusion,
                    mono_clip,
                    tokens,
                    args.sample_steps,
                    args.clip_denoised,
                    args.sampler,
                )
            else:
                binaural = infer_full(
                    diffusion,
                    mono,
                    tokens,
                    args.sample_steps,
                    args.hop_sec,
                    device,
                    args.clip_denoised,
                    args.sampler,
                )

            wav_out = out_dir / "audio" / f"{audio_id}_{direction}.wav"
            row = analyze_tensor(binaural, wav_out)
            if not args.save_audio:
                wav_out.unlink(missing_ok=True)
            row.update(
                {
                    "audio_id": audio_id,
                    "direction": direction,
                    "prompt": prompt,
                    "mode": args.mode,
                    "sampler": args.sampler,
                    "sample_steps": args.sample_steps,
                    "prompt_template": args.prompt_template,
                    "object_phrase": object_phrase,
                    "base_prompt": base_prompt,
                    "is_single_source": is_single_source_prompt(base_prompt),
                }
            )
            rows.append(row)
            rows_by_direction[direction] = row
            print(
                f"  {direction:6s} ILD={row['ild_db']:.4f} "
                f"diff_rms={row['diff_rms']:.5f} corr={row['lr_corr']:.4f}"
            )

        is_correct = direction_order_ok(rows_by_direction)
        is_center_between = center_between_ok(rows_by_direction)
        is_left_right = left_right_ok(rows_by_direction)
        left_right_delta = rows_by_direction["left"]["ild_db"] - rows_by_direction["right"]["ild_db"]
        correct_order += int(is_correct)
        center_between += int(is_center_between)
        left_right_correct += int(is_left_right)
        left_right_deltas.append(left_right_delta)
        center_abs_ild.append(abs(rows_by_direction["center"]["ild_db"]))
        print(
            f"  order_left_gt_center_gt_right={is_correct} "
            f"left_gt_right={is_left_right} "
            f"delta_left_right={left_right_delta:.4f} "
            f"center_between_left_right={is_center_between}"
        )

    metrics_path = out_dir / "prompt_eval_metrics.csv"
    keys = [
        "audio_id",
        "direction",
        "prompt",
        "mode",
        "sampler",
        "sample_steps",
        "prompt_template",
        "object_phrase",
        "base_prompt",
        "is_single_source",
        "duration_sec",
        "ild_db",
        "diff_rms",
        "lr_corr",
        "side_mid_db",
        "peak_abs",
        "preclip_peak_abs",
        "was_clipped_on_write",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})

    summary = {
        "split": args.split,
        "num_items": len(selected_ids),
        "mode": args.mode,
        "sampler": args.sampler,
        "sample_steps": args.sample_steps,
        "prompt_template": args.prompt_template,
        "single_source_only": args.single_source_only,
        "skipped_multi_source": skipped_multi_source,
        "order_accuracy_left_gt_center_gt_right": correct_order / max(len(selected_ids), 1),
        "left_vs_right_accuracy": left_right_correct / max(len(selected_ids), 1),
        "center_between_accuracy": center_between / max(len(selected_ids), 1),
        "mean_delta_ild_left_minus_right": float(np.mean(left_right_deltas)) if left_right_deltas else 0.0,
        "mean_abs_center_ild": float(np.mean(center_abs_ild)) if center_abs_ild else 0.0,
        "mean_abs_ild": float(np.mean([abs(row["ild_db"]) for row in rows])) if rows else 0.0,
        "mean_diff_rms": float(np.mean([row["diff_rms"] for row in rows])) if rows else 0.0,
        "metrics_csv": str(metrics_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
