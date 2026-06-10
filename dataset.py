import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset


SAMPLE_RATE = 16000
SAMPLE_SECONDS = 1.0
SAMPLE_LENGTH = int(SAMPLE_RATE * SAMPLE_SECONDS)
PROMPT_FPS = 10.0
DIRECTION_CATEGORIES = ("left", "center", "right", "mixed")
DIRECTION_TO_ID = {
    "left": 0,
    "center": 1,
    "right": 2,
}


def discover_ids(data_root: Path) -> List[str]:
    audio_dir = data_root / "binaural_audios"
    prompt_dir = data_root / "text_prompts"
    wav_ids = {p.stem for p in audio_dir.glob("*.wav")}
    csv_ids = {p.stem for p in prompt_dir.glob("*.csv")}
    ids = sorted(wav_ids & csv_ids)
    if not ids:
        raise FileNotFoundError(f"No paired wav/csv files found under {data_root}")
    return ids


def make_splits(ids: Sequence[str], seed: int = 2024) -> Dict[str, List[str]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    if len(ids) < 1871:
        raise ValueError(f"Expected 1871 TASBench items, found {len(ids)}")
    return {
        "train": ids[:1497],
        "val": ids[1497:1684],
        "test": ids[1684:1871],
    }


def load_prompts(csv_path: Path) -> List[str]:
    prompts: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                prompts.append(",".join(row[1:]).strip())
    if not prompts:
        raise ValueError(f"No prompts found in {csv_path}")
    return prompts


def prompt_for_center(prompts: Sequence[str], center_sec: float) -> str:
    frame_index = int(round(center_sec * PROMPT_FPS))
    frame_index = max(0, min(frame_index, len(prompts) - 1))
    return prompts[frame_index]


def prompt_directions(prompt: str) -> List[str]:
    return sorted(set(re.findall(r"\b(left|center|right)\b", prompt.lower())))


def classify_prompt_direction(prompt: str) -> str:
    directions = prompt_directions(prompt)
    if len(directions) == 1:
        return directions[0]
    if len(directions) > 1:
        return "mixed"
    return "none"


def is_single_direction_prompt(prompt: str) -> bool:
    return len(prompt_directions(prompt)) == 1


def direction_to_id(direction: str) -> int:
    if direction not in DIRECTION_TO_ID:
        raise ValueError(f"Cannot map direction category {direction!r} to a single direction id.")
    return DIRECTION_TO_ID[direction]


def angle_degrees_to_direction(angle_degrees: float) -> str:
    if angle_degrees < -30.0:
        return "left"
    if angle_degrees > 30.0:
        return "right"
    return "center"


class SimpleTokenizer:
    pad_token = "<pad>"
    unk_token = "<unk>"

    def __init__(self, vocab: Dict[str, int], max_length: int = 48):
        self.vocab = vocab
        self.max_length = max_length

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1, max_length: int = 48):
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(cls.tokenize_text(text))
        vocab = {cls.pad_token: 0, cls.unk_token: 1}
        for token, count in sorted(counter.items()):
            if count >= min_freq and token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab=vocab, max_length=max_length)

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+|[^\s\w]", text.lower())

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_text(text)[: self.max_length]
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        return ids or [self.vocab[self.unk_token]]

    def batch_encode(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = [self.encode(text) for text in texts]
        max_len = min(self.max_length, max(len(item) for item in encoded))
        batch = torch.zeros(len(encoded), max_len, dtype=torch.long)
        for i, item in enumerate(encoded):
            item = item[:max_len]
            batch[i, : len(item)] = torch.tensor(item, dtype=torch.long)
        return batch


class CLIPPromptTokenizer:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        local_files_only: bool = True,
    ):
        try:
            from transformers import CLIPTokenizer
        except ImportError as exc:
            raise ImportError(
                "CLIP text guidance requires transformers. Install it in the HRTF environment, "
                "for example: F:\\Anaconda\\envs\\HRTF\\python.exe -m pip install transformers"
            ) from exc
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model_name = model_name
        self.max_length = max_length

    @property
    def vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def batch_encode(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


class TASBenchDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        ids: Sequence[str],
        split: str = "train",
        sample_rate: int = SAMPLE_RATE,
        sample_length: int = SAMPLE_LENGTH,
        balanced_direction_sampling: bool = False,
        condition_type: str = "text",
        angle_single_direction_only: bool = False,
        prompt_jitter_sec: float = 0.045,
        candidate_seed: int = 2024,
    ):
        if condition_type not in {"text", "angle"}:
            raise ValueError(f"Unknown condition_type: {condition_type}")
        if condition_type == "angle" and not angle_single_direction_only:
            raise ValueError("condition_type='angle' requires angle_single_direction_only=True.")
        self.data_root = Path(data_root)
        self.audio_dir = self.data_root / "binaural_audios"
        self.prompt_dir = self.data_root / "text_prompts"
        self.ids = list(ids)
        self.split = split
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.condition_type = condition_type
        self.angle_single_direction_only = angle_single_direction_only
        self.balanced_direction_sampling = balanced_direction_sampling and split == "train"
        self.prompt_jitter_sec = prompt_jitter_sec
        self.candidate_seed = candidate_seed
        self.active_direction_categories = (
            ("left", "center", "right") if angle_single_direction_only else DIRECTION_CATEGORIES
        )

        self.prompt_cache = {
            audio_id: load_prompts(self.prompt_dir / f"{audio_id}.csv")
            for audio_id in self.ids
        }
        self.direction_candidates: Dict[str, List[Tuple[str, int]]] = {
            category: [] for category in DIRECTION_CATEGORIES
        }
        self.all_direction_candidates: List[Tuple[str, int, str]] = []
        self.mixed_sampling_probability = 0.0
        if self.balanced_direction_sampling or self.angle_single_direction_only:
            self._build_direction_candidates()

    def _build_direction_candidates(self):
        for audio_id in self.ids:
            info = sf.info(str(self.audio_dir / f"{audio_id}.wav"))
            duration_sec = info.frames / info.samplerate
            prompts = self.prompt_cache[audio_id]
            for frame_index, prompt in enumerate(prompts):
                center_sec = frame_index / PROMPT_FPS
                if center_sec - SAMPLE_SECONDS / 2 < 0:
                    continue
                if center_sec + SAMPLE_SECONDS / 2 > duration_sec:
                    continue
                category = classify_prompt_direction(prompt)
                if category in self.active_direction_categories:
                    self.direction_candidates[category].append((audio_id, frame_index))

        empty = [
            category
            for category in self.active_direction_categories
            if not self.direction_candidates[category]
        ]
        if empty:
            raise RuntimeError(f"Direction-balanced sampling has empty candidate pools: {empty}")
        self.all_direction_candidates = [
            (audio_id, frame_index, category)
            for category in self.active_direction_categories
            for audio_id, frame_index in self.direction_candidates[category]
        ]
        random.Random(self.candidate_seed).shuffle(self.all_direction_candidates)
        total = sum(len(self.direction_candidates[category]) for category in self.active_direction_categories)
        if "mixed" in self.active_direction_categories:
            self.mixed_sampling_probability = len(self.direction_candidates["mixed"]) / total
        else:
            self.mixed_sampling_probability = 0.0

    def direction_sampling_summary(self) -> Dict[str, object]:
        counts = {category: len(self.direction_candidates[category]) for category in self.active_direction_categories}
        return {
            "candidate_counts": counts,
            "mixed_sampling_probability": self.mixed_sampling_probability,
            "single_direction_probability": (1.0 - self.mixed_sampling_probability) / 3.0,
            "angle_single_direction_only": self.angle_single_direction_only,
        }

    def _sample_balanced_candidate(self) -> Tuple[str, int, str]:
        if "mixed" in self.active_direction_categories and random.random() < self.mixed_sampling_probability:
            category = "mixed"
        else:
            category = random.choice(("left", "center", "right"))
        audio_id, frame_index = random.choice(self.direction_candidates[category])
        return audio_id, frame_index, category

    def _sample_single_direction_candidate(self) -> Tuple[str, int, str]:
        return random.choice(self.all_direction_candidates)

    def __len__(self) -> int:
        if self.angle_single_direction_only and self.split != "train":
            return len(self.all_direction_candidates)
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        selected_frame_index = None
        selected_category = None
        if self.balanced_direction_sampling:
            audio_id, selected_frame_index, selected_category = self._sample_balanced_candidate()
        elif self.angle_single_direction_only and self.split == "train":
            audio_id, selected_frame_index, selected_category = self._sample_single_direction_candidate()
        elif self.angle_single_direction_only:
            audio_id, selected_frame_index, selected_category = self.all_direction_candidates[index]
        else:
            audio_id = self.ids[index]
        wav_path = self.audio_dir / f"{audio_id}.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio).transpose(0, 1)
        if waveform.shape[0] != 2:
            raise ValueError(f"Expected stereo audio in {wav_path}, got {waveform.shape[0]} channels")
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        total = waveform.shape[-1]
        if total < self.sample_length:
            waveform = F.pad(waveform, (0, self.sample_length - total))
            total = waveform.shape[-1]

        if selected_frame_index is not None:
            center_sec = selected_frame_index / PROMPT_FPS
            jitter_sec = random.uniform(-self.prompt_jitter_sec, self.prompt_jitter_sec)
            start = int(round((center_sec - SAMPLE_SECONDS / 2 + jitter_sec) * self.sample_rate))
            start = max(0, min(start, total - self.sample_length))
        elif self.split == "train":
            start = random.randint(0, total - self.sample_length)
        else:
            start = max(0, (total - self.sample_length) // 2)

        clip = waveform[:, start : start + self.sample_length].contiguous()
        left, right = clip[0:1], clip[1:2]
        # Follow the paper formulation exactly: Am = Al + Ar, Ad = Al - Ar.
        # These signals may exceed [-1, 1]; do not clip away spatial differences.
        mono = left + right
        diff = left - right

        center_sec = (start + self.sample_length / 2) / self.sample_rate
        prompt = prompt_for_center(self.prompt_cache[audio_id], center_sec)
        direction_category = selected_category or classify_prompt_direction(prompt)
        item = {
            "mono": mono,
            "diff": diff,
            "prompt": prompt,
            "direction_category": direction_category,
            "audio_id": audio_id,
            "start_sec": float(start / self.sample_rate),
        }
        if self.condition_type == "angle":
            item["direction_id"] = direction_to_id(direction_category)
        return item


class TASCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        prompts = [item["prompt"] for item in batch]
        result = {
            "mono": torch.stack([item["mono"] for item in batch]),
            "diff": torch.stack([item["diff"] for item in batch]),
            "prompts": prompts,
            "direction_categories": [item["direction_category"] for item in batch],
            "audio_ids": [item["audio_id"] for item in batch],
            "start_sec": torch.tensor([item["start_sec"] for item in batch], dtype=torch.float32),
        }
        if self.tokenizer is None:
            result["tokens"] = {
                "direction_ids": torch.tensor(
                    [int(item["direction_id"]) for item in batch],
                    dtype=torch.long,
                )
            }
        else:
            result["tokens"] = self.tokenizer.batch_encode(prompts)
        return result


def collect_split_prompts(data_root: Path, ids: Sequence[str]) -> List[str]:
    prompt_dir = data_root / "text_prompts"
    texts: List[str] = []
    for audio_id in ids:
        texts.extend(load_prompts(prompt_dir / f"{audio_id}.csv"))
    return texts
