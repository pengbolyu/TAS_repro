# TASBench minimal training reproduction

This is a small trainable reproduction scaffold for **TAS: Personalized Text-guided Audio Spatialization**.

It uses a frozen `openai/clip-vit-base-patch32` text encoder and does not include paper metrics, inference overlap-add, or baseline comparisons.

The HRTF environment must have `transformers` installed. By default, CLIP is loaded from the local HuggingFace cache to avoid network retries. Add `--allow_model_download` only when the model is not cached yet.

## Fast smoke run

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\train.py --data_root data\TASBench --batch_size 1 --fast_dev_run
```

## Normal training start

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\train.py --data_root data\TASBench --batch_size 12 --max_steps 3000 --lr 2e-4 --val_interval 200 --val_batches 20
```

If CUDA memory is insufficient, reduce `--batch_size` or `--hidden_channels`.

`last.pt` is always updated during training. `best.pt` is saved when validation loss improves.

Training also writes:

- TensorBoard events: `code/TAS_repro/runs/tas_smoke/tensorboard`
- Text log: `code/TAS_repro/runs/tas_smoke/train.log`

To view curves:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" -m tensorboard --logdir code\TAS_repro\runs\tas_smoke\tensorboard --port 6006
```

Then open `http://localhost:6006`.

## Inference

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\infer.py --checkpoint code\TAS_repro\runs\tas_smoke\best.pt --input_wav data\TASBench\binaural_audios\000001.wav --prompt "The piano is on the left." --out code\TAS_repro\runs\tas_smoke\sample_left.wav
```

Inference loads one second of audio, generates a differential waveform, and writes a 16 kHz stereo wav for listening.

For longer listening samples, use sliding-window full-audio inference:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\infer.py --checkpoint code\TAS_repro\runs\tas_smoke\best.pt --input_wav data\TASBench\binaural_audios\000001.wav --prompt "The piano is on the left." --out code\TAS_repro\runs\tas_smoke\sample_left_full.wav --mode full --hop_sec 0.25 --sample_steps 20
```

`--mode full` uses 1-second windows with overlap-add. Smaller `--hop_sec` and larger `--sample_steps` improve continuity/quality but take longer.
