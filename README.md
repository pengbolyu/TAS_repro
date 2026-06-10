# TASBench minimal training reproduction

This is a small trainable reproduction scaffold for **TAS: Personalized Text-guided Audio Spatialization**.

It uses a frozen `openai/clip-vit-base-patch32` text encoder and does not include paper metrics, inference overlap-add, or baseline comparisons.

The data path follows the paper equations directly:

- `mono = left + right`
- `diff = left - right`
- reconstructed `left = (mono + diff) / 2`, `right = (mono - diff) / 2`

These internal tensors are not clipped to `[-1, 1]`; generated wav files are clipped only at write time if needed.

The HRTF environment must have `transformers` installed. By default, CLIP is loaded from the local HuggingFace cache to avoid network retries. Add `--allow_model_download` only when the model is not cached yet.

## Fast smoke run

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\train.py --data_root data\TASBench --batch_size 1 --fast_dev_run
```

## Normal training start

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\train.py --data_root data\TASBench --run_dir code\TAS_repro\runs\direction_enhanced --batch_size 12 --max_steps 3000 --lr 2e-4 --val_interval 200 --val_batches 20
```

If CUDA memory is insufficient, reduce `--batch_size` or `--hidden_channels`.

Direction-enhanced training defaults to:

- `hidden_channels=128`
- balanced left/center/right prompt-frame sampling while preserving the natural mixed-direction ratio
- `total_loss = noise_mse + 0.1 * diff_l1 + 0.1 * ild_loss`

Use `--no_balanced_direction_sampling`, `--diff_loss_weight`, `--ild_loss_weight`, or
`--hidden_channels` for ablation runs.

`last.pt` is always updated during training. `best.pt` is saved when the combined validation
`total_loss` improves.

Training also writes:

- TensorBoard events: `<run_dir>/tensorboard`
- Text log: `<run_dir>/train.log`

Both outputs include the four loss components and cumulative sampled direction ratios.

## Angle-condition diagnostic training

To test whether the audio model can learn audible direction control without CLIP text
understanding, train with angle conditioning:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\train.py --data_root data\TASBench --run_dir code\TAS_repro\runs\angle_single_direction --condition_type angle --angle_single_direction_only --batch_size 12 --max_steps 3000 --lr 2e-4 --val_interval 200 --val_batches 20
```

This keeps prompts with exactly one direction (`left`, `center`, or `right`), including
multi-object same-direction prompts such as `female vocals and piano are on the right`.
It drops prompts containing multiple directions, because a single angle cannot represent
different objects at different positions.

Internally, angle mode uses a learned direction embedding:

- `left -> id 0`
- `center -> id 1`
- `right -> id 2`

`--angle_degrees` is still accepted for convenience and is mapped to the nearest direction:
values below `-30` become left, values above `30` become right, and the rest become center.

Run angle-conditioned inference either with a prompt direction:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\infer.py --checkpoint code\TAS_repro\runs\angle_single_direction\best.pt --input_wav data\TASBench\binaural_audios\000001.wav --prompt "The piano is on the left." --out code\TAS_repro\runs\angle_single_direction\sample_left.wav --sample_steps 50
```

or with an explicit angle:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\infer.py --checkpoint code\TAS_repro\runs\angle_single_direction\best.pt --input_wav data\TASBench\binaural_audios\000001.wav --prompt "angle condition" --angle_degrees -60 --out code\TAS_repro\runs\angle_single_direction\sample_left.wav --sample_steps 50
```

Add `--diff_out` to save the generated differential signal as a mono wav for debugging:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\infer.py --checkpoint code\TAS_repro\runs\angle_single_direction\best.pt --input_wav data\TASBench\binaural_audios\000001.wav --prompt "angle condition" --angle_degrees -60 --out code\TAS_repro\runs\angle_single_direction\sample_left.wav --diff_out code\TAS_repro\runs\angle_single_direction\sample_left_diff.wav --sample_steps 50
```

To view curves:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" -m tensorboard --logdir code\TAS_repro\runs\direction_enhanced\tensorboard --port 6006
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

The default sampler is fast DDIM-like sampling. Use `--sampler ddpm` for a slower step-by-step reverse process closer to the paper's diffusion formulation.

## Analyze generated spatial cues

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\analyze_audio.py code\TAS_repro\runs\tas_smoke\sample_left.wav code\TAS_repro\runs\tas_smoke\sample_center.wav code\TAS_repro\runs\tas_smoke\sample_right.wav
```

Useful cues:

- `ILD(dB)`: positive means left channel is louder; negative means right channel is louder.
- `diff_rms`: left-right difference strength.
- `lr_corr`: left/right waveform correlation; values near 1.0 are close to mono.
- `side/mid`: side energy relative to mid energy.

## Batch prompt-direction evaluation

Generate left/center/right prompts for multiple TASBench items and summarize whether ILD follows the expected order:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\batch_eval_prompts.py --checkpoint code\TAS_repro\runs\paper_formula\best.pt --data_root data\TASBench --split val --num_items 20 --mode clip --sample_steps 20 --sampler ddim --out_dir code\TAS_repro\runs\prompt_eval
```

By default, prompts reuse the first object name from each item's TASBench CSV, for example `The keyboard is on the left.` Set `--prompt_template generic` to use `The sound source ...` prompts instead.

For a cleaner ILD sanity check, restrict evaluation to prompts containing one object and one spatial relation. This filters both multi-direction prompts and phrases such as `female vocals and piano`:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\batch_eval_prompts.py --checkpoint code\TAS_repro\runs\paper_formula\best.pt --data_root data\TASBench --split val --num_items 20 --mode clip --sample_steps 20 --sampler ddim --single_source_only --out_dir code\TAS_repro\runs\prompt_eval_single_source
```

For listening-ready full-audio evaluation, add `--mode full --hop_sec 0.25 --save_audio`, but it will be much slower and writes three wav files per item.

For angle checkpoints, use `--single_direction_only` to preserve multi-object same-direction
items while filtering multi-direction prompts:

```powershell
& "F:\Anaconda\envs\HRTF\python.exe" code\TAS_repro\batch_eval_prompts.py --checkpoint code\TAS_repro\runs\angle_single_direction\best.pt --data_root data\TASBench --split val --num_items 50 --mode clip --sample_steps 20 --sampler ddim --single_direction_only --out_dir code\TAS_repro\runs\angle_single_direction\prompt_eval_val50
```
