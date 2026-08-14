#!/usr/bin/env python3
"""Mirror a lerobot-train log into TensorBoard event files.

LeRobot's training script has no TensorBoard support -- its only tracker is
Weights & Biases (`--wandb.enable`), and `MetricsTracker` otherwise just prints
a line every `--log_freq` steps. That line already carries every scalar worth
plotting, so rather than adding a tracker to the vendored trainer (which would
have to be re-applied on every upstream update) this reads the log it already
writes.

Runs incrementally: re-running only appends steps newer than what is already in
the event file, so it is safe to call repeatedly, or with --follow to watch a
run in progress.

TensorBoard lives in the `isaaclab` env here, not in `lerobot`:

    conda activate isaaclab
    python train_log_to_tensorboard.py --log /mnt/hdd/relay_datasets/train.log \\
        --out /mnt/hdd/relay_datasets/tb_smolvla --follow
    tensorboard --logdir /mnt/hdd/relay_datasets/tb_smolvla
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# `step:800 smpl:51K ep:79 epch:0.16 loss:0.093 grdn:1.569 lr:9.8e-05
#  updt_s:0.593 data_s:0.009 smp/s:106 mem_gb:14.24`
#
# Every count field is abbreviated once it passes 1000 -- `smpl:51K`, `ep:1K`,
# and `step:10K` too. So the step in the line is BOTH non-numeric and lossy:
# steps 9800 and 10200 both print as "10K". The step therefore cannot be read
# off the line at all.
#
# It does not have to be. The trainer emits exactly one metrics line every
# log_freq steps, so the n-th such line is step n*log_freq -- exact, and stable
# across restarts because the whole log is re-read from the top each pump().
# log_freq is measured from the first line, whose step is below 1000 and so is
# still printed in full.
STEP_RE = re.compile(r"\bstep:(\d+)(?![\dKM])")
STEP_ABBREV_RE = re.compile(r"\bstep:(\d+(?:\.\d+)?)([KM])\b")
FIELD_RE = re.compile(r"\b([a-zA-Z_/]+):(-?\d+\.?\d*(?:[eE][-+]?\d+)?)(?![\dKM])")
# "py" is not a metric: the logging prefix ends in `ot_train.py:641`, which has
# the same name:number shape as every field on the line.
SKIP = {"step", "smpl", "ep", "py"}

# `INFO 2026-08-13 22:38:56 ot_train.py:673 step 1000: eval_loss=0.0721`
# A different code path from MetricsTracker, hence a different shape: plain
# `step N`, and the step is never abbreviated here.
EVAL_RE = re.compile(r"\bstep (\d+): eval_loss=(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")

SCALE = {"K": 1_000, "M": 1_000_000}


def parse_metrics(line: str):
    """(rounded_step, {metric: value}) for a metrics line, or None.

    The step is only returned to sanity-check the ordinal-derived one; it is
    rounded for anything past 1000 and must not be used as the x-axis.
    """
    exact = STEP_RE.search(line)
    if exact:
        step = int(exact.group(1))
    else:
        abbrev = STEP_ABBREV_RE.search(line)
        if not abbrev:
            return None
        step = int(float(abbrev.group(1)) * SCALE[abbrev.group(2)])
    metrics = {}
    for name, value in FIELD_RE.findall(line):
        if name in SKIP:
            continue
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return (step, metrics) if metrics else None


def parse_eval(line: str):
    """(step, eval_loss) for a validation line, or None."""
    match = EVAL_RE.search(line)
    return (int(match.group(1)), float(match.group(2))) if match else None


# Group related scalars so TensorBoard draws them together rather than as a
# flat alphabetical list.
GROUP = {
    "loss": "train/loss",
    "grdn": "train/grad_norm",
    "lr": "train/lr",
    "epch": "train/epoch",
    "updt_s": "time/update_s",
    "data_s": "time/dataloading_s",
    "smp/s": "throughput/samples_per_s",
    "mem_gb": "system/mem_gb",
}


def pump(log_path: Path, writer: SummaryWriter, seen: dict) -> bool:
    """Mirror everything in the log newer than `seen`. True if anything moved."""
    moved = False
    ordinal = 0
    log_freq = seen.get("log_freq")
    with log_path.open(errors="replace") as f:
        for line in f:
            evaluation = parse_eval(line)
            if evaluation is not None:
                step, eval_loss = evaluation
                if step > seen["eval"]:
                    writer.add_scalar("eval/loss", eval_loss, step)
                    seen["eval"] = step
                    moved = True
                continue

            parsed = parse_metrics(line)
            if parsed is None:
                continue
            reported_step, metrics = parsed
            ordinal += 1
            if log_freq is None:
                # The first metrics line is step == log_freq, printed in full.
                log_freq = reported_step
                seen["log_freq"] = log_freq
            step = ordinal * log_freq
            # The abbreviation rounds to the nearest 1000, so up to 500 of
            # disagreement is expected and means nothing. This is not trying to
            # catch a single dropped line (200 of shift, below the noise) but a
            # wrong log_freq, whose error compounds every line.
            if abs(step - reported_step) > 500 + 0.02 * step:
                print(f"  step drift: line {ordinal} implies {step}, log says {reported_step}",
                      flush=True)
            if step <= seen["train"]:
                continue
            for name, value in metrics.items():
                writer.add_scalar(GROUP.get(name, f"other/{name}"), value, step)
            seen["train"] = step
            moved = True
    writer.flush()
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="lerobot-train stdout log")
    parser.add_argument("--out", type=Path, required=True, help="TensorBoard log directory")
    parser.add_argument("--follow", action="store_true", help="keep watching the log")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between polls with --follow")
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=str(args.out))
    seen = {"train": 0, "eval": 0, "log_freq": None}
    try:
        while True:
            if pump(args.log, writer, seen):
                print(f"wrote up to step {seen['train']} (eval {seen['eval']})", flush=True)
            if not args.follow:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()


if __name__ == "__main__":
    main()
