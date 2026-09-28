# Reproduce the recovery experiments

Use the isolated branch `codex/elo-recovery`. Nothing here requires changing `sweep` or
its environment. All teacher work is offline; only this team's own checkpoint initializes
training. Do not include the corpus, teacher or checkpoint archives in a submission.

## Restore the completed training pass

The archive `/home/edwardb/Documents/chessathon/audit/elo-recovery-artifacts/training-pass-20260911.tar.gz`
contains both labelled sample sets, packed corpora, best checkpoints, quantized candidates,
and the exact original pipeline scripts. `manifest.json` alongside it gives individual
SHA-256 hashes. Restore into an empty directory. The scripts record original absolute
paths; adapt those paths before rerunning, and choose fresh output/checkpoint directories.
The original 8 GB replay corpus and original epoch030.pt are read-only external inputs and
are not duplicated in this archive.

## Build a quiet corpus and train a candidate

From the worktree, with the existing Python environment activated, these are the relevant
commands (substitute paths for your restored artifacts). Preserve the source split manifest.

```sh
python -m tools.leaf_corpus --labels /tmp/recovery-labels-10k/labels.jsonl \
  /tmp/recovery-late-labels/labels.jsonl \
  --splits experiments/recovery/corpus-split.json --out /tmp/new-quiet-corpus \
  --replay /home/edwardb/Documents/chessathon/data/train.shuffled.bin \
  --replay-count 1048576 --leaf-repeat 16 --quiet-only

nice -n 15 python tools/train.py --data /tmp/new-quiet-corpus/mixed-train.bin \
  --holdout-data /tmp/new-quiet-corpus/validation.bin --checkpoints /tmp/new-quiet-flat \
  --epochs 6 --batch 2048 --lr 0.0001 --phase-balanced --selection-cp-max 300 \
  --king-buckets 1 --device cpu --threads 1 --seed 20260911 \
  --resume /home/edwardb/Documents/chessathon/data/checkpoints/epoch030.pt

python tools/quantise.py --checkpoint /tmp/new-quiet-flat/best.pt \
  --out /tmp/new-quiet-flat.npz
python -m tools.eval_quantized --data /tmp/new-quiet-corpus/validation.bin \
  --weights /tmp/new-quiet-flat.npz --out /tmp/new-validation.json
```

Eight king buckets use `--king-buckets 8`, a fresh checkpoint directory and fresh export.
Training does not automatically replace `weights/net.npz`. Do not promote solely from
loss: the first candidate's apparent loss improvement accompanied a 38.75% match score.

## Search speed and playing strength

Freeze runtime source and weights into separate directories before invoking either tool.
Changing files while workers initialize invalidates the experiment. New runs record input
hashes; the earliest results predate that addition and identify their frozen directories.

```sh
nice -n 15 python tools/cpu_compare.py --agent /tmp/recovery-bounded \
  --opponent /tmp/recovery-compact-state --positions experiments/recovery/development.json \
  --out /tmp/new-bounded-cpu.json --repeat 2 --agent-env CHESS_BOUNDED_SEE=1

nice -n 15 python tools/node_match.py --agent /tmp/recovery-net-quiet-flat \
  --opponent /tmp/recovery-compact-state --openings experiments/recovery/acceptance-openings.json \
  --out /tmp/new-quiet-match.json --pairs 20 --nodes 65536
```

The old acceptance openings have now been inspected in candidate comparisons. They provide
screening evidence, not a fresh final confirmation set. Budget a larger match on new source
games and a wall-time comparison before declaring a strength gain. The CPU comparison
checks identical move/depth/score/node results, not Elo. Node matches do not test platform
initialization or time-management compliance; use the unchanged harness separately.

## Broader king-bucket run

The follow-up uses the same corpus builder with `--replay-count 4194304` and otherwise
identical quiet filtering, split manifest, leaf repeat and seed. It produces 4,308,095
training records after excluding protected replay duplicates. Its trainer command is:

```sh
nice -n 15 python tools/train.py --data /tmp/recovery-broad-corpus/mixed-train.bin \
  --holdout-data /tmp/recovery-broad-corpus/validation.bin \
  --checkpoints /tmp/recovery-broad-king8 --epochs 4 --batch 2048 --lr 0.0001 \
  --phase-balanced --selection-cp-max 300 --king-buckets 8 --device cpu --threads 1 \
  --seed 20260911 \
  --resume /home/edwardb/Documents/chessathon/data/checkpoints/epoch030.pt
```

Unlike the earlier pilots, this run includes the original initialized function in best
checkpoint selection. Epoch zero can therefore win if all trained epochs regress. This
safeguard does not make the validation objective an Elo estimate.
