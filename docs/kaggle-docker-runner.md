# Kaggle Docker Runner

This repo includes a local replay harness that gets closer to Kaggle's hosted
simulation runner than in-process `play_kaggle_selfplay.py`:

- runs inside Kaggle's public Python image: `gcr.io/kaggle-images/python`
- mounts the packaged submission bundle read-only
- launches one persistent worker container per seat
- disables network in the worker containers
- replays a saved Kaggle episode JSON through the packaged `main.py`

Note: the current Kaggle public image has a broken `/usr/local/bin/python`
placeholder, so the harness invokes `/usr/bin/python3` explicitly.

It does **not** perfectly reproduce Kaggle's private competition backend. In
particular, it does not guarantee:

- the exact worker orchestration and scheduling used in validation
- the exact resource limits on hosted workers
- the exact outer timing overhead of Kaggle's infrastructure

It is still a useful approximation when comparing:

- local venv vs Kaggle Docker image
- in-process selfplay vs per-seat isolated workers
- old submission bundle vs new submission bundle

## Files

- `scripts/replay_submission_in_kaggle_docker.sh`
- `scripts/replay_submission_worker.py`

## Example

```bash
scripts/replay_submission_in_kaggle_docker.sh \
  dist/2p-4p.tar.gz \
  records/bad-validation/78891162.json \
  --out-dir records/docker-replay-78891162
```

This writes one JSONL file per seat:

- `records/docker-replay-78891162/seat-0.jsonl`
- `records/docker-replay-78891162/seat-1.jsonl`
- `records/docker-replay-78891162/seat-2.jsonl`
- `records/docker-replay-78891162/seat-3.jsonl`

Each line contains:

- `step`
- `seat`
- `player`
- `duration`
- `remainingOverageTime`
- `status`
- `reward`
- `num_actions`
- `action`
- `stdout`
- `stderr`

## Useful options

Use only the opening steps:

```bash
scripts/replay_submission_in_kaggle_docker.sh \
  dist/2p-4p.tar.gz \
  records/bad-validation/78891162.json \
  --step-limit 10
```

Pass extra docker arguments, for example CPU pinning:

```bash
scripts/replay_submission_in_kaggle_docker.sh \
  dist/2p-4p.tar.gz \
  records/bad-validation/78891162.json \
  --docker-arg --cpuset-cpus=0 \
  --docker-arg --memory=4g
```

Skip image pull if you already have the image locally:

```bash
scripts/replay_submission_in_kaggle_docker.sh \
  dist/2p-4p.tar.gz \
  records/bad-validation/78891162.json \
  --no-pull
```
