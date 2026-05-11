# Hölder-MPO on ALFWorld

This subtree is a fork of `verl-agent` extended with the Hölder-MPO loss for
agent-style training. The upstream `README.md` documents environment / vLLM
setup; the additions for Hölder-MPO are minimal and listed here.

## Entry point

```
bash examples/holder_mpo_trainer/run_alfworld.sh <engine> <variant>
```

- `<engine>` — `vllm` (default) or `sglang`
- `<variant>` — `sequence` (`holder_seq`, default in the paper) or `token`
  (`holder_token`)

Hölder-$p$ is configured via env vars:

| variable                  | default     | meaning                              |
|---------------------------|-------------|--------------------------------------|
| `HOLDER_P`                | `1.0`       | initial / constant $p$               |
| `HOLDER_P_SCHEDULE`       | `constant`  | `constant`, `linear`, `linear_dec`, `sin`, `cos`, `poly`, `poly_dec`, `cubic`, `cubic_dec` |
| `HOLDER_P_MIN` / `_MAX`   | `-2.0` / `2.0` | bounds of the schedule            |
| `HOLDER_P_SCHEDULE_STEPS` | `1000`      | how many steps the schedule covers   |
| `HOLDER_P_POWER`          | `2.0`       | exponent for the polynomial schedule |

## Where the loss is implemented

- `verl/trainer/ppo/core_algos.py` — Hölder-$p$ ratio + sequence/token aggregation
- `verl/workers/actor/dp_actor.py` — calls into the loss

Search the tree for `holder` to see all touched files; no upstream verl files
were renamed.
