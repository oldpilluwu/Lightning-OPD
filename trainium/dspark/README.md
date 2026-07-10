# DSpark on Trainium compatibility experiment

This directory tests whether upstream vLLM's DSpark speculative decoder can be
used through `vllm-neuron`. It is isolated from the working SFT-generation
environment: the AWS-managed `/opt` environment is inherited read-only and an
upstream vLLM checkout is exposed only through `PYTHONPATH`.

## Run on the Trainium instance

From the repository root:

```bash
bash trainium/dspark/setup_experiment.sh
bash trainium/dspark/run_dspark_probe.sh
```

The first probe does not download model weights or compile a model. It checks
the installed versions, imports the Qwen3 DSpark model and GPU DSpark worker,
and scans `vllm-neuron` for an integration.

Compare with the currently installed vLLM 0.16 stack:

```bash
USE_UPSTREAM_VLLM=0 bash trainium/dspark/run_dspark_probe.sh
```

Only if the static probe finds Neuron integration, or if you want the exact
engine failure, request the expensive test explicitly:

```bash
ATTEMPT_ENGINE=1 bash trainium/dspark/run_dspark_probe.sh
```

Defaults for the engine attempt are deliberately tiny (`max_model_len=2048`,
`max_num_seqs=1`, TP=4). Override them with the corresponding environment
variables. Logs are written under `trainium/dspark/logs/`.

## Expected result

The current hypothesis is **not yet supported**: recent upstream vLLM contains
`Qwen3DSparkModel`, but its DSpark proposer lives in the GPU worker tree, while
the current `vllm-neuron` plugin has no DSpark proposer implementation. The
probe is intended to verify that boundary against the exact revisions installed
on the instance rather than treating it as an assumption.

Exit codes 20 and 21 are intentional experimental results:

- `20`: the selected vLLM revision does not include Qwen3 DSpark.
- `21`: upstream DSpark exists, but no `vllm-neuron` integration was found.

