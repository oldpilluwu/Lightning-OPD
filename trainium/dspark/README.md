# DSpark on Trainium with the installed vLLM-Neuron stack

This directory probes DSpark without cloning, overlaying, or installing upstream
vLLM. It uses the vLLM and `vllm-neuron` packages already present in the AWS
Neuron inference environment and configures the engine for NxD Inference.

DSpark needs two checkpoints:

- target: `Qwen/Qwen3-8B`
- speculator: `deepseek-ai/dspark_qwen3_8b_block7`

The engine attempt lets Hugging Face/vLLM download both checkpoints. Set
`TARGET_MODEL` or `SPECULATOR_MODEL` to local checkpoint directories to avoid a
download or to use an existing cache.

## Run on the Trainium instance

From the repository root:

```bash
bash trainium/dspark/setup_experiment.sh
bash trainium/dspark/run_dspark_probe.sh
```

`setup_experiment.sh` only validates `neuron-ls`, the selected Python
environment, package versions, and imports. It does not create a virtualenv,
clone vLLM, or install anything.

The default probe is static: it reports whether the installed vLLM includes the
Qwen3 DSpark model and proposer, and whether `vllm-neuron` contains a DSpark
integration. Logs are written under `trainium/dspark/logs/`.

To request a faithful DSpark engine construction:

```bash
ATTEMPT_ENGINE=1 bash trainium/dspark/run_dspark_probe.sh
```

The Neuron attempt passes the target and speculator separately through vLLM's
`speculative_config`, selects the NxD Inference plugin, disables prefix caching,
uses one contiguous KV-cache slot per compiled sequence, and pins the static
Neuron graph to `MAX_MODEL_LEN` and `MAX_NUM_SEQS`.

## Expected compatibility boundary

Upstream DSpark is not an ordinary small causal draft model. Its Qwen3/DFlash
backbone creates a non-causal query block in parallel, then its Markov head adds
a token-to-token transition bias while the DSpark proposer samples the block
sequentially. The proposer currently lives under vLLM's GPU worker tree.

The vLLM 0.16 line used by `vllm-neuron` 0.5 does not accept `method="dspark"`.
The Neuron plugin's fused-speculation loader has special handling for EAGLE but
does not implement DSpark's non-causal block proposal or Markov head. A faithful
run should therefore stop before compilation until both pieces are ported to
NxD Inference/vLLM-Neuron.

For a narrower diagnostic, the checkpoint can be offered to NxDI as a generic
draft model:

```bash
ATTEMPT_ENGINE=1 SPECULATIVE_METHOD=draft_model \
  bash trainium/dspark/run_dspark_probe.sh
```

This is intentionally labeled **not DSpark**. It only tests checkpoint loading
through NxDI generic fused speculation. It does not execute DSpark semantics and
must not be used for performance or correctness claims.

Useful overrides:

```bash
INFER_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16 \
TP_SIZE=4 MAX_MODEL_LEN=2048 MAX_NUM_SEQS=1 \
ATTEMPT_ENGINE=1 bash trainium/dspark/run_dspark_probe.sh
```

Exit codes `20` and `21` are intentional static results:

- `20`: installed vLLM does not contain the Qwen3 DSpark model.
- `21`: vLLM contains DSpark, but installed `vllm-neuron` has no DSpark hook.
