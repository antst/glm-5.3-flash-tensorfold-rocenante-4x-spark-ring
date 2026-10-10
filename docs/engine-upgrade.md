# Mia107 image upgrade on four Sparks

The newest pinned Mia image passed the full four-rank correctness and benchmark suite with prepared RoCEnante. Release **v0.1.2** upgrades the v0.1.1 recipe; the historical GitHub v0.1.1 tag remains unchanged.

TensorFold remains **0.6.0 with 107 Mia patches**, upgraded from 96 patches. It is not TensorFold 1.0.4: the native 1.x release currently qualifies GLM on Metal/MLX, not our CUDA/EXL3 setup.

Both images completed 30 model checks, 120 SparkDash requests and 54 RigMark requests. CPU checks passed 90/90; Mia’s synthetic GPU DFlash prelaunch check also passed. All 16 HCA functions and eight hardware diagonal rules carried payload without the checked error counters increasing.

## Matched generation results

Eight configured decode lanes in both arms. C1/C4/C8 are concurrent clients, not lane settings. Values are aggregate decode tokens/s medians from three 400-token waves; raw ranges and samples are in [the result receipt](../results/mia107-upgrade.json).

| Workload | Clients | Mia96 | Mia107 | Change |
|---|---:|---:|---:|---:|
| code | 1 | 147.27 | 153.27 | +4.07% |
| code | 4 | 226.89 | 231.46 | +2.01% |
| code | 8 | 302.18 | 303.51 | +0.44% |
| prose | 1 | 99.13 | 104.54 | +5.46% |
| prose | 4 | 167.78 | 174.33 | +3.90% |
| prose | 8 | 222.28 | 234.00 | +5.27% |
| structured | 1 | 184.62 | 192.80 | +4.43% |
| structured | 4 | 387.39 | 419.43 | +8.27% |
| structured | 8 | 505.38 | 521.98 | +3.29% |

## RigMark

Completed outputs use five runs per workload; cold/replay prefill and short-code concurrency use three runs. The same pinned RigMark prompts and settings were used in both arms.

| Metric (tokens/s) | Mia96 | Mia107 | Change |
|---|---:|---:|---:|
| Completed code | 122.08 | 127.33 | +4.31% |
| Completed prose | 70.84 | 75.46 | +6.52% |
| Completed structured | 171.64 | 179.10 | +4.35% |
| Cold 8192 prefill | 1828.71 | 1865.03 | +1.99% |
| Cold 32768 prefill | 1848.20 | 1870.45 | +1.20% |
| Cold 65536 prefill | 1828.05 | 1883.82 | +3.05% |
| Short code C1 E2E | 92.46 | 97.59 | +5.56% |
| Short code C2 E2E | 126.74 | 133.87 | +5.62% |
| Short code C4 E2E | 171.31 | 166.75 | -2.66% |

These are sequential point estimates, with ranges and samples retained. The experiment does not establish statistical significance or isolate individual patches. Startup and model checks preceded measurement. Cold prefill excludes prefix-cache reuse; kernel/program caches persist across requests and can warm during the suite. The first new-image 8K sample was slower than later repetitions; its cause was not profiled. Historical [Mia/Alex/transport comparisons](performance.md) retain their original images and source scopes.

## Components and scope

- New image: `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14b5072e8e9918f1080153f57321415a387bc7d78cf9c9b44a7aeae2d4399ccf`. Prior image: `ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:bc34d7d63f978cf601f42863b284bc95a567c50c10e9adb0866a635be568bf5f`.
- Transformers changes from 5.18.0 to 5.19.0. The NVIDIA 26.07 base, PyTorch, Triton, CuTe, CUDA Python, weights and DFlash2 revision remain unchanged.
- Mia107 adds DFlash prelaunch, prompt selection/projection and CUDA indexer score/selection paths, plus small-row Q4 changes. TP3-specific head specializations do not apply to our TP4 local 16-head shape.
- Our prompt-fill row limit remains the engine default 1024. Mia’s launcher now defaults to 2048; that scheduler change was excluded from this image comparison.
- NCCL 2.32.3 and the pinned SparkRing/prepared B12X transport are unchanged. The exact NCCL binary was copied to a new artifact directory and CPU-loaded in the new image; it was not rebuilt. Its new derived receipt records the loader proof and original build lineage.
- The owner, shared root, physical mesh, endpoint routes, hardware forwarding rules, API8888 and performance knobs are unchanged. No network apply/rollback, NIC reload, route/forwarding policy or package/driver changes were performed for this upgrade.

## Upgrade operation and recovery

The operator archived the prior configuration, runtime source guards and both canonical qualification receipts before selecting the new image. All four hosts pulled the digest; the old image and cache were retained. The new image creates its own host-local compiler/cache namespace under the existing cache root. Fresh NCCL and native qualifications authorized the new workers.

For a clean installation, build against this checkout’s pinned image and run the normal qualifications. For an existing deployment, stop its owned workers before selecting a coherent checkout and updating the image/config/library receipt together. Preserve the same owner/root to reuse its healthy mesh. Archive old receipts; both source-bound transport qualifications and the model checks must be rerun. Never bypass the reviewed source hashes.

A copied library can be reused only with a fresh CPU loader proof in the new image and an explicit derived build receipt preserving its exact binary SHA and original build lineage. Alternatively, use the native builder with a new output directory and this image as `--loader-image`. No global library replacement is required.

The passwordless SSH operator and reboot/start commands are unchanged; see [system changes and recovery](system-changes.md). A physical reboot, vision input, a full 1M-token prompt and eight simultaneous 1M prompts remain untested.

## Further updates

- [TensorFold 1.0.4](https://github.com/ashhart/TensorFold/releases/tag/v1.0.4): native rewrite; no supported GLM CUDA/EXL3 replacement for this deployment. The maintained Python 0.6.6 branch is a separate port that currently has 27 conflicting files against Mia107.
- [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/commits/main): our revision remains latest. The latest GLM model-repository change affects documentation/licensing; all 83 safetensor shard objects and runtime config/tokenizer assets match our pinned snapshot.
- [B12X moved to FlashInfer](https://github.com/local-inference-lab/b12x#b12x-has-moved-to-flashinfer). The source migration needs a separate CuTe/dependency and graph/transport qualification; installing the current released FlashInfer package does not bring in that merge.
- [NVIDIA PyTorch 26.09](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/pytorch/26.09-py3): newer PyTorch/CUDA base available, left for a separate compatibility and performance trial.

See the [pinned Mia changelog](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/b5808ac620c795ec9c0640f3ca2a2f7be21aaa63/CHANGELOG.md) for the upstream patches. All exact commands, first-attempt failure/rollback, changes and cleanup are retained in private operator journals.

## Measured tradeoffs and live confirmation

The new image remains selected for this generation-priority profile. The gains are workload dependent: C8 SparkDash code improves only 0.44%, and its three-wave ranges overlap. RigMark short-code C4 E2E is 2.66% lower, with overlapping ranges (166.98–172.12 before, 165.83–177.37 after). This small sample does not establish a repeatable C4 regression or a universal improvement.

Cached replay TTFT medians increase from about 32 to 43 ms at 8K, 42 to 46 ms at 32K, and 54 to 57 ms at 64K. Their apparent prompt-tokens/s rates consequently fall; those rates measure replay of reused prompt states over very short intervals. The cold 8K samples are 1330.04, 1865.03 and 1868.11 tokens/s; all samples remain in the receipt, without attributing the first slower request to a profiled cause.

Final observations verify the exact new image on all four hosts, an idle API8888 with eight lanes and advertised 1M context, and exactly 204 completed suite requests. Both CUDA score extensions are mapped into every model process, with no logged score-kernel fallback. No temporary upgrade containers remain. The old phase-one model remains stopped.
