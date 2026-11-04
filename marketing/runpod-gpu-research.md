# RunPod GPU Rental for VisionBrain Demo Sessions

**Research date: 2026-09-05.** All prices verified from RunPod primary sources on this date unless marked otherwise. GPU pricing changes frequently — treat every number here as "as of Sept 2026" and re-check the [RunPod pricing page](https://www.runpod.io/pricing) before a session.

## Bottom line

- **Recommended setup: one RTX Pro 6000 (96 GB, Blackwell) Secure Cloud pod at $2.09/hr**, on-demand, billed per second. It is the cheapest 94–96 GB-class GPU on RunPod and is the smallest single GPU that runs Falcon + SAM 3.1 + Gemma 4 26B **all in bf16 simultaneously** with comfortable headroom (est. ~70–80 GB total working set).
- **A 1-hour demo costs roughly $2.10 warm / ~$2.60 cold** on that GPU (arithmetic below). Storage adds $7/month for a 100 GB persistent network volume holding the weights.
- **Monthly cost: ~$16/mo at 4 demo sessions, ~$34/mo at 12** (warm-start, 100 GB volume included).
- **Do not keep a "warm" stopped pod.** A stopped pod still bills its local volume disk at $0.20/GB/mo ($20/mo for 100 GB) while buying you almost nothing — a terminated pod boots from a network volume in minutes. Terminate the pod, keep the network volume ($7/mo).
- **Critical caveat — read [the MLX section](#the-mlx-caveat) before planning anything:** MLX does not run on NVIDIA GPUs. RunPod cannot run VisionBrain's actual inference code path; it runs the **upstream PyTorch implementations** of the same models with original Hugging Face checkpoints. This is a porting/demo-stack effort, not a lift-and-shift.

---

## 1. Billing model

Verified from [docs.runpod.io/pods/pricing](https://docs.runpod.io/pods/pricing) and [runpod.io/pricing](https://www.runpod.io/pricing) (both fetched 2026-09-05):

- **Billed by the second** for compute and most storage; network volumes are billed hourly. A 16-minute job on a $2/hr GPU costs ~$0.53, not a full hour. ([docs pricing page](https://docs.runpod.io/pods/pricing), fetched 2026-09-05)
- **On-demand** pods are dedicated and cannot be displaced — the right tier for a live demo (an interrupted recording is unrecoverable).
- **Interruptible ("spot")** pods are ~50% cheaper on average — RunPod's own blog gives the example $0.232/hr spot vs $0.491/hr on-demand for the A6000, updated 2025-08-25 on a 2022 post — but they "can be interrupted without notice." Not recommended for demos; fine for rehearsing or pre-downloading weights. ([Spot vs On-Demand blog](https://www.runpod.io/blog/spot-vs-on-demand-instances-runpod), fetched 2026-09-05). Spot rates are not published on a static page; the console shows current prices per GPU.
- **Community Cloud vs Secure Cloud**: same GPUs, two host tiers. Community (consumer/host-run hardware) is typically 20–40% cheaper per GPU (see table below). Secure Cloud is datacenter-grade with more reliable availability — worth the premium for a timed demo session.
- **Savings plans** (3/6-month commitments) exist but are irrelevant for a few hours of demos per month.
- **No minimum charge is stated anywhere** on the pricing page or docs (checked 2026-09-05). The only floor is a credit requirement: "You must have at least one hour's worth of credits for your selected configuration to deploy an on-demand instance" ([docs pricing page](https://docs.runpod.io/pods/pricing)).
- **No ingress or egress data fees**: "Pods are billed by the second for compute and storage, with no fees for data ingress or egress" ([docs pricing page](https://docs.runpod.io/pods/pricing), fetched 2026-09-05). So streaming recorded demo footage off the pod is free.
- **Default account spend limit**: $80/hour across all resources — far above anything a demo needs.
- **Stopped pod**: GPU billing stops immediately. Storage keeps accruing (see §3). If your prepaid balance hits $0, running pods are stopped and pods without network volumes are terminated with data loss.

## 2. On-demand GPU prices (per GPU-hour, as of 2026-09-05)

From [runpod.io/pricing](https://www.runpod.io/pricing) (fetched 2026-09-05). The page's structured data lists Community Cloud rates; the visible table lists Secure Cloud rates — both reproduced here. **RTX 5090 offered at 32 GB.**

| GPU | VRAM | Community Cloud | Secure Cloud |
|---|---|---|---|
| B300 | 288 GB | $6.94 | $7.89 |
| B200 | 180 GB | $5.98 | $6.79 |
| **H200** | 141 GB | $3.59 | $4.59 |
| H100 NVL | 94 GB | $2.59 | $3.19 |
| H100 SXM | 80 GB | $2.69 | $3.29 |
| H100 PCIe | 80 GB | $1.99 | $2.89 |
| **RTX Pro 6000 (Blackwell)** | **96 GB** | **$1.69** | **$2.09** |
| A100 SXM | 80 GB | $1.39 | $1.59 |
| A100 PCIe | 80 GB | $1.19 | $1.39 |
| **L40S** | 48 GB | $0.79 | $0.99 |
| L40 | 48 GB | $0.69 | $0.82 |
| RTX 6000 Ada | 48 GB | $0.74 | $0.84 |
| RTX A6000 | 48 GB | $0.33 | $0.53 |
| A40 | 48 GB | $0.35 | $0.44 |
| **RTX 5090** | **32 GB** | **$0.69** | **$0.99** |
| **RTX 4090** | 24 GB | $0.34 | $0.74 |
| L4 | 24 GB | $0.44 | $0.49 |
| RTX 3090 | 24 GB | $0.22 | $0.50 |

Cross-check: a third-party guide sampling RunPod's page on 2026-08-20 lists RTX 4090 $0.74, A100 PCIe $1.39, H100 PCIe $2.89, H200 $4.59, B300 $7.89 — all matching the Secure Cloud column above ([Hivenet RunPod pricing guide](https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs), fetched 2026-09-05).

## 3. Storage

From [docs.runpod.io/pods/storage](https://docs.runpod.io/pods/storage) and the [docs pricing page](https://docs.runpod.io/pods/pricing) (fetched 2026-09-05):

| Storage type | Running | Stopped | Notes |
|---|---|---|---|
| Container disk | $0.10/GB/mo (prorated per second) | **not charged** (contents erased on stop) | ephemeral |
| Volume disk (local to pod) | $0.10/GB/mo | **$0.20/GB/mo** | persists across stop; billed while stopped |
| **Network volume, standard** | **$0.07/GB/mo (<1 TB); $0.05/GB/mo (>1 TB)** | same | persists independent of any pod; attachable to multiple pods |
| Network volume, high-performance | $0.14/GB/mo | same | up to 3x throughput / 4x IOPS |

- Network volumes mount at `/workspace` (replacing the volume disk), must be attached at pod creation, survive stop/terminate, and can move between pods. RunPod cautions it "is not designed for long-term cloud storage" — keep a backup of anything irreplaceable.
- Bandwidth/egress: **no data ingress or egress fees** ([docs pricing page](https://docs.runpod.io/pods/pricing), fetched 2026-09-05).
- For a demo rig: a **100 GB standard network volume = $7.00/mo** flat, billed hourly regardless of pod state.

## 4. The MLX caveat — and what RunPod would actually run

> **This is the single most important caveat in this document.**

- **MLX is Apple-Silicon-only** (Metal backend). There is no CUDA build of MLX, so none of VisionBrain's MLX-based inference code — `fp_inference.py`, `sam3_inference.py`, `live_tracking.py`, the `local` VLM path (`vlm_registry.py` / `model_host.py`), or the `mlx-community/sam3.1-bf16` weight layout (which only loads thanks to the MLX-layout sanitize pass in the local mlx_vlm patch) — **can run on a RunPod GPU**.
- What RunPod *can* run is the **upstream PyTorch implementations** of the same three models, with the **original Hugging Face checkpoints** (not the `mlx-community` conversions):
  - **SAM 3.1** — upstream Facebook/Meta SAM 3 checkpoint via PyTorch/transformers (the mlx-community snapshot is an MLX-layout conversion and is useless on CUDA).
  - **Falcon Perception** — the upstream Falcon-Perception repo (which VisionBrain already reads from) is a PyTorch codebase.
  - **Gemma 4 26B** — original Google checkpoint served via transformers, vLLM, or Ollama.
- **Practical consequence for the demo goal ("all three at full power"):** this is a small porting project, not a configuration exercise. VisionBrain itself has no CUDA code path today. The lowest-friction slice already exists: `gemma_inference.available_backend()` already speaks HTTP to a `remote` backend, so a RunPod pod running vLLM/Ollama serving Gemma 4 satisfies ask/report immediately with zero VisionBrain changes. Getting Falcon + SAM 3.1 onto the same pod means writing PyTorch inference wrappers (or driving the upstream repos directly) behind a similar HTTP seam.
- Weight sizes for a CUDA layout (estimates): Gemma 4 26B bf16 ≈ 52 GB (26B × 2 bytes); SAM 3.1 bf16 ≈ 5 GB (~2.3B params); Falcon ≈ 4–8 GB (estimate — small perception model). **Total ≈ 60–65 GB of weights; budget a 100 GB volume.**

## 5. Startup overhead and weight persistence

- **Pod boot itself is fast**: container start + GPU attach is on the order of a minute for a cached image (estimate — RunPod does not publish pod boot times; its FlashBoot ~1 s cold-start claims apply to *serverless* endpoints, [RunPod FlashBoot blog](https://www.runpod.io/blog/introducing-flashboot-serverless-cold-start), fetched 2026-09-05). Budget **2–5 minutes** from "start pod" to shell.
- **Model loading dominates cold start.** RunPod's own numbers: loading a 7B model from a network volume takes ~15 s, versus multi-minute HF downloads at startup ([FlashBoot blog](https://www.runpod.io/blog/serverless-gpu-cold-starts-flashboot)); for a 30 GB model, "vLLM initialization alone can top five minutes" ([vLLM cold-start blog](https://www.runpod.io/blog/cut-vllm-cold-starts-runpod-serverless)); a Discord datapoint records a ~19-minute HF download for a large model ([answeroverflow thread](https://answeroverflow.com/m/1242368715266461757)). **Estimate for our ~60–65 GB of weights: 10–30 min to download fresh from HF vs ~2–5 min to load from a warm network volume.**
- **Persistence strategies:**
  1. **Network volume (recommended)**: attach at pod creation, set `HF_HOME`/`HF_HUB_CACHE` (and any model dirs) to `/workspace`, download once, terminate the pod afterward. Survives everything; $7/mo at 100 GB.
  2. **Custom Docker template with baked-in weights**: builds the weights into the image — fastest warm start, but a 60–65 GB image is slow to build/pull and you pay container-disk $0.10/GB/mo while running (and image pulls eat into session time). Better suited to ≤50 GB of essential weights.
  3. **Stopped pod with local volume disk**: works, but bills $0.20/GB/mo stopped — nearly 3x the network volume rate for worse durability (tied to one host).
- **Account guardrails**: balance at $0 stops pods (network-volume data survives; other pods' data does not) ([docs pricing page](https://docs.runpod.io/pods/pricing)).

## 6. GPU sizing for VisionBrain

| Component | bf16 VRAM (est.) | Notes |
|---|---|---|
| Gemma 4 26B | ~52 GB weights + KV cache → ~58–62 GB | arithmetic: 26B × 2 bytes; the dominant consumer |
| SAM 3.1 | ~10–14 GB working set | runs inside 16 GB unified memory on the Mac today |
| Falcon Perception | <8 GB | smallest of the three |
| **All three, bf16** | **~70–80 GB** | plus CUDA/framework overhead |

**Recommended sweet spot: RTX Pro 6000 Blackwell, 96 GB — $2.09/hr Secure ($1.69/hr Community).** The cheapest GPU on the list that holds all three models in bf16 *simultaneously* with headroom for KV cache, masks, and the demo UI. (The H100 NVL at 94 GB/$3.19 and H100 SXM 80 GB/$3.29 cost 50%+ more for no VRAM advantage; the A100 80 GB at $1.59 is viable only if Gemma runs quantized or the other two are idle.)

**Budget alternative: L40S, 48 GB — $0.79/hr Community ($0.99/hr Secure)**, running Gemma 4 26B in **int4 (~14 GB, est.) or int8 (~27 GB, est.)** with SAM 3.1 + Falcon in bf16 — everything co-resident, quantized-Gemma quality. The absolute cheapest 48 GB option is the older RTX A6000 at $0.33/$0.53/hr if Ampere performance is acceptable for a demo.

**Not recommended**: 24 GB cards (RTX 4090/3090) — Gemma 4 26B at int4 plus anything else is too tight for a reliable "all three at once" demo; 141–288 GB cards (H200/B200/B300) are overkill at 2–4x the price.

## 7. Worked cost scenarios

All arithmetic uses the **RTX Pro 6000 96 GB, Secure Cloud, $2.09/hr**, per-second billing, 100 GB standard network volume at $0.07/GB/mo. Storage prorated per second/minute is cents-level in every scenario.

### (a) One 1-hour demo, cold pod, fresh weights download
| Item | Math | Cost |
|---|---|---|
| GPU (1 h recording + ~15 min download/setup) | 1.25 h × $2.09 | $2.61 |
| Container/volume disk while running | 50 GB × $0.10/GB/mo × (1.25/730 mo) | ~$0.01 |
| Egress of recorded footage | free | $0 |
| **Total** | | **≈ $2.60** (range ~$2.4–3.7 if the download takes 5–30 min) |

### (b) One 1-hour demo, weights on a persistent 100 GB network volume
| Item | Math | Cost |
|---|---|---|
| GPU (1 h + ~5 min boot/load) | 1.08 h × $2.09 | $2.26 |
| Network volume carrying cost | 100 GB × $0.07/GB/mo | $7.00/**month** (≈ $0.0008 during the session itself) |
| **Marginal session cost** | | **≈ $2.30** |

### (c) Monthly cost at 4 and 12 sessions
| Scenario | Math | Total/mo |
|---|---|---|
| 4 sessions/mo, warm (volume) | 4 × $2.26 + $7.00 | **≈ $16/mo** |
| 12 sessions/mo, warm (volume) | 12 × $2.26 + $7.00 | **≈ $34/mo** |
| 12 sessions/mo, cold (no volume) | 12 × $2.61 | ≈ $31/mo — but each session re-downloads 60+ GB and adds 10–30 min of variance; the volume is worth it |
| Same GPU count on A100 80 GB PCIe ($1.39 Secure) for reference | 4 × (1.08 × $1.39) + $7 | ≈ $13/mo (only if Gemma is quantized) |

### (d) Pod created-but-stopped all month ("warm" option)
| Strategy | Math | Cost/mo |
|---|---|---|
| Stopped pod, 100 GB **volume disk** kept | 100 GB × $0.20/GB/mo (stopped rate) | **$20.00/mo** |
| Stopped pod, 100 GB **network volume** | $7.00/mo (network volumes bill identically regardless of pod state) | **$7.00/mo** |
| **Terminated pod + 100 GB network volume** (recommended) | $7.00/mo; next session boots fresh in ~2–5 min from `/workspace` | **$7.00/mo** |

A stopped pod buys you ~3 minutes of boot time over a terminated pod with a network volume, at $0–13/mo extra. Keep a setup script or Docker template instead of a stopped pod. One caveat: a *stopped* pod restarts onto its original configuration; a *terminated* pod is recreated, so script the config (GPU type, volume mount, env vars) once.

## 8. Alternatives sanity check (secondary sources — market prices fluctuate)

For the same workload class (roughly 48–96 GB single GPU, per-second billing):

| Provider | GPU | Price | Billing | Notes |
|---|---|---|---|---|
| **RunPod** (primary, fetched 2026-09-05) | L40S 48 GB | $0.99/hr Secure / $0.79 Community | per-second | no egress fees; NV storage $0.07/GB/mo |
| **RunPod** | RTX Pro 6000 96 GB | $2.09/hr Secure | per-second | recommended SKU above |
| **Vast.ai** ([pricing page](https://vast.ai/pricing), fetched 2026-09-05) | RTX 4090 / RTX 5090 | live marketplace: 4090 from ~$0.13, **median ~$0.37/hr**; 5090 median ~$0.45/hr | per-second, "no minimum hours" | cheaper, but rates/host quality vary across 40+ datacenters; interruptible "50%+ cheaper"; storage/egress terms not published on the page |
| **Lambda** ([lambda.ai/pricing](https://lambda.ai/pricing), fetched 2026-09-05) | H100 SXM 80 GB (1x) | $4.29/hr | hourly | no consumer GPUs (no 4090/5090/L40S listed); storage pricing not published |
| **Lambda** | A100 PCIe 40 GB (1x) | $1.99/hr | hourly | 40 GB won't fit the workload bf16 |
| **AWS EC2** (secondary: [Vantage](https://instances.vantage.sh/aws/ec2/g6e.12xlarge), fetched 2026-09-05) | g6e.xlarge (1x L40S 48 GB) | ~$1.86/hr on-demand, us-east-1 | per-second (Linux), spot lower (est.) | EBS gp3 ≈ $0.08/GB-mo and internet egress ≈ $0.09/GB after 100 GB/mo free (estimates — not verified from AWS directly today); much heavier setup than RunPod |

**Verdict:** RunPod is competitive to cheap for this use case — its L40S is roughly half AWS's equivalent, and it's the only one of the three with simple persistent network-volume storage designed for exactly this spin-up/spin-down pattern. Vast.ai can undercut it on consumer cards but with marketplace variance that a timed demo session doesn't want; Lambda is priced for sustained production, not minutes-long sessions.

## 9. Sources (all fetched 2026-09-05)

Primary (RunPod):
- https://www.runpod.io/pricing — GPU table (Community + Secure), serverless, storage, per-second billing toggle
- https://docs.runpod.io/pods/pricing — per-second billing, credit floor, no egress fees, stopped-pod storage, spend limit
- https://docs.runpod.io/pods/storage — network volumes ($0.07/GB/mo standard, $0.14 HP), persistence, `/workspace` mount
- https://www.runpod.io/blog/spot-vs-on-demand-instances-runpod — spot ~50% discount, A6000 $0.232 vs $0.491 example (post updated 2025-08-25)
- https://www.runpod.io/blog/serverless-gpu-cold-starts-flashboot — 7B-from-network-volume ≈ 15 s
- https://www.runpod.io/blog/cut-vllm-cold-starts-runpod-serverless — 30 GB vLLM init > 5 min
- https://www.runpod.io/blog/introducing-flashboot-serverless-cold-start — FlashBoot 500 ms–1 s (serverless only)

Secondary (cross-checks, labeled as such in text):
- https://www.hivenet.com/post/runpod-pricing-complete-guide-to-gpu-cloud-costs — sample of RunPod's page taken 2026-08-20 (matches Secure Cloud column)
- https://lambda.ai/pricing — Lambda on-demand rates
- https://vast.ai/pricing — Vast.ai billing model + live median rates
- https://instances.vantage.sh/aws/ec2/g6e.12xlarge — AWS g6e pricing
- https://answeroverflow.com/m/1242368715266461757 — real-world ~19-min HF download datapoint

**Not verified from primary sources / estimates in this doc:** pod boot times for interactive pods (no published figure), CUDA VRAM footprints for SAM 3.1 and Falcon (inferred from Mac unified-memory behavior), Gemma 4 quantized sizes (arithmetic), Vast.ai storage/egress terms and AWS EBS/egress rates, and all "range" figures around download durations. Re-check runpod.io/pricing before each billing-relevant decision.
