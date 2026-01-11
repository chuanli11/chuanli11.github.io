---

layout: "post"
title: "Vera Rubin NVL72: What’s New and Why It Matters"
date: 2026-01-11 10:00:00
toc: true
---

**Contents**
* TOC
{:toc}

Vera Rubin NVL72 represents the latest progress in rack-scale AI infrastructure. This write-up walks through the key architectural innovations, from Rubin GPUs and Vera CPUs to NVLink 6, ConnectX-9, Spectrum-X, and BlueField-4, and breaks down which changes drive how much improvement for what workloads. 

The numeric claims are, in my best effort, labeled into three categories: 

- [Spec]: directly from NVIDIA product spec, or computed ratio from those specs
- [NVIDIA claim] = NVIDIA’s marketing/benchmark claim that might involve specific optimization
- [Hypothesis] = my intuition


## TL;DR: Six New Chips And What We Gain from Them

- **Rubin GPU** + **HBM4 GPU memory**
    - [NVIDIA Claim] `1.6-5x` FLOPS increase, benefits per‑GPU throughput.
    - [Spec] `2.74×` higher GPU memory bandwidth, benefits workloads that have low compute-to-memory access ratio. For example, memory-bound kernels (element-wise operations), decoding with large KV footprints etc.
- **Vera CPU**
    - [Spec] `2x` higher CPU-GPU bandwidth, benefits KV-cache offload.
    - [Spec] `3x` CPU memory capacity, benefits large datasets caching.
- **NVLink 6 Switch**
    - [Spec] `2x` intra-rack GPU to GPU communication bandwidth for scaling-up, benefits all reduce operations e.g. MoE all-to-all, model parallelism.
- **ConnectX-9** **SuperNIC**
    - [Spec] `2x` inter-rack GPU to GPU communication bandwidth for scaling-out, benefits operations across different racks.
- **Spectrum‑X Photonics (Co-packaged Optics)**
    - [NVIDIA Claim] `5×` better power efficiency, [NVIDIA Claim] `10×` higher resiliency for scale-out network, reducing cost & reliability penalty of large GPU fabrics.
- **BlueField‑4**
    - [Spec] `2x` Data‑in‑transit encryption, benefits multi-tenant, zero-trust security.
    - [Spec] `3.3x` memory bandwidth, benefits checkpoint bursts and dataset streaming. Together with Spectrum-X, BF-4 creates an extra tier of “warm” KV cache, benefits long-context and agentic workloads.

## Breaking Down the Gains: What’s Different and Why It Matters


**Rubin GPU & NVLink 6 Switch & ConnectX-9 SuperNIC**

- [Spec] `1.6x` [more transistors](https://www.hpcwire.com/aiwire/2026/01/06/nvidia-says-rubin-will-deliver-5x-ai-inference-boost-over-blackwell/) v.s. Blackwell ultra. So by default I expect [Hypothesis] `~1.6x` higher FLOPS for my everyday GPU workloads.
- Rack-level GPU memory bandwidth sees a [Spec] `2.74×` increase, from GB300 NVL72’s 576 TB/s to VR NVL72’s 1580 TB/s. This comes from the HBM3E to HBM4 upgrade.
- Per GPU-GPU bandwidth inside of a rack sees a [Spec] `2x` increase, from 1.8 TB/s to 3.6 TB/s. This comes from NVLink 6 Switch, which [doubles](https://www.nvidia.com/en-us/data-center/nvlink/?ncid=pa-srch-goog-156-prsp-rsa-en-us-1-l1) the number of links per GPU from 18 to 36.
- Per GPU-GPU bandwidth between different racks sees a [Spec] `2x` increase, from 800 Gb/s to 1.6 Tb/s. This comes from the doubling of per-rack OSFP ports and ConnectX-9 NIC, both from [72](https://www.nvidia.com/en-us/data-center/dgx-gb300/) to [144](https://www.notion.so/Vera-Rubin-NVL72-What-s-New-and-Why-It-Matters-2e4b856e4027801d85dfdb1cbe92ba82?pvs=21).

My takes: Compute (FLOPS) & communication (bandwidth) are the bread and butter of today’s rack-level system performance. Vera Rubin NVL72 is likely to be [Hypothesis] `1.5-2x` faster than GB300 NVL72 for most of the “out-of-box” workloads, depends on whether compute or communication is the bottleneck. There are some specific use cases where the performance boost can be higher. For example:

- NVFP4 workloads optimized by NVIDIA’s [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) can potentially give [NVIDIA Claim] `3.5x` speedup for training and [NVIDIA Claim] `5x` speedup for inference. However, [no official breakdown of these numbers](https://github.com/NVIDIA/TransformerEngine/issues/2565) from TransformerEngine. Also NVIDIA could use non-ultra Blackwell GPUs as the baseline here.
- Memory bandwidth deltas matters significantly to workloads that have low compute-to-memory access ratio. For example, memory-bound kernels (layernorm/softmax/element-wise activation functions), decoding with large KV footprints etc.
- NVLink and ConnectX-9’s uplift to GPU-GPU communication is particularly beneficial to large distributed workloads. For example, models that need non-trivial parallelism, MoE models that needs token routing and “all-to-all” expert communication.

**Vera CPU**

- NVLink-C2C bidirectional bandwidth [Spec] `2x` to Vera-Rubin’s [1.8 TB/s](https://www.nvidia.com/en-us/data-center/vera-cpu/), from Grace-Hopper’s [900 GB/s](https://www.nvidia.com/en-us/data-center/grace-cpu/?ncid=pa-srch-goog-156-prsp-rsa-en-us-1-l1).
- Per rack CPU memory capacity [Spec] `3x` to 54 TB, from GB300 NVL72’s 17 TB.
- [Spec] `+22.2%` more CPU cores per rack. No official information on what core speed does Vera’s [88 Olympus cores](https://www.nvidia.com/en-us/data-center/vera-cpu/) offer. But we do know Grace CPU has [72 Neoverse™ V2 cores](https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/?ncid=pa-srch-goog-156-prsp-rsa-en-us-1-l1) each runs at [3.1GHz](https://nvdam.widen.net/s/rrgqqnpbz8/grace-datasheet-gh200-grace-hopper-superchip-3773000%20).

My takes: Rack-scale systems like NVL72 don’t use CPU as the primary compute engine (GPUs handle matrix ops). CPU still matters to peripheral workloads. For example:

- KV-cache offload can benefit from the upgraded NVLink-C2C bandwidth.
- Caching large datasets are easier because of the larger CPU memory.
- Data preprocessing, tokenization, inference scheduling, kernel launch etc all become more parallelizable because of the increased number of cores.

**BlueField-4**

- Upgrade from BF-3 on all fronts: compute cores [Spec] `4x` to 64 Arm Neoverse V2, [Spec] `4x` memory capacity to 128GB, [Spec] `3.3x` memory bandwidth to 250 GB/s, NVMe storage disaggregation `2x` to 20M at 4K IOPs, Data‑in‑transit encryption [Spec] `2x` to 800 Gb/s,

My takes: infrastructure services (networking, storage, telemetry, security) become a bottleneck at AI-factory scale if they run on host CPUs; BF4 moves more of that off-host for more deterministic behavior and better utilization.

- Increased number of Arm Neoverse V2 cores and memory capacity means BlueField-4 can run more & heavier services in parallel and improve overall off-host performance.
- Increased memory bandwidth is particularly useful for checkpoint bursts and dataset streaming.
- Fast disaggregated storage also improves applications that require massive parallel random I/Os on network storage, such as retrieval-augmented generation, data shuffling, recommendation systems etc.
- Encrypt and decrypt traffic at line rate (up to 800 Gb/s) is crucial for multi-tenant, hyperscale AI setups requiring **zero-trust security** and compliance while moving huge datasets, model, and results.

**Scale-out: Spectrum-X™ Ethernet, Quantum-X InfiniBand**

- Co-packaged Optics switch integrates silicon photonics with the ASIC, instead of attached them as external modules. This leads to [NVIDIA Claim] `5×` better power efficiency (measured by network energy per delivered byte), [NVIDIA Claim] `10×` higher network resiliency, and up to [NVIDIA Claim] `5×` more uptime.
- To scale-out, Vera Rubin NVL72 system offers both Spectrum‑X Ethernet and NVIDIA Quantum‑X InfiniBand at the same bandwidth (two 800 Gb/s links per GPU, via the [Connect-X 9 SuperNIC](https://docs.nvidia.com/networking/display/connectx9supernic/supported-interfaces)). How to choose between Spectrum‑X Ethernet and NVIDIA Quantum‑X InfiniBand is a trade‑off between operational familiarity, ecosystem and performance on specific AI workloads.

My takes: Integrated optics reduces the number of independent components and shortens signal paths, which leads to lower per-bit power consumption and fewer possible failure points. This significantly reduces the network power + reliability penalty of scaling to very large GPU fabrics.

- More stable performance under bursty traffic (low jitter during large scale gradient synchronization, Figure 1, [Spectrum-X blog](https://developer.nvidia.com/blog/scaling-power-efficient-ai-factories-with-nvidia-spectrum-x-ethernet-photonics/)), fewer failures during long‑running training and inference jobs.
- For inference workload with long context, Spectrum-X attaches Flash/SSD memory directly to BlueField-4, create [Inference Context Memory Storage](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/) (ICMS) as an extra tier of KV caching that bridges the gap between “hot/warm” memory (HBM, DRAM, local/rack-attached storage) and “cold” memory (network storage). There is no evidence on if Quantum-X InfiniBand can be used as an alternative to Spectrum-X™ Ethernet in this use case.

## Overall Power Efficiency

**Power smoothing for spike usage**

- coordinating power draw across GPUs, CPUs, networking, and system electronics inside the rack, so that short spikes don’t appear as large peaks at the facility level. This enables up to [NVIDIA Claim] [30% more](https://www.notion.so/Vera-Rubin-NVL72-What-s-New-and-Why-It-Matters-2e4b856e4027801d85dfdb1cbe92ba82?pvs=21) compute provisioning within the same power envelope.
- This is particularly helpful for accommodating synchronized all-to-all communication during large-scale training, and bursty inference demands.
- NVIDIA achieves this by reshaping spiky power swings into controlled ramps bounded by a stable power ceiling and floor, and use a local energy buffering (no official details on the specific components) to absorb the demand. In particular, Vera Rubin NVL72 has [NVIDIA Claim] [~6× more](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) local energy buffering than Blackwell Ultra.
- With new liquid-cooled design, Vera Rubin NVL72 nearly [NVIDIA Claim] `2x` its thermal performance in the same rack footprint.

**Improved Tokens Economics**

Compared to Blackwell NVL72

- [NVIDIA Claim] `4×` fewer GPUs to train a 10T MoE model on 100T tokens in 1 month. (*Figure 37,* [NVIDIA Rubin Platform blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)*)*
- [NVIDIA Claim] `10x` higher token throughput per megawatt, at ~210 TPS (Tokens Per Second), for Kimi‑K2‑Thinking 1T MoE, 32K input / 8K output. (*Figure 38,* [NVIDIA Rubin Platform blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)*)*
- [NVIDIA Claim] `5x` higher sustained TPS for long-context and agentic workloads. Because Dynamo KV block managers +  [ICMS](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/) (with BlueField‑4 + Spectrum-X) are able to prestage context and reducing decoder stalls, preventing GPUs from “redundant recomputation of history” and “wasting energy on idle cycles”.

## What Stays The Same

- Per GPU memory capacity stays the same at 288GB per GPU.
- 800 Gb/s bandwidth for Quantum-X800 InfiniBand, Spectrum-X Ethernet, and ConnectX-9 SuperNIC stays the same. But for Vera Rubin NVL72, each GPU will have two ConnectX-9 ports so the per GPU inter-rack bandwidth doubles compared to Blackwell NVL72.