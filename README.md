# llama.cpp: tensor parallelism over USB4 RDMA

> [!IMPORTANT]
> This is an experimental feature at the proof-of-concept stage. It runs on Windows only, has been
> tested on a small number of machines and models, and its options and log messages may change.
> The ranks exchange data over TCP and NetworkDirect with no authentication or encryption.
> **Only run it on machines joined by their own Thunderbolt/USB4 cables, on a network you trust. Never run it
> on an open network.**

## Overview

This repository is llama.cpp with distributed tensor parallelism added. One model is split across
2 to 4 Windows machines joined by Thunderbolt/USB4 cables, and all of them work on every token.

Each machine runs its own `llama-cli` process, called a rank. Each rank holds a slice of every
layer: its share of the weights, the attention heads and the KV cache. After the attention and
feed-forward blocks of each layer, the ranks add up their partial results (an all-reduce) over the
USB4 links using Microsoft NetworkDirect (RDMA) provider APIs. Every rank then continues with the full result.

This lets you:

- run a model that does not fit in one machine's memory, since each machine holds about 1/N of
  the weights;
- use the GPU of every machine on every token, instead of one machine at a time as with a split by
  layers.

## What is added

| Area | Change |
|---|---|
| Transport | All-reduce over NetworkDirect (RDMA) on the Thunderbolt/USB4 links, sending in both directions at once. Messages of any size go over RDMA in 256 KiB chunks, the prompt included. The same can run over TCP on the same cables for comparison. |
| All-reduce | A ring. At 3 ranks, and at 4 ranks with a cable between every pair, small messages (decode) go to all peers at once (full mesh). |
| Model split | `--split-mode tensor` splits every layer across the ranks. Attention is split by whole KV groups, or by query heads when the KV heads do not divide evenly. Qwen3.5 gated DeltaNet layers are split too. |
| Wire format | Partial results go on the wire as f16 by default, with the sums kept in f32. f32 on the wire is optional. |
| Vulkan on Intel Xe2/Xe3 | A Q2_K/Q3_K decode matrix-vector kernel, a Q8_0 flash-attention decode kernel, and recording the next part of the graph while an all-reduce is on the network. |
| Tools | Per-graph cost lines (`LLAMA_DISTPROF=1`), an HTML trace of the graph steps (`GGML_TRACE=1`), and PowerShell scripts in `usb4rdma/scripts/` that prepare the links, launch every rank and check which transport each rank used. |

## Requirements

- 2 to 4 Intel x64 systems running Windows 11, each with an Intel Graphics Processing Unit (GPU) and its latest graphics driver installed on the system (Vulkan backend is used for this fork).
- One Thunderbolt/USB4 cable between each pair of machines that are next to each other in the ring: 1 cable
  for 2 machines, 3 for 3, 4 for 4 (6 for the 4-machine full mesh).
- Make sure USB4 Peer-to-Peer Network driver is correctly enumerated on each system with a NetworkDirect provider (ndprov.dll) installed on every machine. [Windows driver and network direct providers not
  included here. Without it, the ranks run over TCP on the same cables].
- The same model file, prompt and binaries at the same path on every machine.
- To build: Git, Visual Studio 2022 ("Desktop development with C++") and the Vulkan SDK.

## Build

```bat
git clone -c core.longpaths=true https://github.com/swathiananth/llama_cpp_upstream
cd llama_cpp_upstream
usb4rdma\build.cmd
```

`build.cmd` fetches the public Microsoft NetworkDirect SDK, configures with `-DGGML_VULKAN=ON
-DLLAMA_DIST_TP_ND=ON` and builds into `build-usb4rdma\build\bin`. The configure step must print
`llama: NetworkDirect dist-TP transport enabled`. A build without that line runs over TCP only.
Copy `build-usb4rdma\build\bin` to the same folder on every machine.

## Run

Give each end of each USB4 cable a static address on a subnet used only by that cable. In the
firewall, allow inbound TCP from the rendezvous port up to that port + N - 1 (29500 by default),
and TCP 23517-23518. The details are in [usb4rdma/README.md](usb4rdma/README.md).

Two machines, with rank 0 at `192.168.77.1` on the cable. On each machine, in `cmd.exe` in the
binary folder:

```bat
llama-cli -m <model.gguf> -f <prompt.txt> -n 64 -c 2048 -ngl 99 --device Vulkan0 ^
  --split-mode tensor --world-size 2 --world-rank <0 or 1> --dist-master 192.168.77.1:29500
```

Start the highest rank first and rank 0 last. Add `--no-jinja` for gpt-oss. For 3 or 4 machines,
see [usb4rdma/README.md](usb4rdma/README.md) for cabling and ring order. The scripts in
`usb4rdma/scripts/` do all of this from a separate controller machine.

| Option | Meaning |
|---|---|
| `--split-mode tensor` | split every layer across the ranks |
| `--world-size N` | number of ranks, 2 to 4 |
| `--world-rank r` | this machine's rank, 0 to N-1 |
| `--dist-master HOST:PORT` | rank 0's address and rendezvous port, the same on every rank |
| `--dist-listen-port PORT` | optional, default is the master port + rank |

## Check that it ran on RDMA

A run that fell back to TCP still finishes and prints a plausible speed, so check every rank's log:

- `dist-tp transport: RDMA (ND)`, with `ND` on every edge and `(all payloads over ND)`;
- no `falling back to TCP`, `staying on TCP` or `NetworkDirect not established`;
- every `ring edge ... dials <address>` is a USB4 address.

To check that the output is correct, run once with `LLAMA_DIST_TP_WIRE_F32=1` and compare the text
with a single-machine run of the same model, prompt and seed (`--temp 0 --seed 1234`). The two
should be the same.

## Models

| Model | Notes |
|---|---|
| gpt-oss-120b | needs `--no-jinja` |
| Qwen3-235B-A22B | on 3 ranks, attention is split by query heads |
| Qwen3.5 | gated DeltaNet layers are split across the ranks |
| Qwen3-Next | loads, but its output has not been verified |
| others | not every architecture splits correctly; compare with a single machine first |

## Limitations

- Windows only. Tested with `llama-cli`.
- 2 to 4 ranks, one GPU per rank.
- If a cable or a rank fails during a run, the run stops or stalls. There is no recovery.
- On integrated GPUs, load one large model per boot. A second load in the same boot can run much
  slower.
- With the default f16 wire format, the text can drift slightly from a single-machine run after
  some tokens.

## More

[usb4rdma/README.md](usb4rdma/README.md) has the full setup (addresses, firewall, SSH, ring order),
the scripts, all environment variables and the pitfalls.

This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) at commit 91f8c9c5, under the
same MIT license. The upstream llama.cpp README follows.

---
