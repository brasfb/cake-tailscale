# Next Steps: Bringing Up Your Mac Mini LLM Cluster

A step-by-step guide for deploying this fork on a cluster of 2014 Intel Mac
Minis over Tailscale. No prior clustering experience assumed — every step
includes the exact commands and what success looks like.

**What you built (the 30-second version):** each mini runs a *worker* that
holds a slice of the model's layers in RAM. One machine (the *master*) holds
the remaining layers plus the input/output ends of the model and exposes a
chat API. Every generated token flows through all machines over your
Tailscale network. Clients (curl, Open WebUI, ollama apps) talk only to the
master.

```
 you / Open WebUI
        │  http://master:8080  (/v1/chat/completions or /api/chat)
        ▼
   ┌─────────┐    layers 21-27 + embeddings + lm_head
   │ master  │
   └────┬────┘
        │ Tailscale (TCP 10128)
   ┌────┴────┬──────────┐
   ▼         ▼          ▼
 mini2     mini3      mini4
 layers    layers     layers
 0-6       7-13       14-20
```

---

## Phase 0 — Inventory (15 minutes)

Do this once, on each mini. Write the answers down — you'll need them for
the topology file.

1. **Check macOS version** (must be 10.13+; Monterey 12.x is ideal):
   click  → About This Mac.
2. **Check RAM** — this decides how many layers each mini gets:
   ```sh
   sysctl -n hw.memsize
   # 8589934592  = 8 GB
   # 17179869184 = 16 GB
   ```
3. **Enable SSH**: System Preferences → Sharing → check **Remote Login**.
4. **Give each mini a useful name**: System Preferences → Sharing → Computer
   Name → `mini2`, `mini3`, `mini4` (the master can be `mini1` or your main
   computer).

> **Which machine should be master?** The one with the most RAM, because it
> holds the embedding + lm_head on top of its layer share. If you have a more
> powerful always-on machine (a newer Mac, a Linux box), it can be the master
> instead — masters and workers don't need to be the same kind of machine.

## Phase 1 — Install the toolchain on every mini (30 min, mostly waiting)

SSH in from your main computer (`ssh you@mini2.local` works while you're on
the same LAN) or work at the machine directly.

1. **Xcode Command Line Tools** (compiler):
   ```sh
   xcode-select --install
   ```
   Click "Install" in the popup. Verify: `cc --version` prints something.

2. **Rust**:
   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
   Accept the defaults, then `source ~/.cargo/env`.
   Verify: `cargo --version` prints something like `cargo 1.8x`.

3. **Tailscale**: download from <https://tailscale.com/download/macos>
   (App Store version is fine), open it, log in to your tailnet.
   Verify and **record the name/IP** for the topology file:
   ```sh
   /Applications/Tailscale.app/Contents/MacOS/Tailscale status
   ```
   Optional but recommended — put the CLI on your PATH:
   ```sh
   sudo ln -s /Applications/Tailscale.app/Contents/MacOS/Tailscale /usr/local/bin/tailscale
   ```

4. **Stop the minis from sleeping** (a sleeping worker = a dead cluster):
   ```sh
   sudo pmset -a sleep 0 disksleep 0
   ```

## Phase 2 — Build cake on every machine (60-90 min on a 2014 mini)

```sh
git clone https://github.com/brasfb/cake-tailscale.git ~/cake
cd ~/cake
git checkout claude/mac-mini-llm-cluster-sb2m8t
./scripts/build-intel-mac.sh
```

The script builds CPU-only with `-C target-cpu=native` (enables AVX2 on the
minis' Haswell CPUs). **Expect this first build to take an hour or more** on
a 2014 dual-core — start it on all minis in parallel and go do something
else. Rebuilds after `git pull` are much faster.

Verify: `./target/release/cake --help` prints the command list.

> **If the build fails:** this branch was developed and tested on Linux; the
> Intel-macOS build is the one thing that couldn't be verified beforehand.
> Copy the full error message and paste it into a Claude Code session on this
> repo — most likely fixes are small (a platform-gated dependency or API).

## Phase 3 — Download a model on every machine (10 min)

Start small. Qwen3-0.6B (~1.5 GB) proves the pipeline before you commit to
bigger downloads:

```sh
cd ~/cake
./target/release/cake pull evilsocket/Qwen3-0.6B
./target/release/cake list   # should show: evilsocket/Qwen3-0.6B  complete
```

Every machine needs the model (each loads only its own layers from it, but
from a full local copy — this avoids streaming 1.5 GB over the network on
every start).

> **If `cake pull` fails with a TLS/certificate error** (some networks with
> HTTPS inspection): download with Python instead, into the same cache:
> ```sh
> pip3 install -U huggingface_hub
> python3 -m huggingface_hub.commands.huggingface_cli \
>     download evilsocket/Qwen3-0.6B
> ```

Good models for this hardware once the small one works
(sized by **F16 footprint** — quantized files are expanded to F16 in RAM):

| Model | Layers | F16 size | Fits on | Feel |
|---|---|---|---|---|
| `evilsocket/Qwen3-0.6B` | 28 | 1.5 GB | one machine (use the cluster anyway, to test it) | fast, basic |
| `meta-llama/Llama-3.2-1B-Instruct` | 16 | 2.5 GB | 2+ machines | fast, decent |
| `Qwen/Qwen3-1.7B` | 28 | 3.4 GB | 2+ machines | good balance |
| `meta-llama/Llama-3.2-3B-Instruct` | 28 | 6.4 GB | 3-4 machines | good, slower |
| `Qwen/Qwen3-4B` | 36 | 8 GB | 4 machines | best quality, ~1-2 tok/s |
| `ibm-granite/granite-3.3-2b-instruct` | 40 | 5 GB | 3-4 machines | new in this fork — see Phase 7 |

## Phase 4 — Write the topology file (15 min)

The topology says who serves which layers. Edit `topology-minis.yml` on the
**master** machine (only the master strictly needs it, but keeping it
committed in git and pulled everywhere avoids confusion).

Rules of thumb:
- The file lists **workers only**. The master automatically keeps every layer
  not assigned to a worker — give the master a few less if it also runs the
  embedding/lm_head (it does) or has less RAM.
- Layer indices run from `0` to `num_layers - 1`. Splitting a 28-layer model
  across 3 workers + master: `0-6`, `7-13`, `14-20`, master keeps `21-27`.
- Use Tailscale MagicDNS names (`mini2.your-tailnet.ts.net`) or `100.x.y.z`
  addresses from `tailscale status`. **Never `.local` names** — they don't
  resolve across a tailnet.

Example for Qwen3-1.7B (28 layers), 3 workers, 8 GB each:

```yaml
mini2:
  host: 'mini2.tail1234.ts.net:10128'
  description: 'Mac Mini 2014 (CPU)'
  backend: 'CPU'
  vram_bytes: 8589934592
  layers:
    - 'model.layers.0-6'
mini3:
  host: 'mini3.tail1234.ts.net:10128'
  description: 'Mac Mini 2014 (CPU)'
  backend: 'CPU'
  vram_bytes: 8589934592
  layers:
    - 'model.layers.7-13'
mini4:
  host: 'mini4.tail1234.ts.net:10128'
  description: 'Mac Mini 2014 (CPU)'
  backend: 'CPU'
  vram_bytes: 8589934592
  layers:
    - 'model.layers.14-20'
```

## Phase 5 — First launch (the moment of truth)

Pick a cluster key (any secret string — it authenticates cluster members to
each other). Use the same key everywhere.

**Start each worker** (on mini2/3/4 — note: no model name before the flags;
that's what makes it a worker):

```sh
cd ~/cake
./target/release/cake run --cluster-key mysecret \
    --model evilsocket/Qwen3-0.6B --name mini2 \
    --topology topology-minis.yml --address 0.0.0.0:10128
```

Success looks like:
```
[Worker] dtype=F16 device=Cpu ...
loading model.layers.0 ...
...
listening on 0.0.0.0:10128 (mem:662.4 MiB) ...
```

(`scripts/start-minis.sh` automates this over SSH once the manual version
works: `CAKE_CLUSTER_KEY=mysecret MODEL=evilsocket/Qwen3-0.6B MINIS="mini2 mini3 mini4" ./scripts/start-minis.sh`)

**Start the master** (after all workers say "listening"):

```sh
cd ~/cake
./target/release/cake serve evilsocket/Qwen3-0.6B \
    --cluster-key mysecret --topology topology-minis.yml --api 0.0.0.0:8080
```

Success looks like: each worker logged a connection, and the master ends with
`starting service: "actix-web-service-0.0.0.0:8080"`.

**Test it** — from any machine on your tailnet (replace the host):

```sh
# Is the API alive?
curl http://mini1.tail1234.ts.net:8080/api/version

# Does the whole cluster generate? (first request also warms things up)
curl http://mini1.tail1234.ts.net:8080/api/chat \
  -d '{"messages":[{"role":"user","content":"What is 2+2? /no_think"}],"stream":false}'

# See how layers are distributed:
curl http://mini1.tail1234.ts.net:8080/api/v1/topology
```

A correct, coherent answer here means the entire pipeline works:
Tailscale → auth → layer sharding → distributed forward pass → API. 🎉

Also try the **web UI** built into the master: open
`http://mini1.tail1234.ts.net:8080/` in a browser.

## Phase 6 — Connect a real chat app

Anything that speaks OpenAI or ollama now works. Easiest: Open WebUI on
your main computer (needs Docker):

```sh
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://mini1.tail1234.ts.net:8080/v1 \
  -e OPENAI_API_KEY=unused \
  ghcr.io/open-webui/open-webui:main
```

Then open `http://localhost:3000`. Alternatively add it as an **Ollama
connection** pointing at `http://mini1.tail1234.ts.net:8080` — the
`/api/tags`, `/api/chat` endpoints added in this fork handle that.

The built-in TUI also works from any machine with the binary:
```sh
./target/release/cake chat --server http://mini1.tail1234.ts.net:8080
```

## Phase 7 — Try the new Granite support (validation welcome)

This fork added IBM Granite (3.x and the dense granite-4.0 nano models).
The code passes all offline tests, but real-model output hasn't been compared
against the reference implementation yet. You can be the validation:

```sh
./target/release/cake pull ibm-granite/granite-4.0-1b     # 2.2 GB
./target/release/cake run ibm-granite/granite-4.0-1b "Why is the sky blue? Answer in one sentence."
```

If the answer is coherent English → Granite works, enjoy. If it's word salad
→ the chat template or a multiplier needs adjusting; paste the output into a
Claude session on this repo.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Master: `can't connect to mini2...:10128: Connection refused` | Worker not running yet, or wrong host in topology. Start workers first; test with `nc -vz mini2.tail1234.ts.net 10128`. |
| Hostname doesn't resolve | You used `.local` (doesn't cross Tailscale) or MagicDNS is off. Use the `100.x.y.z` IP from `tailscale status` instead. |
| Worker log says `[Master]` instead of `[Worker]` | You passed the model as the first argument. Workers use `--model` as a flag and **no** positional model. |
| `authentication failed` in logs | Different `--cluster-key` on master vs worker. |
| Worker killed / machine freezes while loading | Out of RAM — that mini was assigned too many layers. Give it fewer in the topology, or use a smaller model. Leave ~2 GB headroom per machine. |
| Generation extremely slow (<0.5 tok/s) | Check `tailscale status` says the peers are connected `direct` and not `relay` ("DERP") — relayed traffic adds huge latency. Same physical LAN should always go direct. |
| `cake pull` TLS/certificate error | Use the `huggingface_hub` fallback in Phase 3. |
| Build fails on macOS | Paste the error into a Claude Code session on this repo. |

## Phase 8 — Make it permanent (optional)

- **Auto-start workers at boot**: create
  `~/Library/LaunchAgents/com.cake.worker.plist` on each mini:

  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
  <plist version="1.0"><dict>
    <key>Label</key><string>com.cake.worker</string>
    <key>ProgramArguments</key><array>
      <string>/Users/YOU/cake/target/release/cake</string>
      <string>run</string>
      <string>--cluster-key</string><string>mysecret</string>
      <string>--model</string><string>evilsocket/Qwen3-0.6B</string>
      <string>--name</string><string>mini2</string>
      <string>--topology</string><string>/Users/YOU/cake/topology-minis.yml</string>
      <string>--address</string><string>0.0.0.0:10128</string>
    </array>
    <key>WorkingDirectory</key><string>/Users/YOU/cake</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>/tmp/cake-worker.log</string>
    <key>StandardErrorPath</key><string>/tmp/cake-worker.log</string>
  </dict></plist>
  ```
  Load it with `launchctl load ~/Library/LaunchAgents/com.cake.worker.plist`.
  (Edit `YOU`, the name, the key, and the model.)

- **Updating the cluster** after code changes:
  `MINIS="mini2 mini3 mini4" ./scripts/deploy-minis.sh` pulls + rebuilds
  everywhere over SSH.

## Where this project goes next (roadmap)

In rough order of payoff:

1. **True quantized inference** — the big one. Today GGUF/quantized models
   are expanded to F16 in RAM, so quantization saves download, not memory.
   Keeping Q4/Q8 weights quantized in memory would roughly double-to-quadruple
   both the model size that fits and tokens/sec on these bandwidth-limited
   machines (Q4 7B ≈ 4 GB instead of 14 GB).
2. **`/api/pull` over HTTP** — let chat UIs trigger model downloads remotely,
   like real ollama.
3. **Granite-4.0-H (Mamba-2 hybrid)** and **Gemma 4** — documented stretch
   goals in `docs/models.md`; both are multi-week architecture projects and
   make the most sense *after* quantization lands.

## Getting help

When anything breaks, the fastest path is a Claude Code session on this
repository with: (1) the exact command you ran, (2) the full error/log
output, (3) which machine it happened on. The deploy scripts, topology
format, and API endpoints in this guide are all covered by tests, so most
failures will be environment-specific and quickly fixable.
