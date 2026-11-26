# metallm

A UEFI app that boots from a USB stick and runs an LLM chat on the CPU, with no operating
system. Models from 0.6B to 27B parameters; you put the ones you want on the stick and pick one
at boot.

## Requirements

- An x86-64 PC or Intel Mac that boots UEFI with a graphics framebuffer (GOP).
- AVX2 (Intel Haswell / AMD Zen and newer) for usable speed. Without it a plain C path runs,
  many times slower. AVX-512 and AVX-VNNI are used when present.
- Secure Boot off, or set up the stick with your own signing key so it boots with Secure Boot on
  (see [Secure Boot](#secure-boot)). On Intel Macs with a T2 chip, also allow booting from
  external media in Startup Security Utility.
- RAM for the model file plus its context (see the table below). The model list at boot greys
  out what doesn't fit.
- A USB stick formatted FAT32.

## Models

Each model is one `.BIN` file in the root of the stick (bigger ones also have `.B01`). They are
not in this repo; convert them with `export.py` (see [Making model files](#making-model-files)).

| File | Model | Weights | Size | Context | GB read per token | tok/s (QEMU, 8 threads) |
|---|---|---|---|---|---|---|
| `QWEN3-06.BIN` | Qwen3-0.6B | int8 | 624 MiB | 32k | 0.63 | 71 |
| `QW35-08.BIN` | Qwen3.5-0.8B | int8 | 785 MiB | 64k | 0.80 | 54 |
| `BONSAI8.BIN` | Bonsai-8B 1-bit | 1-bit | 1.1 GiB | 64k | 1.06 | 22 |
| `QWEN3-17.BIN` | Qwen3-1.7B | int8 | 1.7 GiB | 32k | 1.83 | 27 |
| `QW35-2.BIN` | Qwen3.5-2B | int8 | 1.9 GiB | 64k | 2.00 | 25 |
| `TBONSAI8.BIN` | Bonsai-8B ternary | ternary | 2.1 GiB | 64k | 2.01 | 19 |
| `QW35-4.BIN` | Qwen3.5-4B | int4 MLP, int8 rest | 3.3 GiB | 64k | 3.48 | 13 |
| `QW35-9.BIN` + `.B01` | Qwen3.5-9B | int4 MLP, int8 rest | 6.5 GiB | 64k | 6.32 | 7.6 |
| `TBONS27.BIN` + `.B01` | Ternary-Bonsai-2-27B | ternary | 6.7 GiB | 64k | 6.82 | 4.9 |

The speeds are for a ~120-word reply in QEMU (WHPX, 8 virtual CPUs) on a Ryzen 7 7840HS laptop
with DDR5-5600; real hardware will differ. Generating a token reads every weight once, so for the
int8/int4 models speed is close to `memory bandwidth / GB read per token` (here 43-50 GB/s).
The 1-bit and ternary models need more arithmetic per byte and are also limited by the CPU: here
they reached 23-39 GB/s. The app measures bandwidth at boot and shows an estimate per
model in the list. Prompt processing is faster than generation
(tokens are processed in batches).

Which to pick:

- Qwen3.5-2B or Qwen3.5-4B are good general choices on most machines.
- Bonsai-8B (1-bit or ternary) is an 8B model trained for low-bit weights. It is
  about as small and fast as the 2B models.
- Qwen3.5-9B and the 27B Bonsai need 8+ GB RAM and are the slowest, since they read the
  most per token.
- Qwen3-0.6B and Qwen3.5-0.8B are for old or slow machines.

Quality checks against the original Hugging Face models (in `tests/`): Qwen3-0.6B gives the same
greedy tokens as PyTorch FP32 (32/32); Qwen3.5-0.8B logits have cosine similarity 0.9986;
the tokenizer and chat template match on 3052 and 40 test strings. For Qwen3.5-4B,
perplexity versus the original bf16 weights is +0.6% with int8 everywhere and +3.3% with the
MLP in int4 (the export used here, 22% smaller).

### Context

By default the app uses the largest context that fits in memory (Settings, Context size, to
change it). When the context is full, the conversation starts over. The cache costs, per 1000
tokens:

| Model | per 1000 tokens | at the max |
|---|---|---|
| Qwen3-0.6B, Qwen3-1.7B | 112 MiB | 3.5 GiB (32k) |
| Qwen3.5-0.8B, 2B | 12 MiB | 0.75 GiB (64k) |
| Qwen3.5-4B, 9B | 32 MiB | 2 GiB (64k) |
| Bonsai-8B (both) | 144 MiB | 9 GiB (64k) |
| Ternary-Bonsai-2-27B | 64 MiB | 4 GiB (64k) |

Qwen3.5 and the 27B Bonsai use full attention in only every fourth layer, so their cache is small.

## USB stick

1. Format the stick as FAT32 (this erases it).
   - Up to 32 GB: File Explorer, right-click the drive, Format, FAT32.
   - Bigger sticks: Windows won't format these as FAT32. Use Rufus (Boot selection
     "Non bootable", File system "Large FAT32"), or on a Mac Disk Utility, Erase, "MS-DOS (FAT)",
     "GUID Partition Map".
2. Copy onto the stick:
   ```
   EFI\BOOT\BOOTX64.EFI      from the release page, or build/ after building
   QW35-2.BIN                one or more model files, in the root
   ```
   FAT32 files stop at 4 GiB, so `export.py` splits bigger models into `NAME.BIN`, `NAME.B01`,
   ... (a plain byte split). Copy all parts; only the `.BIN` shows up in the list. Use 8.3
   file names (up to 8 characters, a dot, 3 characters); the name shown on screen comes from
   inside the file. Up to 9 models are listed.

## Boot

1. Plug in the stick.
2. Open the boot menu: hold Option on a Mac; on a PC it's usually F12, F11, F8 or Esc right after
   power-on.
3. Pick the USB stick ("EFI Boot" on a Mac).
4. To leave, hold the power button. Nothing is written to the computer's disk, only to the stick.

With more than one model the app shows a list: size, whether it fits in this machine's memory,
and a speed estimate. Enter loads the highlighted one (the last one used, if it still fits).

## Secure Boot

The plain `BOOTX64.EFI` is unsigned, so it only boots with Secure Boot off. If you also use
Windows on the machine, turning Secure Boot off and on again can cost you the Windows Hello PIN
(and with BitLocker, a recovery key prompt): Windows sees a change in the firmware security
state. To avoid that, leave Secure Boot on and boot metallm through shim with your own key:

```
python tools/secureboot.py keygen        # once; needs: pip install cryptography
sh build.sh
python tools/secureboot.py stick E:\     # E: is the stick; copy the model files as usual
```

`stick` signs `build/BOOTX64.EFI` with `keys/MOK.key` and puts it behind shim on the stick:

```
EFI\BOOT\BOOTX64.EFI     shim, signed by Microsoft (downloaded from Ubuntu's shim-signed package)
EFI\BOOT\mmx64.efi       MokManager (same package)
EFI\BOOT\grubx64.efi     metallm, signed with your key (shim starts this file name)
METALLM.CER               your certificate
```

First boot on each machine: shim says "Verification failed: (0x1A) Security Violation" because it
doesn't know your key yet. Press Enter, then any key within 10 seconds ("Press any key to
perform MOK management"), then: Enroll key from disk, pick the stick, `METALLM.CER`,
Continue, Yes, Reboot. After that the stick boots straight into metallm, and so does every build
you sign with the same key. Run `stick` again after each rebuild.

- The firmware must trust the Microsoft UEFI CA (the "3rd party" CA). Most PCs do. Some
  laptops (Secured-core ones, Surface, some Lenovo and Dell models) ship with it off; the option
  is called something like "Allow Microsoft 3rd Party UEFI CA". Changing it is a one-time
  change to Secure Boot's key database. Have your BitLocker recovery key at hand, and you may
  have to set up the Windows PIN again once.
- Enrolling the key only touches shim's own list (MokList), not the firmware's keys, so
  Windows doesn't notice.
- `keys/` is in `.gitignore`. Keep `MOK.key` private: anything signed with it boots on every
  machine where you enrolled `METALLM.CER`. To remove it, use MokManager or
  `mokutil --delete` from Linux.
- `python tools/secureboot.py sign IN.EFI OUT.EFI` only signs a file (Authenticode, SHA-256).
  The signer is plain Python, so it works on Windows without `sbsign`.

## Using it

| Command / key | |
|---|---|
| `/clear` | new conversation |
| `/retry` | new answer to your last message |
| `/think` | step-by-step reasoning on/off |
| `/temp 0.3` | randomness, 0 = always the same answer; `/temp` shows it |
| `/settings` | settings and system info |
| `/model` | switch model |
| `/save` | save this chat now (shows the file name) |
| `/note text` | add a line to `NOTES.TXT`; `/notes` shows them |
| `/remember text` | add a line to `MEMORY.TXT`; `/memory` shows it, `/forget` clears it |
| `/` | list of commands |
| Esc | stop a reply |
| PgUp / PgDn, arrows, mouse wheel | scroll (works while a reply is streaming) |
| Home / End | top / back to the bottom |

The status bar shows speed, threads in use, and context used / context size.

Replies are formatted as they stream: headings, bold, italic, code, code blocks, lists, quotes,
tables and LaTeX math (`$x^2$`, `\frac{a}{b}`, Greek letters and so on are shown as Unicode).

## Settings

`/settings` opens a full-screen menu: Up/Down to choose, Left/Right or Enter to change, Esc to
go back. Saved to `SETTINGS.TXT` on the stick.

- Temperature, Top-p, Top-k: sampling. "auto" uses the model's recommended values.
- Thinking: step-by-step reasoning before the answer, off by default (same as `/think`).
- Reply length limit: stop a reply after this many tokens.
- Use MEMORY.TXT: pass `MEMORY.TXT` to the model as its system prompt.
- Autosave chats: write the chat to `CHATS\` after every reply.
- Context size: tokens the model can see. Changing it starts a new conversation.
- Threads: auto picks one thread per physical core when that is as fast as using all threads
  (it usually is, since generation is limited by memory speed).
- CPU boost: asks the CPU for its highest clock (Intel HWP or turbo ratio, AMD P0/CPPC).
  Firmware often leaves the CPU slow when no OS is running. This has only been tested in QEMU,
  which ignores it; on real hardware compare the clock in Info with boost on and off.
- Font size, Theme (dark, light, high contrast).
- Model: switch to another model on the stick.

Info shows the CPU, vector instructions in use, cores, clock speed before and after boost,
memory, measured memory bandwidth, model, context and speed.

## Files on the stick

| File | |
|---|---|
| `CHATS\CHAT0001.TXT` ... | one file per conversation, plain text |
| `NOTES.TXT` | your `/note` lines |
| `MEMORY.TXT` | system prompt, one line per `/remember`; edit it on any computer |
| `SETTINGS.TXT` | settings, `key=value` lines |
| `LOG.TXT` | details of the last boot (CPU, clocks, memory, timings), for debugging |

If the stick is read-only or full, the app says so and keeps running.

## Making model files

Needs Python 3.10+ and `pip install -r requirements.txt`. Download the source model, then run
`export.py SOURCE OUT.BIN "Display name"`.

```
# Qwen3 / Qwen3.5 (Hugging Face folders, e.g. hf download Qwen/Qwen3.5-2B --local-dir model-q35-2b)
python export.py model-q3-0.6b  build/QWEN3-06.BIN "Qwen3-0.6B"      # Qwen/Qwen3-0.6B
python export.py model-q3-1.7b  build/QWEN3-17.BIN "Qwen3-1.7B"      # Qwen/Qwen3-1.7B
python export.py model-q35-0.8b build/QW35-08.BIN  "Qwen3.5-0.8B"    # Qwen/Qwen3.5-0.8B
python export.py model-q35-2b   build/QW35-2.BIN   "Qwen3.5-2B"      # Qwen/Qwen3.5-2B
python export.py model-q35-4b   build/QW35-4.BIN   "Qwen3.5-4B" --bits 4 --head-bits 8 --attn-bits 8
python export.py model-q35-9b   build/QW35-9.BIN   "Qwen3.5-9B" --bits 4 --head-bits 8 --attn-bits 8

# Bonsai (GGUF weights from prism-ml; the first argument is a folder with the
# tokenizer files and config.json of the base model)
python export.py tok-qwen3-8b build/BONSAI8.BIN  "Bonsai-8B 1-bit"   --gguf Bonsai-8B-Q1_0.gguf            # prism-ml/Bonsai-8B-gguf
python export.py tok-qwen3-8b build/TBONSAI8.BIN "Bonsai-8B ternary" --gguf Ternary-Bonsai-8B-PQ2_0.gguf    # prism-ml/Ternary-Bonsai-8B-gguf
python export.py tok-qwen35   build/TBONS27.BIN  "Bonsai-2-27B ternary" --gguf Ternary-Bonsai-2-27B-PQ2_0.gguf  # prism-ml/Ternary-Bonsai-2-27B-GGUF
```

For the Bonsai-8B models, the tokenizer folder can be the `-unpacked` repo of the same model
(prism-ml/Bonsai-8B-unpacked) without its weight files. For the 27B, use the tokenizer files
and `config.json` of a Qwen3.5 model; without `config.json` the tokenizer loads with the wrong
pre-tokenizer.

Other options: `--bits 4|8` (int8 is the default), `--attn-bits`, `--head-bits`, `--ctx N`
(context length, default 32768 for Qwen3 and 65536 otherwise). `python bin_info.py FILE.BIN`
prints the size and weight mix of a model file.

## Building

- `sh build.sh` builds `build/BOOTX64.EFI` (and `build/qwen.dll`, the engine for the Python
  tests). Needs LLVM clang + lld-link; set `LLVM=/path/to/llvm/bin` if it isn't in
  `C:\Program Files\LLVM\bin`. Written for Git Bash on Windows.
- `python tools/gen_font.py` regenerates `src/font.h` from the fonts in `tools/fonts/`.
- `python tools/make_disk.py build/disk.img --file EFI/BOOT/BOOTX64.EFI=build/BOOTX64.EFI --file QW35-2.BIN=build/QW35-2.BIN`
  builds a FAT32 disk image (file names on the image must be 8.3). Flash it with Rufus ("DD
  Image" mode) or balenaEtcher, or boot it in QEMU.
- `python tools/qemu_test.py build/disk.img build/qemu --accel whpx --mem 12288 --type "hello" --type "@pgup end"`
  boots headless QEMU with OVMF, types, and saves screenshots (`@...` sends raw key names). The
  QEMU and OVMF paths can be set with the `QEMU` and `OVMF` environment variables.
- `tests/`: `test_engine.py` and `test_chat.py` compare the C engine with Hugging Face
  transformers, `test_tokenizer.py` the tokenizer, `bench_kernels.py` the matrix kernels.

## Layout

- `src/engine.c`, `src/engine_api.inc`, `src/tokenizer.inc`: transformer (Qwen3, Qwen3.5 hybrid
  attention), BPE tokenizer, chat template, sampler
- `src/kernels_avx2.c`, `src/kernels_avx512.c`: int8 / int4 / 1-bit / ternary matrix kernels
  (AVX2, AVX-VNNI, AVX-512 VNNI)
- `src/efi_main.c`: UEFI side: memory, keyboard/mouse, model list, multicore, settings, chat
- `src/cpu.c`: CPU features, clock boost, topology, bandwidth probe
- `src/md.c`: markdown and LaTeX formatting while a reply streams
- `src/fs.c`, `src/settings.c`: files on the stick, `SETTINGS.TXT`
- `src/ap_tramp.S`: starts the other cores directly if the firmware won't
- `src/sbat.c`: SBAT metadata, required by shim (Secure Boot)
- `tools/secureboot.py`: signing key, Authenticode signer, shim stick setup
- `src/console.c`, `src/font.h`: framebuffer text with scrollback, themes
- `export.py`, `gguf_read.py`, `bin_info.py`: model conversion

## Licenses

The code is MIT (see `LICENSE`). The models keep their own licenses: Qwen3 and Qwen3.5 are
Apache 2.0, and so are the Bonsai models from prism-ml.

`src/font.h` is rendered from these fonts (in `tools/fonts/` with their license files):

- Cascadia Mono (regular, bold, italic): SIL Open Font License 1.1
- DejaVu Sans Mono, DejaVu Sans: Bitstream Vera license + public domain changes (see `LICENSE-DejaVu.txt`)
- Noto Sans Math, Noto Sans Symbols 2: SIL Open Font License 1.1
