# Third-party components

This repository contains only HexAnima's own source code (MIT, see `LICENSE`). Nothing below is stored here.

| Component | Used for | License |
|---|---|---|
| Anima (CircleStone Labs), built on NVIDIA Cosmos | base model | CircleStone Labs Non-Commercial License; NVIDIA Open Model License |
| Anima Turbo v1.1 Lora (somedoby) and TURBO for ANIMA V4-29B (n_Arno), both on Civitai | distillation LoRAs merged into the downloadable models (4-step generation); the LoRA files are not redistributed | Civitai terms of each model page (credit required for n_Arno's), plus the CircleStone Labs Non-Commercial License v1.2 for the Anima Turbo derivative |
| MiaoMiao Harem / 3D Harem / RealSkin (MIAOKA) | downloadable models | author's terms: non-commercial, no selling of images |
| Qwen3 0.6B text encoder, Qwen-Image VAE | text encoder, VAE (on Snapdragon 8 Elite Gen 5 this fork downloads the stock text encoder file `qwen_3_06b_base.safetensors` from circlestone-labs/Anima for the CPU path) | Apache 2.0 |
| MOSS-Image-SFT (OpenMOSS), Unsloth GGUF quantisation, rapid-moss 4-step adapter (darask0, on lodestones' 8-step DMD), Qwen3.5-9B (Qwen team) with Unsloth GGUF quantisation, FLUX.2 small decoder (Black Forest Labs) | the optional MOSS-Image pack (converted for the NPU, hosted in its own repository with `LICENSE` and `NOTICE.md`) | Apache 2.0 (each component) |
| stable-diffusion.cpp | sampler and CPU parts (`patches/` are applied to it) | MIT |
| TAEHV (`taew2_1`) | preview decoder | see github.com/madebyollin/taehv |
| Danbooru tag list (DominikDoom/a1111-sd-webui-tagcomplete) | prompt suggestions | see that repository |
| Qualcomm AI Engine Direct (QNN) runtime libraries | bundled in the APK | Qualcomm AI Stack License |
| Qualcomm Hexagon SDK / QHL (HMX matrix library) | building the DSP kernels | Qualcomm license, not redistributed here |

## MOSS-Image notice

All MOSS-Image components are under the Apache License 2.0. The pack is converted and quantised (8-bit tiles for the Hexagon matrix unit, the rapid-moss LoRA kept unmerged), see the pack's `NOTICE.md`.
The Apache License 2.0 text and that notice are bundled in the app (`app/assets/licenses/apache-2.0.txt`, `moss_notice.txt`) and shown under Info -> Licenses.
A user-supplied LoRA is a separate work under its author's license; HexAnima only folds it with the adapter on the phone (nothing is uploaded or redistributed).

## CircleStone Model notice

The downloadable model files are modified versions of the CircleStone Model (Anima): MIAOKA's fine-tunes with a turbo LoRA merged in, converted to HexAnima's block layout.
They are not an official product of CircleStone Labs and have not been endorsed by it. The Attribution Notice, the modification statement and the license text are published with the model packs
at https://huggingface.co/ayanoby/anima-npu-packs (`NOTICE.txt`, `LICENSE.md`).

## Qualcomm components

- The QNN runtime libraries (`libQnnHtp*.so`, `libQnnSystem.so`) are **not** in this repository and must never be added to it or published on their own:
  the Qualcomm AI Stack License allows distributing them only in object code and only as part of your own application (not standalone).
  `tools/stage_jnilibs.sh` copies them from a QNN SDK that you download from Qualcomm and accept the license for.
  The license text and notices are shown in the app (Info -> Licenses) and live in `app/assets/licenses/`.
- `hexagon/mega/build.sh` needs the Hexagon SDK and the QHL HMX sources (`hexagon/mega/qhl_hmx`) plus `syslib/libcdsprpc.so` from a device.
  They are Qualcomm proprietary files and are not included; get them from the SDK you are licensed for.
