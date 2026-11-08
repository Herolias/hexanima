<h1 align="center">HexAnima</h1>
<p align="center"><b>Anime image generation that runs entirely on your phone's NPU.</b><br>
No root, no cloud, no account, no ads. Free and non-commercial.</p>

<p align="center">
  <img src="docs/screens/01-generate.jpg" width="240" alt="Generating a picture">
  <img src="docs/screens/02-progress.jpg" width="240" alt="Progress and stop button">
  <img src="docs/screens/08-history.jpg" width="240" alt="History">
</p>

HexAnima runs the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model and fine-tunes of it on the Hexagon NPU of Snapdragon 8 Gen 1 / 8+ Gen 1 phones.
The diffusion transformer is executed by custom HMX kernels; the text encoder and the sampler run on the CPU.
A 512x768 picture takes about **7 s** with the 2B models and **12 s** with the 2.9B models (4 steps, Xiaomi 12T Pro).

It can also run **MOSS-Image 5B** (OpenMOSS, Apache-2.0), a larger model that you prompt with plain sentences. Its text encoder (Qwen3.5-9B), the transformer and the decoder all run on the NPU:
about 30 s for 512x768 and about 80 s for 1 Mpx; any size with sides that are multiples of 16 (up to 6144 patches). It is an optional ~14 GB download.

> **This fork** adds support for the **Snapdragon 8 Elite Gen 5** (SM8850, Hexagon V81, e.g. Xiaomi 17 Pro Max), branch `port/v81-sm8850`.
> - **HMX on V81**: the legacy accumulator store writes zeros on v73+; the DSP skels (Anima and MOSS) now read the accumulator through the v73+ cvt path
>   (scale/bias block in VTCM), picked at run time, so one v69-compiled skel serves V69 and V81.
> - **QNN on V81**: text encoder, decoders and MOSS's FLUX.2 decoder are recompiled as V81 context binaries with QAIRT 2.43 (`tools/qnn_ctx.sh`,
>   `export/build_common_arch.sh`, `moss/build_vaes_v81.sh`; host tools in an Ubuntu 22.04 container, `tools/qairt.Containerfile`). The app ships the 2.43 runtime
>   as `libQnnHtp243.so` / `libQnnSystem243.so` next to upstream's 2.28 libraries. Without the V81 contexts, Anima's text encoder and decoder fall back to the CPU (sd.cpp).
> - **Measured on a Xiaomi 17 Pro Max**: Anima 2B 1024x1024 19.5 s (was 37.9 s with the CPU text encoder/decoder), 512x768 warm ~4 s on the NPU side;
>   MOSS-Image 1024x1024 ~38 s and 832x1216 37.6 s (6 DSP workers and 20 DiT blocks kept in the page cache on 14 GB+ phones; 3 workers: 45 s). Outputs are
>   bit-identical across worker counts; 8 workers hang the DSP call, so the count is capped at 6.
> - Startup HMX self-test with automatic restart and a clean stop (SIGTERM, then SIGKILL), new settings (best quality, CPU threads, NPU worker threads, large sizes),
>   16 KB-aligned APK packaging. Build: `QAIRT=<sdk> DEBUGGABLE=1 tools/build_fork.sh [--install]` (podman or docker, host Android SDK; the unchanged native
>   libraries come from the upstream release APK). The QAIRT runtime is not redistributable: do not publish APKs built with it.
> - **LoKr support** (LyCORIS / ai-toolkit `lokr_w1`/`lokr_w2`): full-rank, so it is merged into the DiT weight tiles on the phone (`lorafold`, exact Kronecker
>   update + requant: 99.5% of the update survives, the rest is one more 8-bit rounding) and mossrun reads the merged blocks; classic LoRAs still fold into the slots.
> - **More accurate MOSS DiT**: `MOSS_GGUF=models/MOSS-Image-SFT-Q8_0.gguf MOSS_REFDIR=out/ref python moss/export_moss.py out/dit` repacks unsloth's Q8_0 GGUF
>   exactly into the same 8-bit tiles (same size and speed); the published pack comes from UD-Q5_K_XL (q/k 4%, o 2% weight error). Block-level check of the V81
>   kernels against the PyTorch reference (moss/mkblk_test.py + cmp_blk.py, block 20 at 512x768): q/k/v/o/FFN/output within 0.03-0.09%.
> - **MOSS-Image full-model mode** (main screen: *Fast · 4 steps* / *Full model*): the rapid-moss adapter is switched off (its LoRA-A matrices are zeroed after
>   each block read) and mossrun samples the way Unsloth Studio runs MOSS-Image-SFT GGUFs (stable-diffusion.cpp defaults): Euler on sd.cpp's discrete
>   shift-5 schedule, plain CFG with a negative prompt. `pipeline_moss_image.py`'s own defaults (schedule snapped to the training grid, block 22 skipped in the
>   negative branch) are kept
>   behind `MOSS_FULL="full CFG NEG 5 reference"`: on the SFT model that skip gives smooth, airbrushed faces (`moss/sampler_gpu.py`, same seed and noise on a desktop GPU: mean gradient
>   3.6 with the skip, 4.3 without, i.e. visibly more skin and hair detail; the schedule alone changes little).
>   Steps 8-60, CFG 1-8, negative prompt; the settings go into the PNG. Both branches run through each DiT block before the next one is read, so the weights are still
>   streamed once per step. Speed on the 8 Elite Gen 5: ~1.6 s per DiT pass at 512x512, ~8.5 s at 1024x1024 (rising to ~10 s as the phone warms up): 20 steps
>   with CFG at 1024x1024 took 6.6 min.
>   Checked against the fp32 PyTorch reference (`moss/check_full.py` on a `MOSS_DUMP` of a phone run, 512x512, 8 steps, CFG 3, reference settings): every step's velocities, recomputed
>   on the phone's own latents, within 0.03-0.32% (both branches); CFG compounds these fp16-level differences to ~7% in the final latent, which decodes to the
>   same picture (`moss/decode_cmp.py`: mean pixel difference 2 of 255). Runs are bit-reproducible for a seed.
>   Needs `files/moss/glue_base.bin` (the SFT model's own time embedding, `moss/export_glue_base.py`). Classic LoRAs are folded a second time without the adapter
>   (`blkNN.ovf`); merged LoKr blocks serve both modes.
> - **Adapter precision**: the rapid-moss LoRA's trained factors cancel heavily (|BA| is ~9% of |B||A|), so storing them as trained in 8-bit tiles lost 6-10% of
>   its delta. `export_moss.py` and `lorafold` now store every rank-128 slot as B' = U sqrt(S), A' = sqrt(S) V^T (same product): 1-2.4% error over all 42 blocks,
>   base weights byte-identical. Same seed in Fast mode: same picture, finer details differ (mean pixel difference 9 of 255).
> - **Seeds match Unsloth Studio**: Studio renders MOSS-Image GGUFs with stable-diffusion.cpp, so mossrun draws the initial noise with sd.cpp's default RNG
>   (`PhiloxRNG`, AUTOMATIC1111's imitation of CUDA randn: `philox_randn` in mossrun.cpp, within 5e-7 of sd.cpp's algorithm, `moss/check_noise.py`) in both
>   Fast and Full mode. With Studio's prompt, negative prompt, size, steps, CFG and seed the phone starts from the same noise and lands on the same picture
>   (checked on Studio's own 960x1280 seed: same pose, light and layout; fine details differ with the arithmetic). Seeds saved before this change give
>   different pictures now.
> - **8-bit text encoder**: CFG amplifies text-side error, and the 4-bit Qwen3.5-9B (Unsloth UD-Q4_K_XL) was ~6% off an 8-bit (Q8_0) reference on the refined
>   prompt (`moss/te_cmp.py`: ~5% from the 4-bit weights, ~3% from the NPU kernels). In full mode that showed up as broken anatomy (limbs out of place, merged fingers):
>   the same seed with the PC's 8-bit text injected came out clean. `moss/export_te_all.py` with `TE_GGUF=Qwen3.5-9B-Q8_0.gguf` repacks every matrix exactly into
>   8-bit tiles (7.6 GB in `te/`, largest layer 246 MB, still fits mossrun's 250 MB stream buffer) plus `embd.q8` (the Q8_0 token table, 1.08 GB); mossrun uses
>   `embd.q8` when it is present and keys its text caches apart. Decoder ruled out: the FLUX.2 small decoder and the full FLUX.2-klein VAE decode the same latent
>   to the same sharpness.
> - **Open**: the V81 context binaries and `glue_base.bin` are not hosted anywhere yet, so they are copied into the app with `adb` (`files/anima/{te0..te3,vae}.bin,
>   te_embed.f16`, `files/vae81/vae_WxH.bin`, `files/moss/glue_base.bin`, the 8-bit `files/moss/te/` + `embd.q8`); sizes above 4096 patches are unverified on V81; full mode is untested on V69.

## Features

- Fully offline after the models are downloaded; nothing leaves the phone.
- Four MiaoMiao models (2B fp16 and 2.9B q8) with a one-tap model manager: resumable downloads, sha256 verification, delete to free space.
- Size picker with portrait / landscape / square presets (all multiples of 16) and custom sizes, seed, batch of several pictures, stop button.
- History with favourites, trash, multi-select, share and save; settings are stored in the PNG, so any picture can be reopened with its prompt and seed.
- Prompt tag suggestions (Danbooru tags), templates, quality-tag prefix (Anima models only; MOSS-Image takes plain sentences and gets no tags).
- Your own LoRA for MOSS-Image: pick a `.safetensors` file, the app checks it at once (layer names, shapes, rank, data type) and says why if it does not fit; a strength slider (0-2). The LoRA is folded with the built-in 4-step adapter on the phone (about 15 s the first time) and never leaves the device.
- Live RAM meter, thermal / battery warnings, the model unloads itself when idle.
- Light / dark / system themes, notifications, 13 languages (English, Russian, Chinese, Japanese, Korean, Spanish, Portuguese, German, French, Indonesian, Turkish, Vietnamese, Hindi).

## Picture quality

Straight from the phone, 512x768, 4 steps, no upscaling, no editing (screenshots of the app; the date and battery are cropped out).

<p align="center">
  <img src="docs/screens/quality/3d-harem-2b.jpg" width="190" alt="MiaoMiao 3D Harem 2B">
  <img src="docs/screens/quality/realskin-2b.jpg" width="190" alt="MiaoMiao RealSkin 2B">
  <img src="docs/screens/quality/harem-2.9b.jpg" width="190" alt="MiaoMiao Harem 2.9B">
  <img src="docs/screens/quality/realskin-2.9b.jpg" width="190" alt="MiaoMiao RealSkin 2.9B">
</p>
<p align="center"><sub>3D Harem LH3D 1.1 (2B, ~7 s) · RealSkin Anima 1.3 (2B, ~7 s) · Harem 2.9B BETA 1.1 (~12 s) · RealSkin 2.9B 1.0 (~12 s)</sub></p>

### MOSS-Image

Four different prompts, 768x1024, 4 steps, about 60 s each, straight from the phone (same method as above, nothing retouched). Text in the picture is a strength of MOSS-Image but not a guarantee:
short words usually come out right, longer or secondary text (shop-front boards, book covers, small signs) is often garbled. The pictures below are the first and only attempt for each prompt, bad text left as it is.

<p align="center">
  <img src="docs/screens/quality/moss-1.jpg" width="190" alt="MOSS-Image: coffee shop with a neon sign">
  <img src="docs/screens/quality/moss-2.jpg" width="190" alt="MOSS-Image: fox reading a book, poster title">
  <img src="docs/screens/quality/moss-3.jpg" width="190" alt="MOSS-Image: girl holding a board with text">
  <img src="docs/screens/quality/moss-4.jpg" width="190" alt="MOSS-Image: arcade machine, GAME OVER screen">
</p>
<p align="center"><sub>"neon sign that says OPEN LATE" (the sign is right, the boards are not) · "title at the top says THE LAST LIBRARY" (title right, subtitle garbled) · "board that says HEXANIMA" (right) · "the words GAME OVER in pixel font" (readable, with small glitches)</sub></p>

**Your own LoRA.** Pick a `.safetensors` file made for MOSS-Image; the app checks it at once (layer names, shapes, rank, data type) and tells you why if it does not fit (a FLUX or SDXL LoRA, a wrong shape, a rank above 128, a text-encoder-only file...). A strength slider goes from 0 to 2.
The LoRA is folded with the built-in 4-step adapter on the phone (about 15 s the first time for a new LoRA or strength, then cached) and never leaves the device. Left: without a LoRA, right: with a third-party style LoRA from Civitai (same prompt, seed and size; LoRAs stay under their authors' licenses).

<p align="center"><img src="docs/screens/quality/moss-lora-compare.jpg" width="640" alt="Without and with a LoRA"></p>

Limits worth knowing: MOSS-Image only does text-to-image (no editing or image input), it needs about 14 GB of storage and was tested on a 12 GB phone only, and its weights are streamed from storage for every picture, so a picture takes 30 s at 512x768 and about 80 s at 1 Mpx.

## A tour of the app

<table>
<tr>
<td align="center" width="33%"><img src="docs/screens/03-suggestions.jpg" width="220"><br><b>Tag suggestions</b><br><sub>Danbooru tags complete as you type</sub></td>
<td align="center" width="33%"><img src="docs/screens/04-templates.jpg" width="220"><br><b>Templates</b><br><sub>Recent and saved prompts</sub></td>
<td align="center" width="33%"><img src="docs/screens/05-size.jpg" width="220"><br><b>Size picker</b><br><sub>Portrait, landscape, square, custom</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/screens/06-model-picker.jpg" width="220"><br><b>Model picker</b><br><sub>Switch models in one tap</sub></td>
<td align="center"><img src="docs/screens/07-model-manager.jpg" width="220"><br><b>Model manager</b><br><sub>Size, speed, download and delete</sub></td>
<td align="center"><img src="docs/screens/10-viewer.jpg" width="220"><br><b>Picture details</b><br><sub>Prompt, seed, size, model; reuse the settings</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/screens/09-select.jpg" width="220"><br><b>Multi-select</b><br><sub>Favourite, save, share or trash many at once</sub></td>
<td align="center"><img src="docs/screens/11-favourites.jpg" width="220"><br><b>Favourites</b><br><sub>Starred pictures in one place</sub></td>
<td align="center"><img src="docs/screens/12-settings.jpg" width="220"><br><b>Settings</b><br><sub>Language, theme, idle unload, quality tags</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/screens/13-info.jpg" width="220"><br><b>Info</b><br><sub>Why 4 steps, which sizes work, MOSS-Image and LoRA notes, licenses</sub></td>
<td align="center"><img src="docs/screens/14-light.jpg" width="220"><br><b>Light theme</b><br><sub>Light, dark or follow the system</sub></td>
<td align="center"><img src="docs/screens/15-languages.jpg" width="220"><br><b>13 languages</b><br><sub>Chosen in Settings</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/screens/16-trash.jpg" width="220"><br><b>Trash</b><br><sub>Deleted pictures stay 30 days, restore any time</sub></td>
<td align="center"><img src="docs/screens/17-moss.jpg" width="220"><br><b>MOSS-Image</b><br><sub>Plain-sentence prompts, any size</sub></td>
<td align="center"><img src="docs/screens/18-lora.jpg" width="220"><br><b>Your LoRA</b><br><sub>Checked at once, strength slider, prepared on the phone</sub></td>
</tr>
<tr>
<td align="center"><img src="docs/screens/19-moss-models.jpg" width="220"><br><b>Five models</b><br><sub>MOSS-Image next to the four Anima models</sub></td>
<td align="center"><img src="docs/screens/20-info-moss.jpg" width="220"><br><b>Info: MOSS-Image</b><br><sub>Sizes, LoRA, licenses</sub></td>
<td></td>
</tr>
</table>

## Requirements

- A phone with **Snapdragon 8 Gen 1 or 8+ Gen 1** (SM8450 / SM8475, Hexagon V69): the tested target.
- **Snapdragon 8 Elite** (SM8750, Hexagon V79, e.g. Galaxy S25): not supported. Versions 1.2.0 and 1.2.1 had experimental support, withdrawn in 1.2.2: on a tester's Galaxy S25 the NPU matrix unit is sometimes silently unavailable after an interrupted run (garbage pictures) and large sizes come out as noise. A proper port is not planned in the short term.
- Other chips (Snapdragon 8 Gen 2 / 8 Gen 3, SM8550 / SM8650) are not supported yet; the app tells you if yours is not.
- Android 10 or newer, 64-bit. No root needed.
- RAM: tested on a 12 GB phone (about 10.9 GB usable). Phones with less RAM are untested; the 2.9B models need the most. Storage: 1.2 GB for the shared files plus 3.6-4.2 GB per Anima model; MOSS-Image needs about 14 GB (and ~300 MB more while a user LoRA is in use). MOSS-Image was tested on a 12 GB phone only.

## Install

1. Download `HexAnima-<version>.apk` from the [Releases](../../releases) page and install it (allow installs from your browser or file manager).
2. Open the app and pick a model in **Models**. The files are downloaded from Hugging Face (`ayanoby/anima-npu-packs`) the first time.
3. Type a prompt (Danbooru-style tags work best), press **Generate**.

MOSS-Image is a separate download in the same manager (it lives in its own Hugging Face repository with its own `LICENSE` and `NOTICE.md`).

Tips: the models are distilled for 4 steps and use no negative prompt; see **Info** in the app for the other notes.

## Disclaimer

**HexAnima is an independent, unofficial project. It is not affiliated with, endorsed by, sponsored by or otherwise connected to CircleStone Labs, NVIDIA, Qualcomm, Alibaba (Qwen), OpenMOSS, Unsloth, Black Forest Labs or the authors of the models it can download.**
"Anima" and "Cosmos" refer to third-party models, "NVIDIA" and "Qualcomm", "Snapdragon", "Hexagon" are trademarks of their respective owners.
They are used only to describe what the software works with.
The models (MiaoMiao Harem / 3D Harem / RealSkin by MIAOKA, based on Anima by CircleStone Labs and NVIDIA Cosmos) stay under their authors' licenses: non-commercial use only.
The models are made with a turbo (distillation) LoRA merged in: Anima Turbo v1.1 Lora by somedoby for the 2B models and TURBO for ANIMA (V4-29B) by n_Arno for the 2.9B models (both on Civitai). See `THIRD_PARTY.md`.
MOSS-Image, Qwen3.5-9B, the rapid-moss 4-step adapter (darask0, on lodestones' 8-step DMD), the FLUX.2 small decoder (Black Forest Labs) and the Unsloth GGUF quantisations are all under the **Apache License 2.0**; the MOSS-Image pack holds converted copies (the changes are listed in its `NOTICE.md`, also shown in the app under Info -> Licenses).
LoRA files that you load yourself stay under their authors' licenses; check them before you share pictures made with them.
A model author can ask for removal of their model from the download catalog: open a discussion in the Community tab of the model packs repository.

## Third-party components

The APK includes the Qualcomm AI Engine Direct (QNN) runtime libraries (`libQnnHtp*.so`, `libQnnSystem.so`) under the Qualcomm AI Stack License.
They are redistributed only in object code and only as part of the app, never standalone; the license text and the third-party notices are in `app/assets/licenses/` and in the app (Info -> Licenses).
Qualcomm may terminate that license at any time, in which case distribution of the APK stops.

## Layout

- `app/`: the Android app (plain Java, built without Gradle by `app/build.sh`), package `com.hexanima.npu`. Languages in `app/assets/lang/`.
- `phone/`: the on-device server (`animad.cpp`) and `anima.sh`, which the app runs.
- `chainrun/`, `hexagon/mega/`: helper processes and the Hexagon HMX kernels (see `THIRD_PARTY.md` for the Qualcomm files you have to supply).
- `patches/`: patches for stable-diffusion.cpp (NPU/TE/VAE hooks, serve mode); `android/build_sd.sh` builds it.
- `export/`, `tools/`: model conversion (`make_pack.py` builds the downloadable packs), translations (`build_lang.py`, `lang/`), `stage_jnilibs.sh`.
- `docs/PITFALLS.md`: engineering notes on the NPU work.

## Build

You need the Android SDK (build-tools 34, platform 34), Docker, the Qualcomm QNN SDK (download it from Qualcomm and accept its license) and, for the DSP kernels, the Hexagon SDK.
Paths in the scripts default to `~/anima-npu/sdk/...`. The order is: build the native helpers (`phone/build.sh`, `chainrun/build.sh`, `hexagon/mega/build*.sh`, `android/build_sd.sh`),
copy them with `tools/stage_jnilibs.sh`, then `app/build.sh` produces and installs `hexanima.apk`.
The model packs are built by `tools/make_pack.py` and downloaded by the app from a Hugging Face repository (set in `Main.DEFAULT_BASE`).

## License

The HexAnima source code is MIT (`LICENSE`). Models, QNN and the other components keep their own licenses: see `THIRD_PARTY.md`.
The app shows all required license texts and notices (Info -> Licenses): Qualcomm AI Stack License, QNN notices, Apache License 2.0 and the MOSS-Image notice; the files are in `app/assets/licenses/`.
