#!/bin/bash
# Full-frame FLUX.2 small decoder contexts for Hexagon V81 (Snapdragon 8 Elite Gen 5), every MOSS size preset (Moss.PRESETS) plus 512x512 (tiled fallback):
#   moss/build_vaes_v81.sh [WORK]   -> WORK/v81/out_v81/vae_WxH.bin      (WORK default ~/anima-npu/work/moss; needs WORK/vae/{config.json,diffusion_pytorch_model.safetensors}
#                                       from black-forest-labs/FLUX.2-small-decoder, the python env of ~/anima-npu/work/.venv and the qairt243 container, see tools/qnn_ctx.sh)
set -u
R=$(cd "$(dirname "$0")/.." && pwd); W=${1:-$HOME/anima-npu/work/moss}; cd "$W"; mkdir -p v81
source "$HOME/anima-npu/work/.venv/bin/activate"
for s in 512x512 512x768 768x512 640x640 768x768 1024x1024 576x1024 1024x576 640x1152 1152x640 768x1024 1024x768 896x1152 1152x896 832x1216 1216x832 768x1344 1344x768; do
  Wd=${s%x*}; Ht=${s#*x}; n=vae_${Wd}x${Ht}
  [ -f v81/out_v81/$n.bin ] && { echo "have $n"; continue; }
  if [ ! -f v81/$n.dlc ] && [ ! -f v81/$n.onnx ]; then
    NOREF=1 systemd-run --user --scope -q -p MemoryMax=48G python "$R/moss/export_vae_moss.py" v81/$n.onnx $((Ht / 8)) $((Wd / 8)) > v81/$n.export.log 2>&1 || { echo "export failed $s"; continue; }
    python -c "import onnx; m=onnx.load('v81/$n.onnx'); m.ir_version=9; onnx.save(m,'v81/$n.onnx')"
  fi
  systemd-run --user --scope -q -p MemoryMax=48G "$R/tools/qnn_ctx.sh" v81 $n > v81/$n.ctx.log 2>&1
  if [ -f v81/out_v81/$n.bin ]; then echo "built $n $(stat -c %s v81/out_v81/$n.bin)"; rm -f v81/$n.onnx v81/$n.onnx.z.bin; else echo "context failed $s"; tail -3 v81/$n.ctx.log; fi
done
echo finished; ls -la v81/out_v81/*.bin
