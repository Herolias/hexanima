#!/bin/bash
# Anima's NPU text encoder (Qwen3 0.6B, 4 slices of 7 layers, 128 tokens) and TAEHV decoder (64x96 latent = 512x768 px, animad tiles other sizes)
# as QNN context binaries for one Hexagon generation, plus the fp16 token table animad gathers from:
#   export/build_common_arch.sh [WORK] [ARCH SOC]   -> WORK/out_ARCH/{te0,te1,te2,te3,vae}.bin and WORK/te_embed.f16   (default WORK ~/anima-npu/work/anima, v81 / 87)
# needs WORK/models/{miaomiao3DHarem_animaLH3D10_txt.safetensors (= qwen_3_06b_base.safetensors), taew2_1.safetensors}, the python env of
# ~/anima-npu/work/.venv and the qairt243 container (tools/qnn_ctx.sh). Install on the phone: files/anima/{te0..te3,vae}.bin + te_embed.f16.
set -u
R=$(cd "$(dirname "$0")/.." && pwd); W=${1:-$HOME/anima-npu/work/anima}; A=${2:-v81}; S=${3:-87}
cd "$W"; source "$HOME/anima-npu/work/.venv/bin/activate"; export MODELS=$W/models
ir9() { python -c "import onnx,sys; m=onnx.load(sys.argv[1]); m.ir_version=9; onnx.save(m,sys.argv[1])" "$1"; }
[ -f te_embed.f16 ] || (cd "$R/export" && python extract_embed.py "$W/te_embed.f16")
for i in 0 1 2 3; do
  n=te$i; [ -f out_$A/$n.bin ] && { echo "have $n"; continue; }
  [ -f $n.dlc ] || [ -f $n.onnx ] || { (cd "$R/export" && systemd-run --user --scope -q -p MemoryMax=24G python export_te.py $((i * 7)) $((i * 7 + 7)) "$W/$n.onnx" 128) > $n.export.log 2>&1 && ir9 $n.onnx; }
  systemd-run --user --scope -q -p MemoryMax=24G "$R/tools/qnn_ctx.sh" . $n $A $S > $n.ctx.log 2>&1; ls -la out_$A/$n.bin 2>/dev/null || { echo "context failed $n"; tail -3 $n.ctx.log; }
done
n=vae
if [ ! -f out_$A/$n.bin ]; then
  [ -f $n.dlc ] || [ -f $n.onnx ] || { (cd "$R/export" && python export_vae.py "$W/$n.onnx") > $n.export.log 2>&1 && ir9 $n.onnx; }
  systemd-run --user --scope -q -p MemoryMax=24G "$R/tools/qnn_ctx.sh" . $n $A $S > $n.ctx.log 2>&1; ls -la out_$A/$n.bin 2>/dev/null || { echo "context failed $n"; tail -3 $n.ctx.log; }
fi
ls -la out_$A te_embed.f16
