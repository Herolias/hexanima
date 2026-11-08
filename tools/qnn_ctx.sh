#!/bin/bash
# ONNX -> fp16 DLC -> HTP context binary for one Hexagon generation, with the QAIRT host tools in a container (Ubuntu 22.04, tools/qairt.Containerfile).
#   tools/qnn_ctx.sh DIR NAME [ARCH SOC]      DIR/NAME.onnx -> DIR/out_ARCH/NAME.bin      (default ARCH v81, SOC 87 = SM8850)
# SOC_MODEL (QnnTypes.h): SM8450=36 SM8475=42 SM8550=43 SM8650=57 SM8750=69 SM8850=87
# env: QAIRT (SDK root, default ~/anima-npu/sdk/qairt/qairt/2.43.0.260128), IMG (container image, default qairt243), VTCM_MB (8), EXTRA (more converter flags)
set -euo pipefail
D=$(cd "$1" && pwd); N=$2; A=${3:-v81}; S=${4:-87}
QAIRT=${QAIRT:-$HOME/anima-npu/sdk/qairt/qairt/2.43.0.260128}; IMG=${IMG:-qairt243}
CR=$(command -v podman || command -v docker)
mkdir -p "$D/out_$A"
"$CR" run --rm -v "$QAIRT":/qairt:ro,z -v "$D":/w:z "$IMG" bash -c "
set -e; cd /w
echo '{\"graphs\":[{\"vtcm_mb\":${VTCM_MB:-8},\"graph_names\":[\"$N\"],\"O\":3.0,\"fp16_relaxed_precision\":1}],\"devices\":[{\"dsp_arch\":\"$A\",\"soc_model\":$S,\"cores\":[{\"perf_profile\":\"burst\",\"rpc_control_latency\":100}]}]}' > out_$A/htp_config_$N.json
echo '{\"backend_extensions\":{\"shared_library_path\":\"libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"/w/out_$A/htp_config_$N.json\"}}' > out_$A/htp_backend_$N.json
[ -f $N.dlc ] || qairt-converter --onnx_no_simplification --input_network $N.onnx --output_path $N.dlc --float_bitwidth 16 ${EXTRA:-} 2>&1 | grep -iE 'rror' | head -5 || true
rm -f out_$A/$N.bin
qnn-context-binary-generator --dlc_path $N.dlc --model /qairt/lib/x86_64-linux-clang/libQnnModelDlc.so --backend /qairt/lib/x86_64-linux-clang/libQnnHtp.so \
  --output_dir out_$A --binary_file $N --config_file out_$A/htp_backend_$N.json --log_level warn 2>&1 | grep -E 'ERROR|rror' | head -5 || true
ls -la out_$A/$N.bin"
