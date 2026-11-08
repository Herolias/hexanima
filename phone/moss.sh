#!/system/bin/sh
# MOSS-Image on the NPU (text encoder, refiner, DiT, VAE): one picture per call, weights are streamed from flash.
#   moss.sh IDS_FILE W H SEED STEPS OUT_RAW      tokens (int32) -> raw fp32 RGB [3][H][W] in [-1,1]; exit 0 on success
#   moss.sh stop                                  kill the helpers
# env: M (model dir, files/moss: glue.bin embd.q4k te/ ref/ dit/ vae/vae_<W>x<H>.bin), BIN (the app's native lib dir),
#      VAED (decoder contexts, default $M/vae; files/vae81 on Hexagon V81), QH / QS (QNN HTP / System runtime in $BIN: libQnnHtp243.so / libQnnSystem243.so on V81),
#      MOSS_FULL ("full CFG NEG_IDS_FILE [SHIFT]": the full model with the official sampler instead of the 4-step adapter; STEPS up to 100, needs $M/glue_base.bin)
# The MOSS DSP skel has the same file name as Anima's (libmega_skel.so): it is copied to $M/dsp and that dir comes first in ADSP_LIBRARY_PATH.
if [ "$1" = stop ]; then for p in libmossrun.so libvaerun.so libvaetile.so; do pidof $p >/dev/null && kill -9 $(pidof $p); done; exit 0; fi
IDS=$1 W=$2 H=$3 SEED=$4 STEPS=$5 OUT=$6
VAED=${VAED:-$M/vae} QH=${QH:-libQnnHtp.so} QS=${QS:-libQnnSystem.so}
[ -f $M/tune.env ] && { set -a; . $M/tune.env; set +a; }   # developer overrides for mossrun (MOSS_NW, MOSS_KEEP, ...), one VAR=value per line
LIBS=$BIN:/system/lib64:/vendor/lib64; ADSPP="$M/dsp;$BIN"
RUNX='export LD_LIBRARY_PATH="$LIBS" ADSP_LIBRARY_PATH="$ADSPP"; exec "$@"'
mkdir -p $M/dsp && cp -f $BIN/libmossskel.so $M/dsp/libmega_skel.so
rm -f $M/moss.log $M/z.raw $OUT
printf 'load %s\ngen %s %s %s %s %s %s/z.raw %s\nquit\n' "$M" "$IDS" "$H" "$W" "$SEED" "$STEPS" "$M" "${MOSS_FULL:-}" | LIBS=$LIBS ADSPP=$ADSPP sh -c "$RUNX" sh $BIN/libmossrun.so > $M/moss.log 2>&1
[ -s $M/z.raw ] || exit 1
if [ -f $VAED/vae_${W}x${H}.bin ]; then   # a full-frame decoder context exists for this exact size (best fidelity)
  printf 'run %s/z.raw %s\nquit\n' "$M" "$OUT" | LIBS=$LIBS ADSPP=$ADSPP sh -c "$RUNX" sh $BIN/libvaerun.so $BIN/$QH $BIN/$QS $VAED/vae_${W}x${H}.bin >> $M/moss.log 2>&1
else   # any other size: 512 px tiles of the 512x512 context, blended (GroupNorm statistics per tile: a slight smooth tone shift, ~31 dB vs full frame)
  printf 'run %s/z.raw %s %s %s\nquit\n' "$M" "$W" "$H" "$OUT" | LIBS=$LIBS ADSPP=$ADSPP sh -c "$RUNX" sh $BIN/libvaetile.so $BIN/$QH $BIN/$QS $VAED/vae_512x512.bin >> $M/moss.log 2>&1
fi
[ -s $OUT ]
