#!/bin/bash
# DSP skel for one Hexagon generation, inside the Hexagon SDK container (ghcr.io/snapdragon-toolchain/arm64-android:v0.8), repo mounted at /work:
#   ARCH=v81 bash /work/hexagon/mega/build_arch.sh   -> /work/hexagon/mega/out_v81/libmega_skel.so
# Uses the SDK's own QHL HMX sources (libs/qhl_hmx), so nothing proprietary has to be copied into the repository.
# v73+ builds read the HMX accumulator through the cvt path (gemm2.c: hmx_store / hmx_scales_init); the v69 build keeps the legacy store.
set -e
ARCH=${ARCH:-v81}; S=/opt/hexagon/6.7.0.0; T=$S/tools/HEXAGON_Tools/21.0.04/Tools; Q=$S/libs/qhl_hmx
cd /work/hexagon/mega; O=out_$ARCH; mkdir -p $O
$S/ipc/fastrpc/qaic/bin/qaic -mdll -I$S/incs/stddef -I$S/incs -I$S/ipc/fastrpc/incs mega.idl -o $O
INC="-I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/rtos/qurt/compute$ARCH/include/qurt -I$S/rtos/qurt/compute$ARCH/include/posix -I$O -I$Q/inc/qhblas_hmx -I$S/libs/qhl_hvx/inc/internal -I$S/utils/examples"
HF="-G0 -m$ARCH -mhvx -mhvx-length=128B -mhmx -O2 -fPIC -DUSE_OS_QURT -D_HAS_EXCEPTIONS=0 -w $INC"
for f in mega_imp.c gemm.c stream.c copybw.c powermax.c kern.c ktest.c pool.c gemm2.c gtest.c blk.c $O/mega_skel.c $Q/src/qhblas_hmx/*.c $Q/src/qhblas_hmx/*.S hmx8.S hmxacc.S; do
  $T/bin/hexagon-clang $HF -c $f -o $O/$(basename $f).o
done
$T/bin/hexagon-clang -G0 -m$ARCH -shared -fPIC -Wl,-Bsymbolic -Wl,-z,max-page-size=16384 -nodefaultlibs -o $O/libmega_skel.so $O/*.o -Wl,--start-group $T/target/hexagon/lib/$ARCH/G0/pic/libgcc.a -Wl,--end-group
ls -la $O/libmega_skel.so
