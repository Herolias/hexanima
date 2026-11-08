#!/bin/bash
# MOSS DSP skel inside the Hexagon SDK container, repo at /work:  ARCH=v69 bash /work/moss/hexagon/mega/build_arch.sh -> out_v69/libmega_skel.so (+ ARM test programs if /work/build_fork exists)
# The v69 build serves every generation: the HMX accumulator store is chosen at run time (gemm2.c: legacy on V69..V79, the v73+ cvt path on V81).
set -e
ARCH=${ARCH:-v69}; S=/opt/hexagon/6.7.0.0; T=$S/tools/HEXAGON_Tools/21.0.04/Tools; Q=$S/libs/qhl_hmx
cd /work/moss/hexagon/mega; O=out_$ARCH; mkdir -p $O
$S/ipc/fastrpc/qaic/bin/qaic -mdll -I$S/incs/stddef -I$S/incs -I$S/ipc/fastrpc/incs mega.idl -o $O
INC="-I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/rtos/qurt/compute$ARCH/include/qurt -I$S/rtos/qurt/compute$ARCH/include/posix -I$O -I$Q/inc/qhblas_hmx -I$S/libs/qhl_hvx/inc/internal -I$S/utils/examples"
HF="-G0 -m$ARCH -mhvx -mhvx-length=128B -mhmx -O2 -fPIC -DUSE_OS_QURT -D_HAS_EXCEPTIONS=0 -w $INC"
for f in mega_imp.c gemm.c stream.c copybw.c powermax.c kern.c ktest.c pool.c gemm2.c gtest.c blk.c $O/mega_skel.c $Q/src/qhblas_hmx/*.c $Q/src/qhblas_hmx/*.S hmx8.S hmxacc.S; do
  $T/bin/hexagon-clang $HF -c $f -o $O/$(basename $f).o
done
$T/bin/hexagon-clang -G0 -m$ARCH -shared -fPIC -Wl,-Bsymbolic -Wl,-z,max-page-size=16384 -nodefaultlibs -o $O/libmega_skel.so $O/*.o -Wl,--start-group $T/target/hexagon/lib/$ARCH/G0/pic/libgcc.a -Wl,--end-group
if [ -d /work/build_fork/syslib ]; then
  A=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin; L=/work/build_fork/syslib; TO=/work/build_fork/moss_tests; mkdir -p $TO
  CI="-O2 -w -I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/ipc/fastrpc/rpcmem/inc -I$O"
  for t in mega_test:mega_test ktest_main:ktest gtest_main:gtest blk_main:blk; do $A/aarch64-linux-android31-clang $CI ${t%%:*}.c $O/mega_stub.c -L$L -lcdsprpc -o $TO/${t##*:}; done
fi
ls -la $O/libmega_skel.so
