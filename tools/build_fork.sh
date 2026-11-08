#!/bin/bash
# Builds the fork's APK without the Qualcomm QNN SDK and without the phone:
#   1. the native parts this fork changes, in the public Hexagon SDK + NDK container (ghcr.io/snapdragon-toolchain/arm64-android:v0.8; podman or docker):
#      the DSP skels libmega_skel.so (Anima) and libmossskel.so (MOSS-Image) (v69 code for every generation, the HMX accumulator store picked at run time:
#      legacy on V69..V79, the v73+ cvt path on V81),
#      animad, megarun, mossrun and lorafold (linked against a link-only libcdsprpc.so stub; the real one is /vendor/lib64/libcdsprpc.so on the phone, loaded by its soname),
#      and the NPU test programs (build_fork/tests: mega_test, gtest, ktest, ...);
#   2. every other native library (QNN 2.28 runtime for V69, sd-cli, MOSS helpers) is taken unchanged from the upstream release APK (UPSTREAM_TAG);
#      with QAIRT set (Qualcomm AI Engine Direct SDK 2.43+, not redistributable: never publish the resulting APK) its V81 runtime is added as
#      libQnnHtp243.so / libQnnSystem243.so / libQnnHtpV81Stub.so / libQnnHtpV81Skel.so (MOSS-Image decoder on Hexagon V81);
#   3. the APK is built and signed with the host Android SDK (build-tools 36: the d8 of 34 crashes on classes from JDK 21+; platform 34) and JDK 17+.
# usage: tools/build_fork.sh [--install]      env: DEBUGGABLE=1 (adb run-as access), SDK (default ~/Android/Sdk), KS (keystore, default ~/.android/debug.keystore), UPSTREAM_TAG (v1.3.0),
#                                             QAIRT (SDK root with lib/aarch64-android and lib/hexagon-v81; empty = no V81 MOSS decoder)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD; OUT=$ROOT/build_fork; mkdir -p "$OUT/syslib" "$OUT/upstream"
IMG=ghcr.io/snapdragon-toolchain/arm64-android:v0.8
CR=$(command -v podman || command -v docker)
SDK=${SDK:-$HOME/Android/Sdk}; BT=$SDK/build-tools/${BUILD_TOOLS:-36.0.0}; AJ=$SDK/platforms/android-34/android.jar
KS=${KS:-$HOME/.android/debug.keystore}; TAG=${UPSTREAM_TAG:-v1.3.0}

# ---- 1. native helpers ------------------------------------------------------------------------------------------------
cat > "$OUT/syslib/cdsprpc_stub.c" <<'EOF'
/* link-time stand-in for the phone's /vendor/lib64/libcdsprpc.so (same soname; never shipped) */
int rpcmem_init(void) { return 0; } int rpcmem_deinit(void) { return 0; } void *rpcmem_alloc(int a, unsigned b, int c) { return 0; } void rpcmem_free(void *p) {}
int rpcmem_to_fd(void *p) { return -1; } int remote_session_control(unsigned r, void *d, int l) { return -1; }
int fastrpc_mmap(int d, int fd, void *a, int o, unsigned long l, int f) { return -1; } int fastrpc_munmap(int d, int fd, void *a, unsigned long l) { return -1; }
int remote_handle64_open(const char *n, unsigned long long *h) { return -1; } int remote_handle64_close(unsigned long long h) { return -1; }
int remote_handle64_invoke(unsigned long long h, unsigned s, void *a) { return -1; }
int remote_handle_open(const char *n, unsigned *h) { return -1; } int remote_handle_close(unsigned h) { return -1; } int remote_handle_invoke(unsigned h, unsigned s, void *a) { return -1; }
void HAP_debug_v2(int l, const char *f, int n, const char *fmt, ...) {} void HAP_debug_runtime(int l, const char *f, int n, const char *fmt, ...) {}
EOF
"$CR" run --rm --platform linux/amd64 -v "$ROOT":/work:z "$IMG" bash -c '
set -e; S=/opt/hexagon/6.7.0.0; NDK=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin; O=/work/build_fork
mkdir -p /work/hexagon/mega/out
$S/ipc/fastrpc/qaic/bin/qaic -mdll -I$S/incs/stddef -I$S/incs -I$S/ipc/fastrpc/incs /work/hexagon/mega/mega.idl -o /work/hexagon/mega/out
$NDK/aarch64-linux-android31-clang -shared -fPIC -Wl,-soname,libcdsprpc.so -w $O/syslib/cdsprpc_stub.c -o $O/syslib/libcdsprpc.so
INC="-w -I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/ipc/fastrpc/rpcmem/inc -I/work/hexagon/mega/out -I/work/hexagon/mega"
$NDK/aarch64-linux-android31-clang -O2 $INC -c /work/hexagon/mega/out/mega_stub.c -o $O/mega_stub.o
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod $INC /work/hexagon/mega/megarun.cpp $O/mega_stub.o -L$O/syslib -lcdsprpc -llog -static-libstdc++ -o $O/megarun
$NDK/aarch64-linux-android29-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod /work/phone/animad.cpp -o $O/animad -llog -static-libstdc++
$NDK/llvm-strip $O/megarun $O/animad
mkdir -p $O/tests; M=/work/hexagon/mega
for t in mega_test:mega_test gtest_main:gtest ktest_main:ktest bench_main:bench stream_main:stream copy_main:copybw; do $NDK/aarch64-linux-android31-clang -O2 $INC $M/${t%%:*}.c $O/mega_stub.o -L$O/syslib -lcdsprpc -o $O/tests/${t##*:}; done
ARCH=v69 bash /work/hexagon/mega/build_arch.sh
ARCH=v69 bash /work/moss/hexagon/mega/build_arch.sh
MO=/work/moss/hexagon/mega/out_v69; MI="-w -I$S/incs -I$S/incs/stddef -I$S/ipc/fastrpc/incs -I$S/ipc/fastrpc/rpcmem/inc -I$MO -I/work/moss/hexagon/mega"
$NDK/aarch64-linux-android31-clang -O2 $MI -c $MO/mega_stub.c -o $O/moss_stub.o
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod $MI /work/moss/mossrun/mossrun.cpp $O/moss_stub.o -L$O/syslib -lcdsprpc -llog -static-libstdc++ -o $O/mossrun
$NDK/aarch64-linux-android31-clang++ -std=c++17 -O3 -ffast-math -march=armv8.2-a+fp16+dotprod -static-libstdc++ /work/moss/mossrun/lorafold.cpp -o $O/lorafold
$NDK/llvm-strip $O/mossrun $O/lorafold
ls -la $O/megarun $O/animad'

# ---- 2. stage the native libraries ----------------------------------------------------------------------------------
APK=$OUT/upstream/HexAnima-${TAG#v}.apk
[ -f "$APK" ] || gh release download "$TAG" -R ayanoby/hexanima -p "HexAnima-*.apk" -D "$OUT/upstream" --clobber
J=$ROOT/app/jnilibs/lib/arm64-v8a; rm -rf "$ROOT/app/jnilibs"; mkdir -p "$J"
(cd "$OUT/upstream" && rm -rf x && mkdir x && cd x && unzip -q "$APK" 'lib/arm64-v8a/*' && cp lib/arm64-v8a/*.so "$J/")
cp "$OUT/megarun" "$J/libmegarun.so"; cp "$OUT/animad" "$J/libanimad.so"; cp "$ROOT/hexagon/mega/out_v69/libmega_skel.so" "$J/libmega_skel.so"
cp "$ROOT/moss/hexagon/mega/out_v69/libmega_skel.so" "$J/libmossskel.so"; cp "$OUT/mossrun" "$J/libmossrun.so"; cp "$OUT/lorafold" "$J/liblorafold.so"
QAIRT=${QAIRT-$HOME/anima-npu/sdk/qairt/qairt/2.43.0.260128}
if [ -n "$QAIRT" ] && [ -d "$QAIRT/lib/hexagon-v81" ]; then
  cp "$QAIRT/lib/aarch64-android/libQnnHtp.so" "$J/libQnnHtp243.so"; cp "$QAIRT/lib/aarch64-android/libQnnSystem.so" "$J/libQnnSystem243.so"
  cp "$QAIRT/lib/aarch64-android/libQnnHtpV81Stub.so" "$QAIRT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "$J/"
else echo "no QAIRT 2.43: MOSS-Image will not decode on Hexagon V81"; fi
ls -la "$J"

# ---- 3. APK -----------------------------------------------------------------------------------------------------------
cd "$ROOT/app"; cp ../phone/anima.sh assets/anima.sh; cp ../phone/moss.sh assets/moss.sh
rm -rf out; mkdir -p out/cls
"$BT/aapt2" compile --dir res -o out/res.zip
"$BT/aapt2" link -o out/base.apk -I "$AJ" ${DEBUGGABLE:+--debug-mode} -A assets --manifest AndroidManifest.xml --min-sdk-version 29 --target-sdk-version 34 out/res.zip
javac -encoding UTF-8 --release 11 -nowarn -classpath "$AJ" -d out/cls src/com/hexanima/npu/*.java
"$BT/d8" --lib "$AJ" --min-api 29 --output out out/cls/com/hexanima/npu/*.class
cd out && cp base.apk unsigned.apk && "$BT/aapt" add unsigned.apk classes.dex >/dev/null && cd ..
# native libraries stored uncompressed and 16 KB page-aligned (Android 15+ 16 KB page-size check; aapt add ignores -0)
python3 -c 'import sys, zipfile, os; z = zipfile.ZipFile(sys.argv[1], "a"); [z.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), "jnilibs"), zipfile.ZIP_STORED) for r, _, fs in sorted(os.walk("jnilibs/lib")) for f in sorted(fs)]; z.close()' out/unsigned.apk
"$BT/zipalign" -f -P 16 4 out/unsigned.apk out/aligned.apk
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --out "$OUT/hexanima-fork.apk" out/aligned.apk
rm -rf out; ls -l "$OUT/hexanima-fork.apk"
[ "${1:-}" = --install ] && "$SDK/platform-tools/adb" install -r "$OUT/hexanima-fork.apk"
exit 0
