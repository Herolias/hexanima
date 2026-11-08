#!/bin/bash
# Builds the fork's APK without the Qualcomm QNN SDK and without the phone:
#   1. animad and megarun (the only native helpers this fork changes) in the public Hexagon SDK + NDK container
#      (ghcr.io/snapdragon-toolchain/arm64-android:v0.8; podman or docker), linking megarun against a link-only libcdsprpc.so stub
#      (the real one is /vendor/lib64/libcdsprpc.so on the phone, loaded at run time by its soname);
#   2. every other native library (QNN runtime, sd-cli, DSP skel, MOSS helpers) is taken unchanged from the upstream release APK (UPSTREAM_TAG);
#   3. the APK is built and signed with the host Android SDK (build-tools 36: the d8 of 34 crashes on classes from JDK 21+; platform 34) and JDK 17+.
# usage: tools/build_fork.sh [--install]      env: SDK (default ~/Android/Sdk), KS (keystore, default ~/.android/debug.keystore), UPSTREAM_TAG (v1.3.0)
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
ls -la $O/megarun $O/animad'

# ---- 2. stage the native libraries ----------------------------------------------------------------------------------
APK=$OUT/upstream/HexAnima-${TAG#v}.apk
[ -f "$APK" ] || gh release download "$TAG" -R ayanoby/hexanima -p "HexAnima-*.apk" -D "$OUT/upstream" --clobber
J=$ROOT/app/jnilibs/lib/arm64-v8a; rm -rf "$ROOT/app/jnilibs"; mkdir -p "$J"
(cd "$OUT/upstream" && rm -rf x && mkdir x && cd x && unzip -q "$APK" 'lib/arm64-v8a/*' && cp lib/arm64-v8a/*.so "$J/")
cp "$OUT/megarun" "$J/libmegarun.so"; cp "$OUT/animad" "$J/libanimad.so"
ls -la "$J"

# ---- 3. APK -----------------------------------------------------------------------------------------------------------
cd "$ROOT/app"; cp ../phone/anima.sh assets/anima.sh; cp ../phone/moss.sh assets/moss.sh
rm -rf out; mkdir -p out/cls
"$BT/aapt2" compile --dir res -o out/res.zip
"$BT/aapt2" link -o out/base.apk -I "$AJ" -A assets --manifest AndroidManifest.xml --min-sdk-version 29 --target-sdk-version 34 out/res.zip
javac -encoding UTF-8 --release 11 -nowarn -classpath "$AJ" -d out/cls src/com/hexanima/npu/*.java
"$BT/d8" --lib "$AJ" --min-api 29 --output out out/cls/com/hexanima/npu/*.class
cd out && cp base.apk unsigned.apk && "$BT/aapt" add unsigned.apk classes.dex >/dev/null && cd ..
(cd jnilibs && "$BT/aapt" add -0 so ../out/unsigned.apk $(find lib -type f | sort) >/dev/null)
"$BT/zipalign" -f -p 4 out/unsigned.apk out/aligned.apk
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --out "$OUT/hexanima-fork.apk" out/aligned.apk
rm -rf out; ls -l "$OUT/hexanima-fork.apk"
[ "${1:-}" = --install ] && "$SDK/platform-tools/adb" install -r "$OUT/hexanima-fork.apk"
exit 0
