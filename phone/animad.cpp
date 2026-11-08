#include <sys/resource.h>
#include <sys/mman.h>
// animad: on-phone DiT server for the sd.cpp hook (SD_NPU_MODE=extern). Per step it reads x_<n>/t_<n>/ctx_<n>/ids_<n>/w_<n> from DIR, runs
// prelude (x_embedder, timestep MLPs, LLM adapter once) and final layer on the phone CPU and the 28 blocks via three chainrun processes (NPU),
// then writes out_<n>.bin + out_<n>.bin.done. usage: animad SMALL.bin DIR [STEPS] [PH=/data/local/tmp/anima]
//        animad SMALL.bin DIR --adapter N    (test: writes DIR/ctx_adapter.bin = adapter output of step N, no NPU)
#include <arm_neon.h>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <sys/stat.h>
#include <sys/wait.h>
#include <csignal>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

using f32 = float;
static const int D = 2048, NH = 16, NTHREADS = 8;

struct Tn {
    std::vector<int64_t> d;
    const uint16_t* p;
};
static std::map<std::string, Tn> W;
static std::vector<char> blob;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void die(const char* m, const std::string& s = "") { fprintf(stderr, "animad: %s %s\n", m, s.c_str()); exit(1); }

static void load_small(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open", path);
    fseek(f, 0, SEEK_END);
    size_t n = ftell(f);
    fseek(f, 0, SEEK_SET);
    blob.resize(n);
    if (fread(blob.data(), 1, n, f) != n) die("short read");
    fclose(f);
    const char* q = blob.data();
    uint32_t cnt;
    memcpy(&cnt, q, 4);
    q += 4;
    for (uint32_t i = 0; i < cnt; i++) {
        uint32_t nl, nd;
        memcpy(&nl, q, 4); q += 4;
        std::string name(q, nl); q += nl;
        memcpy(&nd, q, 4); q += 4;
        Tn t; t.d.resize(nd);
        memcpy(t.d.data(), q, 8 * nd); q += 8 * nd;
        int64_t el = 1;
        for (auto v : t.d) el *= v;
        t.p = (const uint16_t*)q;
        q += 2 * el;
        W[name] = t;
    }
}
static const Tn& w(const std::string& k) {
    auto it = W.find(k);
    if (it == W.end()) die("missing weight", k);
    return it->second;
}

static void par(int n, const std::function<void(int, int)>& fn) {
    int nt = std::min(NTHREADS, n);
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++) th.emplace_back([=, &fn] { fn(n * t / nt, n * (t + 1) / nt); });
    for (auto& t : th) t.join();
}

// C[M,N] = A[M,K] * W[N,K]^T (+ bias[N]); W fp16
static void linear(const f32* A, int M, int K, const Tn& Wt, const Tn* bias, f32* C) {
    int N = (int)Wt.d[0];
    if ((int)Wt.d[1] != K) die("linear K mismatch");
    std::vector<f32> zero(K, 0.f);
    par(N, [&](int n0, int n1) {
        for (int n = n0; n < n1; n++) {
            const __fp16* wr = (const __fp16*)Wt.p + (size_t)n * K;
            f32 b = bias ? (f32)((const __fp16*)bias->p)[n] : 0.f;
            for (int m = 0; m < M; m += 4) {
                const f32* a[4];
                for (int r = 0; r < 4; r++) a[r] = (m + r < M) ? A + (size_t)(m + r) * K : zero.data();
                float32x4_t acc[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
                int k = 0;
                for (; k + 8 <= K; k += 8) {
                    float16x8_t wv = vld1q_f16(wr + k);
                    float32x4_t lo = vcvt_f32_f16(vget_low_f16(wv)), hi = vcvt_f32_f16(vget_high_f16(wv));
                    for (int r = 0; r < 4; r++) {
                        acc[r] = vfmaq_f32(acc[r], vld1q_f32(a[r] + k), lo);
                        acc[r] = vfmaq_f32(acc[r], vld1q_f32(a[r] + k + 4), hi);
                    }
                }
                for (int r = 0; r < 4 && m + r < M; r++) {
                    f32 s = vaddvq_f32(acc[r]);
                    for (int kk = k; kk < K; kk++) s += a[r][kk] * (f32)wr[kk];
                    C[(size_t)(m + r) * N + n] = s + b;
                }
            }
        }
    });
}

static void rms_rows(f32* x, int rows, int dim, const Tn& g, f32 eps = 1e-6f) {
    const __fp16* gp = (const __fp16*)g.p;
    for (int r = 0; r < rows; r++) {
        f32* v = x + (size_t)r * dim;
        double s = 0;
        for (int i = 0; i < dim; i++) s += (double)v[i] * v[i];
        f32 k = 1.f / sqrtf((f32)(s / dim) + eps);
        for (int i = 0; i < dim; i++) v[i] = v[i] * k * (f32)gp[i];
    }
}
static void rms_into(const f32* x, f32* o, int rows, int dim, const Tn& g) {
    memcpy(o, x, sizeof(f32) * rows * dim);
    rms_rows(o, rows, dim, g);
}

// cos/sin for split-half rope, dim 64 -> [n,32]
static void rope1d(int n, std::vector<f32>& c, std::vector<f32>& s) {
    c.resize(n * 32); s.resize(n * 32);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 32; j++) {
            double om = 1.0 / pow(10000.0, (2.0 * j) / 64.0), a = i * om;
            c[i * 32 + j] = (f32)cos(a); s[i * 32 + j] = (f32)sin(a);
        }
}
static void rope_apply(f32* t, int n, const std::vector<f32>& c, const std::vector<f32>& s) {  // t [n,16,64]
    for (int i = 0; i < n; i++)
        for (int h = 0; h < NH; h++) {
            f32* v = t + ((size_t)i * NH + h) * 64;
            for (int j = 0; j < 32; j++) {
                f32 a = v[j], b = v[j + 32], cc = c[i * 32 + j], ss = s[i * 32 + j];
                v[j] = a * cc - b * ss; v[j + 32] = a * ss + b * cc;
            }
        }
}
// q [T,1024], k/v [S,1024] -> o [T,1024]
static void attention(const f32* q, const f32* k, const f32* v, int T, int S, f32* o) {
    par(NH * T, [&](int i0, int i1) {
        std::vector<f32> sc(S);
        for (int idx = i0; idx < i1; idx++) {
            int h = idx / T, t = idx % T;
            const f32* qq = q + (size_t)t * 1024 + h * 64;
            f32 mx = -1e30f;
            for (int s = 0; s < S; s++) {
                const f32* kk = k + (size_t)s * 1024 + h * 64;
                float32x4_t a = vdupq_n_f32(0);
                for (int j = 0; j < 64; j += 4) a = vfmaq_f32(a, vld1q_f32(qq + j), vld1q_f32(kk + j));
                sc[s] = vaddvq_f32(a) * 0.125f;
                mx = std::max(mx, sc[s]);
            }
            f32 sum = 0;
            for (int s = 0; s < S; s++) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
            f32* oo = o + (size_t)t * 1024 + h * 64;
            float32x4_t acc[16];
            for (int j = 0; j < 16; j++) acc[j] = vdupq_n_f32(0);
            for (int s = 0; s < S; s++) {
                float32x4_t p = vdupq_n_f32(sc[s] / sum);
                const f32* vv = v + (size_t)s * 1024 + h * 64;
                for (int j = 0; j < 16; j++) acc[j] = vfmaq_f32(acc[j], p, vld1q_f32(vv + 4 * j));
            }
            for (int j = 0; j < 16; j++) vst1q_f32(oo + 4 * j, acc[j]);
        }
    });
}

static std::vector<f32> adapter(const f32* hidden, int S, const int32_t* ids, const f32* wts, int L) {
    std::vector<f32> x((size_t)L * 1024), h(x.size()), q(x.size()), k((size_t)std::max(L, S) * 1024), v(k.size()), o(x.size());
    const __fp16* emb = (const __fp16*)w("llm_adapter.embed.weight").p;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < 1024; j++) x[(size_t)i * 1024 + j] = (f32)emb[(size_t)ids[i] * 1024 + j];
    std::vector<f32> tcos, tsin, scos, ssin;
    rope1d(L, tcos, tsin); rope1d(S, scos, ssin);
    auto attn = [&](const std::string& p, const f32* xq, const f32* xkv, int nkv, bool cross) {
        linear(xq, L, 1024, w(p + ".q_proj.weight"), nullptr, q.data());
        linear(xkv, nkv, 1024, w(p + ".k_proj.weight"), nullptr, k.data());
        linear(xkv, nkv, 1024, w(p + ".v_proj.weight"), nullptr, v.data());
        rms_rows(q.data(), L * NH, 64, w(p + ".q_norm.weight"));
        rms_rows(k.data(), nkv * NH, 64, w(p + ".k_norm.weight"));
        rope_apply(q.data(), L, tcos, tsin);
        rope_apply(k.data(), nkv, cross ? scos : tcos, cross ? ssin : tsin);
        attention(q.data(), k.data(), v.data(), L, nkv, o.data());
        std::vector<f32> r((size_t)L * 1024);
        linear(o.data(), L, 1024, w(p + ".o_proj.weight"), nullptr, r.data());
        for (size_t i = 0; i < r.size(); i++) x[i] += r[i];
    };
    for (int b = 0; b < 6; b++) {
        std::string p = "llm_adapter.blocks." + std::to_string(b);
        rms_into(x.data(), h.data(), L, 1024, w(p + ".norm_self_attn.weight"));
        attn(p + ".self_attn", h.data(), h.data(), L, false);
        rms_into(x.data(), h.data(), L, 1024, w(p + ".norm_cross_attn.weight"));
        attn(p + ".cross_attn", h.data(), hidden, S, true);
        rms_into(x.data(), h.data(), L, 1024, w(p + ".norm_mlp.weight"));
        std::vector<f32> m1((size_t)L * 4096), m2((size_t)L * 1024);
        linear(h.data(), L, 1024, w(p + ".mlp.0.weight"), &w(p + ".mlp.0.bias"), m1.data());
        for (auto& e : m1) e = 0.5f * e * (1.f + erff(e * 0.70710678f));
        linear(m1.data(), L, 4096, w(p + ".mlp.2.weight"), &w(p + ".mlp.2.bias"), m2.data());
        for (size_t i = 0; i < m2.size(); i++) x[i] += m2[i];
    }
    linear(x.data(), L, 1024, w("llm_adapter.out_proj.weight"), &w("llm_adapter.out_proj.bias"), h.data());
    rms_rows(h.data(), L, 1024, w("llm_adapter.norm.weight"));
    for (int i = 0; i < L; i++)
        for (int j = 0; j < 1024; j++) h[(size_t)i * 1024 + j] *= wts[i];
    h.resize((size_t)512 * 1024, 0.f);  // zero padded to 512 tokens
    return h;
}

static std::vector<char> readfile(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) die("cannot open", p);
    fseek(f, 0, SEEK_END);
    size_t n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> b(n);
    if (fread(b.data(), 1, n, f) != n) die("short read", p);
    fclose(f);
    return b;
}
static std::vector<int64_t> readshape(const std::string& p) {
    std::vector<int64_t> d;
    FILE* f = fopen(p.c_str(), "r");
    if (!f) die("cannot open", p);
    long long v;
    while (fscanf(f, "%lld", &v) == 1) d.push_back(v);
    fclose(f);
    return d;
}
static void writefile(const std::string& p, const void* d, size_t n) {
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) die("cannot write", p);
    fwrite(d, 1, n, f);
    fclose(f);
}

struct Chain { FILE* in; FILE* out; pid_t pid; };
static Chain spawn(const std::string& ph, int first, int count, const std::string& bdir) {
    int pin[2], pout[2];
    if (pipe(pin) || pipe(pout)) die("pipe");
    pid_t pid = fork();
    if (pid == 0) {
        dup2(pin[0], 0); dup2(pout[1], 1); dup2(pout[1], 2);
        close(pin[1]); close(pout[0]);
        if (chdir(ph.c_str())) _exit(127);
        if (!getenv("ANIMAD_BIN")) { setenv("LD_LIBRARY_PATH", ".", 1); setenv("ADSP_LIBRARY_PATH", ".", 1); }   /* app: inherited, they point at the native lib dir */
        std::string a = std::to_string(first), b = std::to_string(count);
        execl("./chainrun", "chainrun", "libQnnHtp.so", "libQnnSystem.so", "libQnnRmsRopePkg_v69.so:RmsRopePkgInterfaceProvider", bdir.c_str(), a.c_str(), b.c_str(), (char*)nullptr);
        _exit(127);
    }
    close(pin[0]); close(pout[1]);
    return {fdopen(pin[1], "w"), fdopen(pout[0], "r"), pid};
}
// helper executables: ./name in the work dir (root setup) or $ANIMAD_BIN/libname.so (the app ships them as native libs, the only place an app may exec from)
static std::string binpath(const char* n) { const char* d = getenv("ANIMAD_BIN"); return d ? std::string(d) + "/lib" + n + ".so" : std::string("./") + n; }
static Chain spawn_mega(const std::string& ph, int first, int count, const std::string& hd = "hmx") {   // HMX block runner (hexagon/mega): all 28 blocks in one process, fp16 weights from ph/hmx
    int pin[2], pout[2];
    if (pipe(pin) || pipe(pout)) die("pipe");
    pid_t pid = fork();
    if (pid == 0) {
        dup2(pin[0], 0); dup2(pout[1], 1); dup2(pout[1], 2);
        close(pin[1]); close(pout[0]);
        if (chdir(ph.c_str())) _exit(127);
        if (!getenv("ANIMAD_BIN")) { setenv("LD_LIBRARY_PATH", ".", 1); setenv("ADSP_LIBRARY_PATH", ".", 1); }   /* app: inherited, they point at the native lib dir */
        std::string a = std::to_string(first), b = std::to_string(count);
        execl(binpath("megarun").c_str(), "megarun", hd.c_str(), a.c_str(), b.c_str(), (char*)nullptr);
        _exit(127);
    }
    close(pin[0]); close(pout[1]);
    return {fdopen(pin[1], "w"), fdopen(pout[0], "r"), pid};
}
static Chain spawn_vae(const std::string& ph) {
    int pin[2], pout[2];
    if (pipe(pin) || pipe(pout)) die("pipe");
    pid_t pid = fork();
    if (pid == 0) {
        dup2(pin[0], 0); dup2(pout[1], 1); dup2(pout[1], 2);
        close(pin[1]); close(pout[0]);
        if (chdir(ph.c_str())) _exit(127);
        if (!getenv("ANIMAD_BIN")) { setenv("LD_LIBRARY_PATH", ".", 1); setenv("ADSP_LIBRARY_PATH", ".", 1); }   /* app: inherited, they point at the native lib dir */
        execl(binpath("vaerun").c_str(), "vaerun", "libQnnHtp.so", "libQnnSystem.so", "vae.bin", (char*)nullptr);
        _exit(127);
    }
    close(pin[0]); close(pout[1]);
    return {fdopen(pin[1], "w"), fdopen(pout[0], "r"), pid};
}
static Chain spawn_te(const std::string& ph) {
    int pin[2], pout[2];
    if (pipe(pin) || pipe(pout)) die("pipe");
    pid_t pid = fork();
    if (pid == 0) {
        dup2(pin[0], 0); dup2(pout[1], 1); dup2(pout[1], 2);
        close(pin[1]); close(pout[0]);
        if (chdir(ph.c_str())) _exit(127);
        if (!getenv("ANIMAD_BIN")) { setenv("LD_LIBRARY_PATH", ".", 1); setenv("ADSP_LIBRARY_PATH", ".", 1); }   /* app: inherited, they point at the native lib dir */
        execl(binpath("terun").c_str(), "terun", "libQnnHtp.so", "libQnnSystem.so", "te0.bin", "te1.bin", "te2.bin", "te3.bin", (char*)nullptr);
        _exit(127);
    }
    close(pin[0]); close(pout[1]);
    return {fdopen(pin[1], "w"), fdopen(pout[0], "r"), pid};
}
static std::string waitline(Chain& c, const char* a, const char* b) {
    char line[4096];
    while (fgets(line, sizeof line, c.out)) {
        if (!strncmp(line, a, strlen(a)) || (b && !strncmp(line, b, strlen(b)))) return line;
        if (strstr(line, "ERROR") || strstr(line, "failed")) fprintf(stderr, "chainrun: %s", line);
    }
    die("chainrun exited");
    return "";
}
// like waitline(c, "ready"), but a helper that exits or reports an error returns false instead of ending animad (megarun: "error hmx" = HMX self-test failed)
static bool waitready(Chain& c) {
    char line[4096];
    while (fgets(line, sizeof line, c.out)) {
        if (!strncmp(line, "ready", 5)) return true;
        if (!strncmp(line, "error", 5)) { fprintf(stderr, "animad: helper: %s", line); break; }
        if (strstr(line, "ERROR") || strstr(line, "failed") || strstr(line, "FAILED")) fprintf(stderr, "chainrun: %s", line);
    }
    fclose(c.in); fclose(c.out); kill(c.pid, SIGTERM); waitpid(c.pid, nullptr, 0);
    return false;
}

int main(int argc, char** argv) {
    if (argc < 3) die("usage: animad SMALL.bin DIR [STEPS|--adapter N] [PH]");
    bool serve = !strcmp(argv[2], "--serve");  // animad SMALL.bin --serve ROOT: keeps everything loaded, handles ROOT/r*/ request dirs (file "start" holds the step count)
    std::string dir = argv[2];
    double t0 = now();
    load_small(argv[1]);
    if (!getenv("ANIMAD_NOMLOCK")) {   // 2.9B: RAM is oversubscribed (7 GB of dmabuf), keep the CPU-side weights/code resident instead of swap-thrashing every step
        struct rlimit rl = {RLIM_INFINITY, RLIM_INFINITY}; setrlimit(RLIMIT_MEMLOCK, &rl);
        if (mlockall(MCL_CURRENT)) perror("mlockall");
    }
    fprintf(stderr, "animad: weights loaded %.1fs\n", now() - t0);
    bool test = !serve && argc > 3 && !strcmp(argv[3], "--adapter");
    int steps = (argc > 3 && !test) ? atoi(argv[3]) : 4;
    std::string ph = (argc > 4 && !test && !serve) ? argv[4] : getenv("ANIMAD_HOME") ? getenv("ANIMAD_HOME") : "/data/local/tmp/anima";   // ANIMAD_HOME: the app keeps its files elsewhere
    if (serve) { dir = argc > 3 ? argv[3] : "/data/local/tmp/anima/req"; steps = 4; }
    if (test && argc > 4) steps = atoi(argv[4]);
    std::vector<Chain> ch, vae, te;
    bool hmx = false;
    const __fp16* embed = nullptr;  // Qwen3 embedding table fp16 [151936][1024] (mmap) for the NPU text encoder
    if (!test) {
        // q4/blk0..6.bin = the 28 blocks as 7 merged groups of 4 (12% faster than 28 single-block contexts); fall back to blk0..27.bin
        const char* he = getenv("ANIMAD_HMXDIR"); std::string hd = he ? he : "hmx";   // 2.9B: hmx29 (40 blocks)
        int nblk = 0; while (!access((ph + "/" + hd + (nblk < 10 ? "/blk0" : "/blk") + std::to_string(nblk) + ".bin").c_str(), F_OK)) nblk++;
        hmx = nblk >= 28 && !access(binpath("megarun").c_str(), F_OK) && !getenv("ANIMAD_NOHMX");
        bool grp = !access((ph + "/q4/blk6.bin").c_str(), F_OK) && !getenv("ANIMAD_NOGROUPS");
        if (hmx) {   // ANIMAD_HMXG="9,9,10" = blocks per megarun process (big token counts need a bigger activation buffer per process, so fewer weight buffers fit)
            std::vector<int> cnt = {10, 10, 8}; if (nblk == 40) cnt = {10, 10, 10, 10};
            if (const char* g = getenv("ANIMAD_HMXG")) { cnt.clear(); for (const char* p = g; *p;) { cnt.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; } }
            int f0 = 0; for (int c : cnt) { ch.push_back(spawn_mega(ph, f0, c, hd)); f0 += c; }
            // a megarun whose HMX self-test fails (or that dies while loading) is started again, at most twice, after a pause: a fresh process gets a
            // fresh DSP session, and a run interrupted just before may still be releasing the HMX
            f0 = 0;
            for (size_t i = 0; i < ch.size(); f0 += cnt[i], i++) {
                bool ok = waitready(ch[i]);
                for (int a = 0; !ok && a < 2; a++) { fprintf(stderr, "animad: restarting megarun %d\n", f0); usleep(1500000); ch[i] = spawn_mega(ph, f0, cnt[i], hd); ok = waitready(ch[i]); }
                if (!ok) die("megarun failed (HMX self-test or load)");
            }
        }   // one DSP process maps at most ~13 of the 126 MB weight buffers
        else if (grp) { ch.push_back(spawn(ph, 0, 4, "q4")); ch.push_back(spawn(ph, 4, 3, "q4")); }  // one DSP session holds only 4 of these (0x3ea at the 5th)
        else { int rg[3][2] = {{0, 10}, {10, 10}, {20, 8}}; for (auto& r : rg) ch.push_back(spawn(ph, r[0], r[1], ".")); }
        if (!hmx) for (auto& c : ch) waitline(c, "ready", nullptr);
        // NPU decoder only when its QNN context is installed (chips without QNN contexts, e.g. V81: sd.cpp decodes on the CPU, the app sends v=0)
        if ((serve || getenv("SD_VAE_DIR")) && !access((ph + "/vae.bin").c_str(), F_OK) && !getenv("ANIMAD_NOVAE")) { vae.push_back(spawn_vae(ph)); waitline(vae[0], "ready", nullptr); }
        if (serve && !access((ph + "/te3.bin").c_str(), F_OK) && !access((ph + "/te_embed.f16").c_str(), F_OK)) {
            int fd = open((ph + "/te_embed.f16").c_str(), O_RDONLY);
            embed = (const __fp16*)mmap(nullptr, (size_t)151936 * 1024 * 2, PROT_READ, MAP_SHARED, fd, 0);
            if (embed == MAP_FAILED) embed = nullptr;
            if (embed) { te.push_back(spawn_te(ph)); waitline(te[0], "ready", nullptr); }
        }
        fprintf(stderr, "animad: chain ready %.1fs (text encoder on NPU: %s, decoder on NPU: %s)\n", now() - t0, te.empty() ? "no" : "yes", vae.empty() ? "no" : "yes");
    }
    auto handle = [&](const std::string& dir, int steps, const std::string& vd) {
    std::vector<f32> ctx512;  // adapter output [512,1024]
    if (te.empty() && serve) writefile(dir + "/qh.cpu", "", 0);  // no NPU text encoder: sd.cpp computes it
    if (!te.empty()) {  // sd.cpp (SD_TE_MODE=extern) writes qids.bin/qids.shape (Qwen token ids) and waits for qh.bin(.done) or qh.cpu (= compute it yourself)
        while (access((dir + "/qids.shape").c_str(), F_OK) && access((dir + "/ready_0").c_str(), F_OK)) usleep(2000);
        if (!access((dir + "/qids.shape").c_str(), F_OK)) {
            double tt = now();
            auto ib = readfile(dir + "/qids.bin");
            int S = (int)(ib.size() / 4); const int SM = 128;
            if (S > SM) { writefile(dir + "/qh.cpu", "", 0); fprintf(stderr, "animad: text S=%d > %d, CPU fallback\n", S, SM); }
            else {
                std::vector<f32> in((size_t)SM * 1024);
                for (int i = 0; i < SM; i++) {
                    int id = i < S ? ((const int32_t*)ib.data())[i] : 151643;
                    const __fp16* r = embed + (size_t)id * 1024;
                    for (int c = 0; c < 1024; c++) in[(size_t)i * 1024 + c] = (f32)r[c];
                }
                writefile(ph + "/te_in.raw", in.data(), in.size() * 4);
                fprintf(te[0].in, "run %s/te_in.raw %s/te_out.raw\n", ph.c_str(), ph.c_str()); fflush(te[0].in);
                std::string r = waitline(te[0], "done", "error");
                if (r[0] != 'd') die("terun error", r);
                auto ob = readfile(ph + "/te_out.raw");
                writefile(dir + "/qh.bin", ob.data(), (size_t)S * 1024 * 4);
                writefile(dir + "/qh.bin.done", "", 0);
                fprintf(stderr, "animad: text encoder S=%d %.3fs (npu %s)\n", S, now() - tt, r.c_str() + 5);
            }
        }
    }
    // low-res first steps (ANIMAD_LOWRES=K): the first K steps run the DiT on a 2x downsampled latent (4x fewer tokens); the full-res x0 estimate is the upsampled low-res one,
    // and the full-res state is rebuilt as (1-t')*up(x0_low) + t'*noise, so sd.cpp's Euler update lands exactly there.
    std::vector<int> lowF;   // per-step downsample factor (1 = full res); ANIMAD_LOWRES="2,2" or "4,2"; default for the HMX path: 2,2 for 4+ steps (the first two at half size), 2 for 3 steps; never the second-to-last step: the last step (t~0) cannot add detail
    { std::string cfg = getenv("ANIMAD_LOWRES") ? getenv("ANIMAD_LOWRES") : (hmx ? (steps >= 4 ? "2,2" : steps == 3 ? "2" : "") : "");
      for (size_t p = 0; p < cfg.size();) { size_t q = cfg.find(',', p); if (q == std::string::npos) q = cfg.size(); int f = atoi(cfg.substr(p, q - p).c_str()); if (f > 0) lowF.push_back(f); p = q + 1; } }
    std::vector<double> sig(steps + 1);   // sd.cpp flow schedule: s = linspace(1, 0.001), sigma = 3s/(1+2s), final 0
    for (int i = 0; i < steps; i++) { double sv = steps > 1 ? 1.0 - (1.0 - 0.001) * i / (steps - 1) : 1.0; sig[i] = 3 * sv / (1 + 2 * sv); }
    sig[steps] = 0;
    std::vector<f32> nfull, xlow, dhatf;
    for (int n = test ? steps : 0; n < (test ? steps + 1 : steps); n++) {
        std::string sn = std::to_string(n);
        while (access((dir + "/ready_" + sn).c_str(), F_OK)) usleep(1000);  // sd.cpp writes it after all inputs of the step are complete
        double ts = now();
        auto xb = readfile(dir + "/x_" + sn + ".bin");
        auto xs = readshape(dir + "/x_" + sn + ".shape");  // ggml order: W,H,C
        int Wd = (int)xs[0], H = (int)xs[1], gw = Wd / 2, gh = H / 2, T = gh * gw;
        const int Wf = Wd, Hf = H; const int fct = n < (int)lowF.size() ? lowF[n] : 1; const bool lr = fct > 1 && (H % (2 * fct) == 0) && (Wd % (2 * fct) == 0) && (((H / 2 / fct) * (Wd / 2 / fct)) % 32 == 0);   // the low-res token count must also be a multiple of 32
        f32 t = ((const f32*)readfile(dir + "/t_" + sn + ".bin").data())[0];
        if (ctx512.empty()) {
            auto hb = readfile(dir + "/ctx_" + sn + ".bin");
            auto idb = readfile(dir + "/ids_" + sn + ".bin");
            auto wb = readfile(dir + "/w_" + sn + ".bin");
            int S = (int)(hb.size() / 4 / 1024), L = (int)(idb.size() / 4);
            double ta = now();
            ctx512 = adapter((const f32*)hb.data(), S, (const int32_t*)idb.data(), (const f32*)wb.data(), L);
            fprintf(stderr, "animad: adapter S=%d L=%d %.2fs\n", S, L, now() - ta);
            if (test) { writefile(dir + "/ctx_adapter.bin", ctx512.data(), ctx512.size() * 4); return; }
            std::vector<f32> ct((size_t)1024 * 512);
            for (int i = 0; i < 512; i++) for (int c = 0; c < 1024; c++) ct[(size_t)c * 512 + i] = ctx512[(size_t)i * 1024 + c];
            writefile(ph + "/st_ctx.raw", ct.data(), ct.size() * 4);
        }
        // timestep embedding
        std::vector<f32> tp(D), emb(D), lora(6144), tmp(D);
        for (int i = 0; i < D / 2; i++) { f32 a = t * expf(-logf(10000.f) * i / (D / 2)); tp[i] = cosf(a); tp[i + D / 2] = sinf(a); }
        memcpy(emb.data(), tp.data(), D * 4);
        rms_rows(emb.data(), 1, D, w("t_embedding_norm.weight"));
        std::vector<f32> sil(D);
        linear(tp.data(), 1, D, w("t_embedder.1.linear_1.weight"), nullptr, tmp.data());
        for (auto& e : tmp) e = e / (1.f + expf(-e));
        linear(tmp.data(), 1, D, w("t_embedder.1.linear_2.weight"), nullptr, lora.data());
        // patchify: x [C=16][H][W] -> tokens [T,68] (17th channel zero), feature order (c, ph, pw)
        const f32* xin = (const f32*)xb.data();
        const f32* xfull = xin;
        if (lr) {   // switch the whole step to a latent downsampled by fct: x_low = (1-t)*pool(x0 estimate) + t*fct*pool(noise)
            if (n == 0) { nfull.assign(xin, xin + (size_t)16 * Hf * Wf); dhatf.assign(nfull.size(), 0.f); }
            const int Hl = Hf / fct, Wl = Wf / fct;
            xlow.assign((size_t)16 * Hl * Wl, 0.f);
            for (int c = 0; c < 16; c++) for (int y = 0; y < Hl; y++) for (int x = 0; x < Wl; x++) {
                double sn = 0, sd = 0;
                for (int a = 0; a < fct; a++) for (int b = 0; b < fct; b++) { size_t i = ((size_t)c * Hf + y * fct + a) * Wf + x * fct + b; sn += nfull[i]; sd += dhatf[i]; }
                xlow[((size_t)c * Hl + y) * Wl + x] = (f32)((1 - t) * sd / (fct * fct) + t * sn / fct);   // noise mean * fct keeps unit variance
            }
            Wd = Wl; H = Hl; gw = Wd / 2; gh = H / 2; T = gh * gw;
            xin = xlow.data();
        }
        std::vector<f32> pt((size_t)T * 68, 0.f), xtok((size_t)T * D);
        for (int gy = 0; gy < gh; gy++) for (int gx = 0; gx < gw; gx++) for (int c = 0; c < 16; c++) for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)
            pt[(size_t)(gy * gw + gx) * 68 + c * 4 + a * 2 + b] = xin[((size_t)c * H + gy * 2 + a) * Wd + gx * 2 + b];
        linear(pt.data(), T, 68, w("x_embedder.proj.1.weight"), nullptr, xtok.data());
        std::vector<f32> xt((size_t)D * T);
        for (int i = 0; i < T; i++) for (int c = 0; c < D; c++) xt[(size_t)c * T + i] = xtok[(size_t)i * D + c];
        if (getenv("ANIMAD_DBG")) { writefile(dir + "/dbg_x_" + sn + ".bin", xtok.data(), xtok.size() * 4); writefile(dir + "/dbg_emb_" + sn + ".bin", emb.data(), D * 4); writefile(dir + "/dbg_lora_" + sn + ".bin", lora.data(), 6144 * 4); }
        writefile(ph + "/in_x.raw", xt.data(), xt.size() * 4);
        writefile(ph + "/st_emb.raw", emb.data(), D * 4);
        writefile(ph + "/st_lora.raw", lora.data(), 6144 * 4);
        unsigned skipmask = 0;   // experiment: /data/local/tmp/anima/skip.txt lines "<step> <mask>" (bit i = skip block i, full-res HMX steps only)
        if (hmx && !lr) { FILE* sf = fopen((ph + "/skip.txt").c_str(), "r"); if (sf) { int st; unsigned mk; while (fscanf(sf, "%d %u", &st, &mk) == 2) if (st == n) skipmask |= mk; fclose(sf); } }
        double tb = now(), npu = 0;
        std::string src = "in_x.raw";
        for (size_t k = 0; k < ch.size(); k++) {
            std::string dst = "chain_x" + std::to_string(k) + (hmx && k + 1 < ch.size() ? ".x32" : ".raw");   // HMX stages pass tiles directly
            fprintf(ch[k].in, "run %s st_emb.raw st_lora.raw st_ctx.raw %s %d %d %u\n", src.c_str(), dst.c_str(), gh, gw, skipmask);
            fflush(ch[k].in);
            std::string r = waitline(ch[k], "done", "error");
            if (r[0] != 'd') die("chainrun error", r);
            npu += atof(r.c_str() + 5);
            src = dst;
        }
        auto ob = readfile(ph + "/" + src);
        const f32* xo = (const f32*)ob.data();  // [D][T]
        std::vector<f32> xr((size_t)T * D);
        for (int c = 0; c < D; c++) for (int i = 0; i < T; i++) xr[(size_t)i * D + c] = xo[(size_t)c * T + i];
        // final layer
        std::vector<f32> h256(256), m(4096);
        for (int i = 0; i < D; i++) sil[i] = emb[i] / (1.f + expf(-emb[i]));
        linear(sil.data(), 1, D, w("final_layer.adaln_modulation.1.weight"), nullptr, h256.data());
        linear(h256.data(), 1, 256, w("final_layer.adaln_modulation.2.weight"), nullptr, m.data());
        for (int i = 0; i < 4096; i++) m[i] += lora[i];
        for (int i = 0; i < T; i++) {
            f32* v = &xr[(size_t)i * D];
            double mu = 0, var = 0;
            for (int c = 0; c < D; c++) mu += v[c];
            mu /= D;
            for (int c = 0; c < D; c++) var += (v[c] - mu) * (v[c] - mu);
            f32 rs = 1.f / sqrtf((f32)(var / D) + 1e-6f);
            for (int c = 0; c < D; c++) v[c] = (v[c] - (f32)mu) * rs * (1.f + m[D + c]) + m[c];
        }
        std::vector<f32> y((size_t)T * 64), out((size_t)16 * H * Wd);
        linear(xr.data(), T, D, w("final_layer.linear.weight"), nullptr, y.data());
        for (int gy = 0; gy < gh; gy++) for (int gx = 0; gx < gw; gx++) for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 16; c++)
            out[((size_t)c * H + gy * 2 + a) * Wd + gx * 2 + b] = y[(size_t)(gy * gw + gx) * 64 + (a * 2 + b) * 16 + c];
        if (lr) {
            double tn = sig[n + 1], dt = t - tn;
            std::vector<f32> dl((size_t)16 * H * Wd);   // x0 estimate (low-res) = x_t - t*v
            for (size_t i = 0; i < dl.size(); i++) dl[i] = xlow[i] - t * out[i];
            std::vector<f32> vf((size_t)16 * Hf * Wf);
            for (int c = 0; c < 16; c++) for (int y = 0; y < Hf; y++) {
                float fy = (y + 0.5f) / fct - 0.5f; int y0 = (int)floorf(fy); float wy = fy - y0; int y1 = std::min(std::max(y0 + 1, 0), H - 1); y0 = std::min(std::max(y0, 0), H - 1);
                for (int x = 0; x < Wf; x++) {
                    float fx = (x + 0.5f) / fct - 0.5f; int x0 = (int)floorf(fx); float wx = fx - x0; int x1 = std::min(std::max(x0 + 1, 0), Wd - 1); x0 = std::min(std::max(x0, 0), Wd - 1);
                    const f32* b = &dl[(size_t)c * H * Wd];
                    f32 up = (1 - wy) * ((1 - wx) * b[y0 * Wd + x0] + wx * b[y0 * Wd + x1]) + wy * ((1 - wx) * b[y1 * Wd + x0] + wx * b[y1 * Wd + x1]);
                    size_t i = ((size_t)c * Hf + y) * Wf + x;
                    dhatf[i] = up;
                    f32 xnext = (f32)(1 - tn) * up + (f32)tn * nfull[i];
                    vf[i] = dt > 1e-6 ? (xfull[i] - xnext) / (f32)dt : 0.f;
                }
            }
            out.swap(vf);
        }
        writefile(dir + "/out_" + sn + ".bin", out.data(), out.size() * 4);
        writefile(dir + "/out_" + sn + ".bin.done", "", 0);
        fprintf(stderr, "animad: step %d t=%.4f total %.2fs (prelude %.2f, npu %.2f, final+io %.2f)\n", n, t, now() - ts, tb - ts, npu / 1000, now() - tb - npu / 1000);
    }
    if (!vae.empty() && !vd.empty()) {  // sd.cpp (SD_VAE_MODE=extern) writes z.bin/z.shape and waits for vae_out.bin(.done)
        while (access((vd + "/z.shape").c_str(), F_OK)) usleep(1000);
        usleep(3000);
        double tv = now();
        auto zs = readshape(vd + "/z.shape");   // ggml order: W, H, C
        const int Wl = (int)zs[0], Hl = (int)zs[1], TW = 64, TH = 96, CH = 16, OV = 16;   // the NPU VAE graph decodes a 64x96 latent (512x768 px)
        if (Wl == TW && Hl == TH) {
            fprintf(vae[0].in, "run %s/z.bin %s/vae_out.bin\n", vd.c_str(), vd.c_str()); fflush(vae[0].in);
            std::string r = waitline(vae[0], "done", "error");
            if (r[0] != 'd') die("vaerun error", r);
            fprintf(stderr, "animad: vae %.2fs (npu %s)\n", now() - tv, r.c_str() + 5);
        } else {   // other sizes: overlapping 64x96 tiles through the same graph (latents smaller than a tile are mirror-padded and the result cropped), blended with linear ramps
            auto zb = readfile(vd + "/z.bin"); const f32* z0 = (const f32*)zb.data();
            const int Wq = std::max(Wl, TW), Hq = std::max(Hl, TH);   // padded latent size
            std::vector<f32> zpad;
            const f32* z = z0;
            if (Wq != Wl || Hq != Hl) {
                zpad.assign((size_t)CH * Hq * Wq, 0.f);
                auto mir = [](int i, int L) { if (i >= L) i = 2 * L - 2 - i; return std::min(std::max(i, 0), L - 1); };
                for (int c = 0; c < CH; c++) for (int y = 0; y < Hq; y++) for (int x = 0; x < Wq; x++) zpad[((size_t)c * Hq + y) * Wq + x] = z0[((size_t)c * Hl + mir(y, Hl)) * Wl + mir(x, Wl)];
                z = zpad.data();
            }
            auto starts = [&](int L, int T) { if (L == T) return std::vector<int>{0}; int n = std::max(2, (L - OV + (T - OV) - 1) / (T - OV)); std::vector<int> s; for (int i = 0; i < n; i++) s.push_back((int)lround((double)i * (L - T) / (n - 1))); return s; };
            std::vector<int> xs = starts(Wq, TW), ys = starts(Hq, TH);
            const int Wp = Wq * 8, Hp = Hq * 8, PW = TW * 8, PH = TH * 8;
            std::vector<f32> acc((size_t)3 * Hp * Wp, 0.f), wsum((size_t)Hp * Wp, 0.f), tz((size_t)CH * TH * TW);
            for (size_t iy = 0; iy < ys.size(); iy++) for (size_t ix = 0; ix < xs.size(); ix++) {
                const int y0 = ys[iy], x0 = xs[ix];
                for (int c = 0; c < CH; c++) for (int y = 0; y < TH; y++) memcpy(&tz[((size_t)c * TH + y) * TW], &z[((size_t)c * Hq + y0 + y) * Wq + x0], TW * 4);
                writefile(vd + "/tz.bin", tz.data(), tz.size() * 4);
                fprintf(vae[0].in, "run %s/tz.bin %s/tout.bin\n", vd.c_str(), vd.c_str()); fflush(vae[0].in);
                std::string r = waitline(vae[0], "done", "error");
                if (r[0] != 'd') die("vaerun error", r);
                auto ob = readfile(vd + "/tout.bin"); const f32* o = (const f32*)ob.data();
                auto ramp = [&](int p, int len, int lo, int hi) { f32 w = 1.f; if (lo > 0 && p < lo) w = std::min(w, (p + 1.f) / (lo + 1.f)); if (hi > 0 && p >= len - hi) w = std::min(w, (len - p) / (hi + 1.f)); return w; };
                const int ol = ix > 0 ? (xs[ix - 1] + TW - x0) * 8 : 0, orr = ix + 1 < xs.size() ? (x0 + TW - xs[ix + 1]) * 8 : 0;
                const int ot = iy > 0 ? (ys[iy - 1] + TH - y0) * 8 : 0, ob2 = iy + 1 < ys.size() ? (y0 + TH - ys[iy + 1]) * 8 : 0;
                for (int y = 0; y < PH; y++) {
                    const f32 wy = ramp(y, PH, ot, ob2);
                    for (int x = 0; x < PW; x++) {
                        const f32 w = wy * ramp(x, PW, ol, orr); const size_t pi = (size_t)(y0 * 8 + y) * Wp + x0 * 8 + x;
                        wsum[pi] += w;
                        for (int c = 0; c < 3; c++) acc[(size_t)c * Hp * Wp + pi] += w * o[((size_t)c * PH + y) * PW + x];
                    }
                }
            }
            const int Wo = Wl * 8, Ho = Hl * 8;   // crop to the real size
            std::vector<f32> outv((size_t)3 * Ho * Wo);
            for (int c = 0; c < 3; c++) for (int y = 0; y < Ho; y++) for (int x = 0; x < Wo; x++) outv[((size_t)c * Ho + y) * Wo + x] = acc[(size_t)c * Hp * Wp + (size_t)y * Wp + x] / wsum[(size_t)y * Wp + x];
            writefile(vd + "/vae_out.bin", outv.data(), outv.size() * 4);
            fprintf(stderr, "animad: vae tiled %zux%zu tiles %.2fs\n", xs.size(), ys.size(), now() - tv);
        }
        writefile(vd + "/vae_out.bin.done", "", 0);
    }
    };
    if (serve) {
        mkdir(dir.c_str(), 0777);
        fprintf(stderr, "animad: serving %s\n", dir.c_str());
        for (;;) {
            bool did = false;
            if (DIR* d = opendir(dir.c_str())) {
                std::vector<std::string> names;
                while (dirent* e = readdir(d)) if (e->d_name[0] == 'r') names.push_back(e->d_name);
                closedir(d);
                for (auto& nm : names) {
                    std::string rd = dir + "/" + nm;
                    if (access((rd + "/start").c_str(), F_OK) || !access((rd + "/finished").c_str(), F_OK)) continue;
                    auto sb = readfile(rd + "/start");
                    int st = atoi(std::string(sb.begin(), sb.end()).c_str());
                    double tr = now();
                    handle(rd, st > 0 ? st : 4, rd + "/vae");
                    writefile(rd + "/finished", "", 0);
                    fprintf(stderr, "animad: request %s done in %.1fs\n", nm.c_str(), now() - tr);
                    did = true;
                }
            }
            if (!did) usleep(5000);
            if (!access((dir + "/quit").c_str(), F_OK)) break;
        }
    } else {
        handle(dir, steps, getenv("SD_VAE_DIR") ? getenv("SD_VAE_DIR") : "");
    }
    if (!vae.empty()) { fprintf(vae[0].in, "quit\n"); fflush(vae[0].in); }
    if (!te.empty()) { fprintf(te[0].in, "quit\n"); fflush(te[0].in); }
    for (auto& c : ch) { fprintf(c.in, "quit\n"); fflush(c.in); }
    for (auto& c : ch) waitpid(c.pid, nullptr, 0);
    return 0;
}
