package com.hexanima.npu;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Build;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Can this phone run the models? Hexagon V69 (Snapdragon 8 Gen 1 / 8+ Gen 1) is the tested target. Hexagon V81 (Snapdragon 8 Elite Gen 5, SM8850)
 * is supported by this fork: the DiTs (Anima, MOSS-Image) run on the HMX kernels as on V69 (the accumulator is read through the v73+ cvt path).
 * QNN context binaries are compiled per Hexagon generation and need the QAIRT 2.43 runtime on V81 (libQnnHtp243.so): Anima's text encoder (Qwen3 0.6B)
 * and decoder (TAEHV) run on the NPU when their V81 contexts are in files/anima (te0..te3.bin + te_embed.f16, vae.bin; export/build_common_arch.sh),
 * otherwise on the CPU through sd.cpp; MOSS-Image decodes through V81 contexts of the FLUX.2 decoder in files/vae81 (moss/build_vaes_v81.sh).
 * V79 (Snapdragon 8 Elite) stays switched off, see V79_ENABLED.
 */
final class Compat {
    static String soc() { return Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : Build.HARDWARE; }
    /** Snapdragon 8 Elite (V79) was withdrawn upstream in 1.2.2: the HMX is silently unavailable after an interrupted run and large sizes come out as noise there (see PITFALLS.md) */
    static final boolean V79_ENABLED = false;
    /** Hexagon generation of the NPU by SoC model ("" = unknown) */
    static String dspArch() {
        String s = soc();
        if (s.equals("SM8475") || s.equals("SM8450")) return "v69";
        if (V79_ENABLED && s.equals("SM8750")) return "v79";
        if (s.startsWith("SM8850")) return "v81";   // Snapdragon 8 Elite Gen 5 (e.g. Xiaomi 17 Pro Max); some phones report a suffix
        return "";
    }
    static boolean chipOk() { return !dspArch().isEmpty(); }
    /** the QNN text encoder / VAE contexts exist only for V69 (and V79 upstream); elsewhere sd.cpp runs both on the CPU */
    static boolean qnnCommon() { String a = dspArch(); return a.equals("v69") || a.equals("v79"); }
    /** MOSS-Image decodes through a QNN context (vaerun/vaetile): V69 contexts come with the pack, V81 ones from files/vae81 (Moss.vaeDir) */
    static boolean mossOk() { String a = dspArch(); return a.equals("v69") || a.equals("v81"); }
    /** V81 needs the QAIRT 2.43 runtime (shipped as libQnnHtp243.so / libQnnSystem243.so next to the 2.28 libs that serve V69) */
    static boolean qnn243() { return dspArch().equals("v81"); }
    /** largest patch count offered: V69 is verified to 6144 (megarun TMAX); on other generations large sizes are unverified (V79 gave noise above ~4800) */
    static int maxPatches(Context c) { return dspArch().equals("v69") || Settings.largeSizes(c) ? 6144 : 4096; }
    static String archLabel() { String a = dspArch(); return "Hexagon " + a.toUpperCase() + (a.equals("v79") || a.equals("v81") ? L.s(", экспериментально", ", experimental") : ""); }
    static boolean chipAllowed(Context c) { return chipOk() || c.getSharedPreferences("anima", 0).getBoolean("chip_override", false); }
    static void allowChip(Context c) { c.getSharedPreferences("anima", 0).edit().putBoolean("chip_override", true).apply(); }

    /**
     * Extra environment for anima.sh (and through it animad, megarun and sd-cli), per chip:
     * V=0 decodes on the CPU (no NPU VAE context), LLM = the full text encoder file (CPU text encoder), T = CPU threads for sd.cpp,
     * MEGA_NW = DSP worker threads per block (V81 has more HVX contexts than V69's 4), ANIMAD_LOWRES= (empty) = every step at full resolution.
     */
    static String env(Context c) {
        StringBuilder b = new StringBuilder();
        if (dspArch().equals("v81")) {   // LLM = the real text encoder weights also with the NPU one: sd.cpp falls back to them above 128 tokens
            b.append("LLM=").append(Models.TE_FILE).append(" T=").append(Settings.threads(c)).append(' ');
            if (!npuVae(c)) b.append("V=0 ");
            b.append("ANIMAD_QNN_HTP=libQnnHtp243.so ANIMAD_QNN_SYS=libQnnSystem243.so ");
            int nw = Settings.dspWorkers(c); if (nw > 0) b.append("MEGA_NW=").append(nw).append(' ');
        }
        if (Settings.bestQuality(c)) b.append("ANIMAD_LOWRES= ");
        b.append("CFG=").append(cfgKey(c)).append(' ');
        return b.toString();
    }
    /** anima.sh restarts a loaded server when this changes (thread / worker / quality settings are read at load time) */
    /** MOSS-Image DSP workers: the setting, else 6 on V81 (measured ~13% faster than 3, same output; 8 hangs) and mossrun's 3 elsewhere */
    static int mossWorkers(Context c) { int nw = Settings.dspWorkers(c); return nw > 0 ? nw : dspArch().equals("v81") ? 6 : 0; }
    /** DiT blocks MOSS keeps in the page cache between steps: 20 (2.7 GB) on V81 phones with 14 GB+ RAM (45 -> 38 s at 1024x1024 with 6 workers; all 42 was no faster); 0 elsewhere (slower on 12 GB phones) */
    static int mossKeep(Context c) { return dspArch().equals("v81") && ramGb(c) >= 14 ? 20 : 0; }
    static String cfgKey(Context c) { return dspArch() + "." + (Settings.bestQuality(c) ? "q" : "f") + Settings.threads(c) + "." + Settings.dspWorkers(c) + (npuTe(c) ? "t" : "") + (npuVae(c) ? "v" : ""); }
    /** Anima's text encoder / decoder contexts for this chip are installed (animad then runs them on the NPU) */
    static boolean npuTe(Context c) { File a = new File(c.getFilesDir(), "anima"); return new File(a, "te3.bin").exists() && new File(a, "te_embed.f16").exists(); }
    static boolean npuVae(Context c) { return new File(new File(c.getFilesDir(), "anima"), "vae.bin").exists(); }

    static double ramGb(Context c) { ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo(); ((ActivityManager) c.getSystemService(Context.ACTIVITY_SERVICE)).getMemoryInfo(mi); return mi.totalMem / 1073741824.0; }
    static double freeGb(Context c) { return c.getFilesDir().getUsableSpace() / 1e9; }

    /** [chip, ram, storage] each as {ok?"1":"0", text} */
    static List<String[]> report(Context c) {
        List<String[]> r = new ArrayList<>();
        r.add(new String[]{chipOk() ? "1" : "0", chipOk() ? L.f("Чип %s: подходит (%s)", "Chip %s: supported (%s)", soc(), archLabel()) : L.f("Чип %s: не поддерживается, нужен Snapdragon 8 Gen 1 / 8+ Gen 1 или 8 Elite Gen 5", "Chip %s: not supported, a Snapdragon 8 Gen 1 / 8+ Gen 1 or 8 Elite Gen 5 is required", soc())});
        double ram = ramGb(c);
        r.add(new String[]{ram >= 7.5 ? "1" : "0", ram >= 7.5 ? L.f("ОЗУ %.1f ГБ: достаточно", "RAM %.1f GB: enough", ram) : L.f("ОЗУ %.1f ГБ: мало, нужно от 8 ГБ", "RAM %.1f GB: low, 8 GB or more is needed", ram)});
        double fr = freeGb(c);
        r.add(new String[]{fr >= 6 ? "1" : "0", fr >= 6 ? L.f("Свободно %.1f ГБ: достаточно", "Free space %.1f GB: enough", fr) : L.f("Свободно %.1f ГБ: мало, модель занимает до 5,4 ГБ", "Free space %.1f GB: low, a model needs up to 5.4 GB", fr)});
        return r;
    }
}
