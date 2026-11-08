package com.hexanima.npu;

import android.graphics.Bitmap;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** MOSS-Image (5B DiT + Qwen3.5-9B text encoder + FLUX.2 VAE) on the NPU: files under files/moss, one picture per moss.sh call. */
final class Moss {
    static final String PREFIX_ID = "moss";
    static boolean is(String id) { return id != null && id.startsWith(PREFIX_ID); }
    static File dir(File home) { return new File(home, "moss"); }
    static boolean installed(File home) { return new File(dir(home), ".installed").exists(); }
    /** decoder contexts: the pack's vae/ (V69), or files/vae81 on V81 (kept outside the model dir, so reinstalling the model keeps them) */
    static File vaeDir(File home) { return Compat.qnn243() ? new File(home, "vae81") : new File(dir(home), "vae"); }
    /** the pack's vae/ holds V69 contexts only: not downloaded on other generations */
    static boolean skipFile(String path) { return Compat.qnn243() && path.startsWith("vae/"); }
    /** a decoder exists for any size (512x512 is the tiled fallback); null = fine */
    static String vaeError(File home) {
        return new File(vaeDir(home), "vae_512x512.bin").exists() ? null
            : L.f("Нет декодера для этого чипа (%s/vae_512x512.bin)", "No decoder for this chip yet (%s/vae_512x512.bin)", vaeDir(home).getPath());
    }
    /** environment for moss.sh: decoder dir, the QNN runtime of this generation, DSP workers and page-cache blocks for mossrun */
    static String env(android.content.Context c) {
        File home = c.getFilesDir(); int nw = Compat.mossWorkers(c), keep = Compat.mossKeep(c);
        return "VAED='" + vaeDir(home).getPath() + "' " + (Compat.qnn243() ? "QH=libQnnHtp243.so QS=libQnnSystem243.so " : "")
            + (nw > 0 ? "MOSS_NW=" + nw + " " : "") + (keep > 0 ? "MOSS_KEEP=" + keep + " " : "");
    }
    /** rough seconds per picture: V81 measured ~38 s at 1024x1024 (4 steps, 6 workers) plus a few seconds of text encoder / decoder; elsewhere the catalog speed (512x768) scaled by patches.
     *  Full mode: one DiT pass per step, two with CFG, plus the negative prompt's text encoder on its first use. */
    static long estimate(int speed512x768, int w, int h, boolean full, int steps, float cfg) {
        double p = (w / 16) * (h / 16), passes = full ? steps * (cfg != 1f ? 2.0 : 1.0) : 4;
        return Math.round(Compat.qnn243() ? (full ? 6 : 4) + 8.5 * passes * p / 4096 : speed512x768 * p / 1536.0 * passes / 4);
    }

    /** full model (rapid-moss adapter off) sampled like Unsloth Studio does MOSS-Image-SFT: FlowMatch Euler on ComfyUI's sgm_uniform shift-5 schedule, plain CFG with
     *  a negative prompt (the reference pipeline's block-22 skip in the negative branch makes SFT pictures smooth and airbrushed) */
    static final int FULL_STEPS_MIN = 8, FULL_STEPS_MAX = 60, FULL_STEPS_DEF = 20;
    static final float CFG_DEF = 3f;
    /** the SFT model's own time embedding (glue.bin carries the adapter's): files/moss/glue_base.bin from moss/export_glue_base.py; null = fine */
    static String fullError(File home) {
        return new File(dir(home), "glue_base.bin").exists() ? null
            : L.s("Для полной модели нужен файл files/moss/glue_base.bin (moss/export_glue_base.py)", "The full model needs files/moss/glue_base.bin (moss/export_glue_base.py)");
    }

    /** size presets (W×H): around 1 Mpx is what the 4-step adapter was trained on (1024², 896×1152, 832×1216, 768×1344 and their landscape twins); smaller ones are faster */
    static final int[][] PRESETS = {{512, 512}, {640, 640}, {768, 768}, {1024, 1024}, {512, 768}, {768, 512}, {576, 1024}, {1024, 576}, {640, 1152}, {1152, 640}, {768, 1024}, {1024, 768},
        {896, 1152}, {1152, 896}, {832, 1216}, {1216, 832}, {768, 1344}, {1344, 768}};
    /** any size with sides that are multiples of 16, at least 256, at most 6144 patches (the DiT attention must fit the 8 MB VTCM); null = fine */
    static String sizeError(int w, int h) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 16 != 0 || h % 16 != 0) return L.s("Ширина и высота должны быть кратны 16", "Width and height must be multiples of 16");
        if ((w / 16) * (h / 16) > 6144) return L.s("Слишком большой размер: максимум 6144 патча (например 1536×1024)", "Too large: at most 6144 patches (e.g. 1536×1024)");
        return null;
    }

    /** tokens of the prompt (chat template included) -> ids file for mossrun; an empty prompt is the template alone (the official negative prompt '') */
    static void writeIds(File home, String prompt, File out) throws Exception {
        QwenTok t = new QwenTok(dir(home).getPath());
        int[] ids = t.encodePrompt(prompt);
        if (ids.length > 480) ids = java.util.Arrays.copyOf(ids, 480);   // mossrun takes at most 512 tokens
        ByteBuffer b = ByteBuffer.allocate(ids.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : ids) b.putInt(i);
        try (FileOutputStream o = new FileOutputStream(out)) { o.write(b.array()); }
    }

    /** raw fp32 RGB [3][h][w] in [-1,1] -> PNG bytes */
    static byte[] png(File raw, int w, int h) throws IOException {
        byte[] bytes = Models.readBytes(raw);
        if (bytes.length != 3 * w * h * 4) throw new IOException("decoder output has " + bytes.length + " bytes");
        java.nio.FloatBuffer f = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        int[] px = new int[w * h]; int n = w * h;
        for (int i = 0; i < n; i++) {
            int r = c(f.get(i)), g = c(f.get(n + i)), b = c(f.get(2 * n + i));
            px[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        Bitmap bm = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888);
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        bm.compress(Bitmap.CompressFormat.PNG, 100, bo);
        return bo.toByteArray();
    }
    private static int c(float v) { int x = Math.round((Math.max(-1f, Math.min(1f, v)) + 1f) * 127.5f); return x < 0 ? 0 : x > 255 ? 255 : x; }
}
