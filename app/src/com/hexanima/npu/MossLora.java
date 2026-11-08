package com.hexanima.npu;

import java.io.*;
import java.util.*;

/** User LoRA for MOSS-Image. The picked .safetensors is copied to files/moss/lora/user.safetensors and checked by liblorafold.so ("check": key names, shapes, rank, dtype, all without
 *  loading weights). Generation folds it with the turbo adapter on the phone ("fold": ~15 s) into per-block patches files/moss/lora/ovl_<id>_<strength>/blkNN.ovl that mossrun lays over the
 *  packs' rank-128 slots (MOSS_LORA=dir). A LoKr (full rank) is merged into the base weights instead: complete blocks files/moss/lora/ovl_.../blkNN.bin (~5.5 GB) that
 *  mossrun reads in place of the pack's. Only the last (LoRA, strength) pair is kept on disk (~310 MB for a LoRA, ~5.5 GB for a LoKr). The same fold also writes blkNN.ovf, the LoRA without the
 *  adapter, for the full-model mode; merged LoKr blocks serve both modes. */
final class MossLora {
    static File dir(File home) { return new File(Moss.dir(home), "lora"); }
    static File src(File home) { return new File(dir(home), "user.safetensors"); }

    /** result of lorafold check: ok + facts, or an error code and its detail */
    static final class Check {
        boolean ok; String code = "", detail = ""; int layers, rank, lokr; String alpha = "", warn = "";
        String message() {
            if (ok) {
                StringBuilder s = new StringBuilder(lokr > 0 && rank == 0 ? L.f("Подходит: LoKr, блоков %d из 42 (вливается в веса модели: ~5,5 ГБ и около минуты при первом запуске)", "Compatible: LoKr, %d of 42 blocks (merged into the model weights: ~5.5 GB and about a minute the first time)", layers)
                    : L.f("Подходит: ранг %d, блоков %d из 42", "Compatible: rank %d, %d of 42 blocks", rank, layers));
                if (warn.contains("rank>64")) s.append("\n").append(L.s("Высокий ранг: при сборке с турбо-адаптером часть тонких деталей может потеряться", "High rank: some fine detail may be lost when it is merged with the turbo adapter"));
                if (warn.contains("text_encoder_ignored")) s.append("\n").append(L.s("Часть для текстового энкодера не применяется", "The text-encoder part is not applied"));
                if (warn.contains("unknown_keys")) s.append("\n").append(L.s("Часть лишних тензоров пропущена", "Some extra tensors are skipped"));
                return s.toString();
            }
            switch (code) {
                case "OTHERMODEL": return L.s("Это LoRA для другой модели (FLUX, SDXL или похожей). Для MOSS-Image она не подходит.", "This LoRA is for another model (FLUX, SDXL or similar). It does not fit MOSS-Image.");
                case "NOTLORA": return L.s("Файл не похож на LoRA для MOSS-Image: не найдены нужные слои (layers.N.proj.*).", "This file does not look like a MOSS-Image LoRA: the expected layers (layers.N.proj.*) are missing.");
                case "TEXTONLY": return L.s("В файле только LoRA для текстового энкодера. Такие пока не поддерживаются.", "The file only has text-encoder LoRA weights. Those are not supported yet.");
                case "SHAPE": return L.f("Размеры слоёв не совпадают с MOSS-Image (%s). Скорее всего это LoRA для другой версии модели.", "Layer sizes do not match MOSS-Image (%s). It is probably a LoRA for another version of the model.", detail);
                case "RANK": return L.f("Слишком большой ранг (%s). Поддерживается от 1 до 128.", "The rank is too large (%s). Supported: 1 to 128.", detail);
                case "LAYER": return L.f("В LoRA есть блоки, которых нет в MOSS-Image (%s).", "The LoRA has blocks that MOSS-Image does not have (%s).", detail);
                case "DTYPE": return L.s("Неподдерживаемый формат весов, нужны F32, F16 или BF16.", "Unsupported weight format: F32, F16 or BF16 are needed.");
                case "FILE": return L.f("Не удалось прочитать файл (%s). Нужен .safetensors.", "Cannot read the file (%s). A .safetensors file is needed.", detail);
                default: return L.f("Ошибка проверки: %s %s", "Check failed: %s %s", code, detail);
            }
        }
    }

    static Check parse(String line) {
        Check c = new Check(); String[] a = line.trim().split("\t");
        if (a.length > 0 && a[0].equals("OK")) {
            c.ok = true;
            for (String f : a) {
                if (f.startsWith("layers=")) c.layers = Integer.parseInt(f.substring(7)); else if (f.startsWith("rank=")) c.rank = Integer.parseInt(f.substring(5));
                else if (f.startsWith("alpha=")) c.alpha = f.substring(6); else if (f.startsWith("warn=")) c.warn = f.substring(5); else if (f.startsWith("lokr=")) c.lokr = Integer.parseInt(f.substring(5));
            }
        } else { c.code = a.length > 1 ? a[1] : "?"; c.detail = a.length > 2 ? a[2] : ""; }
        return c;
    }

    static String run(String bin, String... args) throws IOException {
        List<String> cmd = new ArrayList<>(); cmd.add(bin + "/liblorafold.so"); cmd.addAll(Arrays.asList(args));
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) { for (String l; (l = r.readLine()) != null; ) sb.append(l).append('\n'); }
        try { p.waitFor(); } catch (InterruptedException ignored) { }
        return sb.toString();
    }

    static Check check(String bin, File f) {
        try {
            String out = run(bin, "check", f.getPath()); String last = "";
            for (String l : out.split("\n")) if (l.startsWith("OK") || l.startsWith("ERR")) last = l;
            return last.isEmpty() ? parse("ERR\tFILE\t" + out.trim()) : parse(last);
        } catch (IOException e) { return parse("ERR\tFILE\t" + e.getMessage()); }
    }

    interface Progress { void at(int done, int total); }
    /** folded patches for the picked LoRA at this strength; reuses the cache, null on failure/cancel (msg[0] = reason) */
    static File ensure(File home, String bin, String id, float strength, Progress pr, java.util.function.BooleanSupplier cancelled, String[] msg) {
        File root = dir(home), out = new File(root, "ovl_" + id + "_" + Math.round(strength * 100));
        if (new File(out, ".ok").exists() && new File(out, "blk00.ovf").exists()) return out;   // .ovf (fold without the adapter, full mode) since 1.3.0-sm8850.3: older caches are rebuilt
        File[] old = root.listFiles(); if (old != null) for (File f : old) if (f.getName().startsWith("ovl_")) deleteTree(f);
        out.mkdirs();
        try {
            Process p = new ProcessBuilder(bin + "/liblorafold.so", "fold", Moss.dir(home).getPath(), src(home).getPath(), String.valueOf(strength), out.getPath()).redirectErrorStream(true).start();
            String last = "";
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                for (String l; (l = r.readLine()) != null; ) {
                    if (cancelled.getAsBoolean()) { p.destroy(); deleteTree(out); return null; }
                    if (l.startsWith("progress ")) { String[] a = l.split(" "); pr.at(Integer.parseInt(a[1]), Integer.parseInt(a[2])); } else last = l;
                }
            }
            if (p.waitFor() != 0 || !last.startsWith("OK")) { msg[0] = last; deleteTree(out); return null; }
            new File(out, ".ok").createNewFile(); return out;
        } catch (Exception e) { msg[0] = String.valueOf(e); deleteTree(out); return null; }
    }

    static void deleteTree(File f) { File[] k = f.listFiles(); if (k != null) for (File c : k) deleteTree(c); f.delete(); }
}
