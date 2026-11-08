package com.hexanima.npu;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Model packs: a catalog (catalog.json) lists models; every model and the shared "common" pack have a manifest with raw files (size + sha256) that are
 * downloaded one by one (resumable, verified). Nothing is zipped: the weights do not compress and per-file resume/verification is simpler.
 * The multi-GB safetensors files that sd.cpp opens are NOT downloaded: only their JSON header (hdr.bin) is, and the app rebuilds a sparse stub of the real size.
 * Layout: files/anima/{te*.bin, te_embed.f16, vae.bin} (common), files/anima/m_ID/{small.bin, blkNN.bin}, files/sd/{ID.safetensors stub, llm stub, taew2_1.safetensors}.
 */
final class Models {
    static class Item { String id, name, desc, descEn, params = "", quant = "", author = "", manifest; long size; int speed; boolean moss; }   // speed: seconds per picture at 512x768
    interface Progress { void on(long done, long total, String what); }

    final File anima, sd, home;
    Models(File home) { this.home = home; anima = new File(home, "anima"); sd = new File(home, "sd"); anima.mkdirs(); sd.mkdirs(); }

    static String get(String url) throws IOException {
        HttpURLConnection c = open(url, 0);
        try (InputStream in = c.getInputStream(); ByteArrayOutputStream bo = new ByteArrayOutputStream()) {
            byte[] b = new byte[8192]; for (int n; (n = in.read(b)) > 0; ) bo.write(b, 0, n);
            return bo.toString("UTF-8");
        } finally { c.disconnect(); }
    }

    static HttpURLConnection open(String url, long from) throws IOException {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(20000); c.setReadTimeout(30000); c.setInstanceFollowRedirects(true);
        if (from > 0) c.setRequestProperty("Range", "bytes=" + from + "-");
        return c;
    }

    long commonSize;   // shared files of the QNN pipeline (catalog); see commonBytes() for the CPU text-encoder pipeline
    /** the full Qwen3 0.6B text encoder for the CPU path: the stock Anima file (same header and weights as the MiaoMiao packs' text encoder, checked byte for byte
     *  on the header and on the embedding table), saved under its own name so a sparse stub from a QNN install is never mistaken for it */
    static final String TE_FILE = "qwen_3_06b_base.safetensors";
    static final String TE_URL = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/" + TE_FILE;
    static final long TE_SIZE = 1192135096L;
    static final String TE_SHA = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba";
    static final long TAE_SIZE = 22642902L;
    long commonBytes() { return Compat.qnnCommon() || Compat.dspArch().isEmpty() ? commonSize : TE_SIZE + TAE_SIZE; }
    List<Item> catalog(String base) throws Exception {
        String txt;   // catalog2.json lists the Anima models and MOSS-Image (1.3+); catalog.json stays as is for older apps, which would misread a MOSS entry
        try { txt = get(base + "catalog2.json"); } catch (Exception e) { txt = get(base + "catalog.json"); }
        JSONObject j = new JSONObject(txt);
        commonSize = j.optLong("common_size", 0);
        List<Item> out = new ArrayList<>();
        JSONArray a = j.getJSONArray("models");
        for (int i = 0; i < a.length(); i++) {
            JSONObject m = a.getJSONObject(i); Item it = new Item();
            it.id = m.getString("id"); it.name = m.getString("name"); it.desc = m.optString("desc", ""); it.size = m.getLong("size"); it.manifest = m.getString("manifest");
            it.descEn = m.optString("desc_en", it.desc); it.params = m.optString("params", ""); it.quant = m.optString("quant", ""); it.author = m.optString("author", ""); it.speed = m.optInt("speed", 0); it.moss = m.optString("kind", "").equals("moss");
            out.add(it);
        }
        return out;
    }

    // ---- state ----------------------------------------------------------------------------------
    File modelDir(String id) { return Moss.is(id) ? Moss.dir(home) : new File(anima, "m_" + id); }
    File doneMark(File dir) { return new File(dir, ".installed"); }
    boolean modelInstalled(String id) { if (Moss.is(id)) return Moss.installed(home) && id.equals(mossId()); return commonInstalled() && doneMark(modelDir(id)).exists() && new File(sd, id + ".safetensors").exists(); }
    /** v69 keeps the original "common" folder (older apps use it); other Hexagon generations have their own QNN contexts in common_<arch>,
     *  or (no QNN contexts for that generation, e.g. v81) only a local marker folder: the CPU text encoder and the TAEHV decoder are fetched instead */
    static String commonDir() { String a = Compat.dspArch(); return a.isEmpty() || a.equals("v69") ? "common" : "common_" + a; }
    boolean commonInstalled() { return doneMark(new File(anima, commonDir())).exists(); }
    String mossId() { try { return new JSONObject(read(new File(Moss.dir(home), "meta.json"))).getString("id"); } catch (Exception e) { return ""; } }
    /** ids of installed models, in catalog order of installation directory names */
    List<String> installed() {
        List<String> out = new ArrayList<>();
        if (Moss.installed(home)) out.add(mossId());
        File[] ds = anima.listFiles();
        if (ds != null) for (File d : ds) if (d.getName().startsWith("m_") && modelInstalled(d.getName().substring(2))) out.add(d.getName().substring(2));
        java.util.Collections.sort(out);
        return out;
    }
    String name(String id) { try { return new JSONObject(read(new File(modelDir(id), "meta.json"))).getString("name"); } catch (Exception e) { return id; } }

    /** seconds per picture at 512x768 (from the catalog at install time; models installed before that get the measured value of their size class) */
    int speed(String id) { try { int v = new JSONObject(read(new File(modelDir(id), "meta.json"))).optInt("speed", 0); if (v > 0) return v; } catch (Exception e) { } return id.contains("29") ? 12 : 7; }

    static byte[] readBytes(File f) throws IOException {
        try (FileInputStream in = new FileInputStream(f)) { byte[] b = new byte[(int) f.length()]; int o = 0; for (int n; o < b.length && (n = in.read(b, o, b.length - o)) > 0; ) o += n; return b; }
    }
    static String read(File f) throws IOException {
        try (FileInputStream in = new FileInputStream(f)) { byte[] b = new byte[(int) f.length()]; int o = 0; for (int n; o < b.length && (n = in.read(b, o, b.length - o)) > 0; ) o += n; return new String(b, "UTF-8"); }
    }

    // ---- install --------------------------------------------------------------------------------
    void installCommon(String base, Progress pg, AtomicBoolean cancel) throws Exception {
        if (commonInstalled()) return;
        if (!Compat.qnnCommon() && !Compat.dspArch().isEmpty()) { installCommonCpu(base, pg, cancel); return; }
        JSONObject m = new JSONObject(get(base + commonDir() + "/manifest.json"));
        File marker = new File(anima, commonDir()); marker.mkdirs();
        long total = total(m.getJSONArray("files")) + total(m.getJSONArray("sd"));
        long[] done = {0};
        fetchAll(base + commonDir() + "/", m.getJSONArray("files"), anima, total, done, pg, cancel);
        fetchAll(base + commonDir() + "/", m.getJSONArray("sd"), sd, total, done, pg, cancel);
        JSONObject llm = m.getJSONObject("llm_stub");
        File hdr = new File(marker, "llm_hdr.bin");
        fetch(base + commonDir() + "/" + llm.getString("hdr"), hdr, llm.optLong("hdr_size", -1), llm.optString("hdr_sha256", null), null, cancel);
        stub(hdr, new File(sd, llm.getString("name")), llm.getLong("size"));
        doneMark(marker).createNewFile();
    }

    /** no QNN contexts for this Hexagon generation: sd.cpp runs the text encoder and the TAEHV decoder on the CPU, so the real text encoder weights
     *  (1.2 GB, from the Anima repository) and taew2_1.safetensors (from the v69 common pack) are all that is shared; nothing else of "common" is needed */
    void installCommonCpu(String base, Progress pg, AtomicBoolean cancel) throws Exception {
        JSONObject m = new JSONObject(get(base + "common/manifest.json"));
        JSONArray sdf = m.getJSONArray("sd");
        File marker = new File(anima, commonDir()); marker.mkdirs();
        long total = TE_SIZE + total(sdf);
        long[] done = {0};
        fetchAll(base + "common/", sdf, sd, total, done, pg, cancel);
        final long before = done[0];
        fetch(TE_URL, new File(sd, TE_FILE), TE_SIZE, TE_SHA, n -> { if (pg != null) pg.on(before + n, total, TE_FILE); }, cancel);
        doneMark(marker).createNewFile();
    }

    void installModel(String base, Item it, Progress pg, AtomicBoolean cancel) throws Exception {
        final String murl = it.manifest.startsWith("http") ? it.manifest : base + it.manifest;   // a pack may live in another repository (MOSS-Image: Apache-2.0, not in the Anima repo)
        JSONObject m = new JSONObject(get(murl));
        File dir = modelDir(it.id); dir.mkdirs(); doneMark(dir).delete();
        JSONArray files = m.getJSONArray("files");
        long[] done = {0};
        fetchAll(murl.substring(0, murl.lastIndexOf('/') + 1), files, dir, total(files), done, pg, cancel);
        if (it.moss) {   // MOSS-Image: raw files only, no sd.cpp stub
            try (FileWriter w = new FileWriter(new File(dir, "meta.json"))) { w.write(new JSONObject().put("id", it.id).put("name", it.name).put("speed", it.speed).toString()); }
            doneMark(dir).createNewFile(); return;
        }
        JSONObject st = m.getJSONObject("stub");
        File hdr = new File(dir, st.getString("hdr"));
        stub(hdr, new File(sd, it.id + ".safetensors"), st.getLong("size"));
        try (FileWriter w = new FileWriter(new File(dir, "meta.json"))) { w.write(new JSONObject().put("id", it.id).put("name", it.name).put("speed", it.speed).toString()); }
        doneMark(dir).createNewFile();
    }

    /** installed models take the name / speed of the current catalog (names are shown from meta.json) */
    void refreshMeta(Item it) {
        if (!modelInstalled(it.id)) return;
        try (FileWriter w = new FileWriter(new File(modelDir(it.id), "meta.json"))) { w.write(new JSONObject().put("id", it.id).put("name", it.name).put("speed", it.speed).toString()); } catch (Exception ignored) { }
    }

    void delete(String id) {
        File dir = modelDir(id);
        if (Moss.is(id)) { deleteTree(dir); return; }
        File[] fs = dir.listFiles(); if (fs != null) for (File f : fs) f.delete();
        dir.delete(); new File(sd, id + ".safetensors").delete();
    }

    static void deleteTree(File f) { File[] fs = f.listFiles(); if (fs != null) for (File c : fs) deleteTree(c); f.delete(); }

    /** sparse stub of a safetensors file: the real header, then a hole up to the real size (sd.cpp maps the file but never touches tensor data when the DiT/LLM run on the NPU) */
    static void stub(File hdr, File out, long size) throws IOException {
        try (FileInputStream in = new FileInputStream(hdr); FileOutputStream o = new FileOutputStream(out)) {
            byte[] b = new byte[65536]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n);
        }
        try (RandomAccessFile r = new RandomAccessFile(out, "rw")) { r.setLength(size); }
    }

    static long total(JSONArray a) throws Exception { long t = 0; for (int i = 0; i < a.length(); i++) t += a.getJSONObject(i).getLong("size"); return t; }

    void fetchAll(String urlBase, JSONArray files, File dir, long total, long[] done, Progress pg, AtomicBoolean cancel) throws Exception {
        for (int i = 0; i < files.length(); i++) {
            JSONObject f = files.getJSONObject(i);
            File out = new File(dir, f.getString("path"));
            final long before = done[0];
            fetch(urlBase + f.getString("path"), out, f.getLong("size"), f.optString("sha256", null), n -> { if (pg != null) pg.on(before + n, total, f.optString("path")); }, cancel);
            done[0] += f.getLong("size");
        }
    }

    interface Count { void on(long n); }

    /** download url to out (resumes a .part file), verify size and sha256, rename; a finished file with the right size is kept as is */
    static void fetch(String url, File out, long size, String sha, Count cb, AtomicBoolean cancel) throws Exception {
        if (out.exists() && (size < 0 || out.length() == size)) { if (cb != null) cb.on(out.length()); return; }
        File part = new File(out.getPath() + ".part");
        out.getParentFile().mkdirs();
        for (int attempt = 0; ; attempt++) {
            try {
                long have = part.exists() ? part.length() : 0;
                if (size >= 0 && have > size) { part.delete(); have = 0; }
                if (size < 0 || have < size) {
                    HttpURLConnection c = open(url, have);
                    int code = c.getResponseCode();
                    if (code == 200 && have > 0) { part.delete(); have = 0; }   // server ignored Range
                    else if (code != 200 && code != 206) throw new IOException("HTTP " + code + " " + url);
                    try (InputStream in = c.getInputStream(); FileOutputStream o = new FileOutputStream(part, have > 0)) {
                        byte[] b = new byte[1 << 16]; long n = have;
                        for (int r; (r = in.read(b)) > 0; ) {
                            if (cancel != null && cancel.get()) throw new InterruptedIOException("cancelled");
                            o.write(b, 0, r); n += r; if (cb != null) cb.on(n);
                        }
                    } finally { c.disconnect(); }
                }
                if (size >= 0 && part.length() != size) throw new IOException("size " + part.length() + " != " + size + " " + url);
                if (sha != null && !sha.isEmpty() && !sha256(part).equalsIgnoreCase(sha)) { part.delete(); throw new IOException("sha256 mismatch " + url); }
                if (!part.renameTo(out)) throw new IOException("rename " + out);
                return;
            } catch (InterruptedIOException e) { throw e;
            } catch (IOException e) { if (attempt >= 5) throw e; Thread.sleep(2000L * (attempt + 1)); }   // flaky network: resume from the .part file
        }
    }

    static String sha256(File f) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (FileInputStream in = new FileInputStream(f)) { byte[] b = new byte[1 << 20]; for (int n; (n = in.read(b)) > 0; ) md.update(b, 0, n); }
        StringBuilder sb = new StringBuilder(); for (byte x : md.digest()) sb.append(String.format("%02x", x));
        return sb.toString();
    }
}
