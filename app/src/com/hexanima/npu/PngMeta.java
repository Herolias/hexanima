package com.hexanima.npu;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.zip.CRC32;
import org.json.JSONObject;

/** Generation settings inside the PNG (iTXt chunks): "anima" = our JSON, "parameters" = the A1111-style text other tools understand. */
final class PngMeta {
    static byte[] embed(byte[] png, JSONObject m) {
        try {
            String prompt = m.optString("prompt");
            String neg = m.optString("negative", "");   // MOSS full mode carries its own steps / CFG / negative prompt
            String params = prompt + (neg.isEmpty() ? "" : "\nNegative prompt: " + neg) + "\nSteps: " + m.optInt("steps", 4) + ", Sampler: Euler, CFG scale: "
                + String.format(java.util.Locale.US, "%.4g", m.optDouble("cfg", 1.0)) + ", Seed: " + m.optString("seed") + ", Size: " + m.optInt("w") + "x" + m.optInt("h") + ", Model: " + m.optString("model");
            ByteArrayOutputStream o = new ByteArrayOutputStream();
            o.write(png, 0, 33);   // signature + IHDR (8 + 25 bytes)
            chunk(o, "anima", m.toString()); chunk(o, "parameters", params);
            o.write(png, 33, png.length - 33);
            return o.toByteArray();
        } catch (Exception e) { return png; }
    }

    static void chunk(ByteArrayOutputStream o, String key, String text) throws Exception {
        ByteArrayOutputStream d = new ByteArrayOutputStream();
        d.write(key.getBytes(StandardCharsets.UTF_8)); d.write(0); d.write(0); d.write(0); d.write(0); d.write(0);   // key, NUL, not compressed, method, empty language, empty translated key
        d.write(text.getBytes(StandardCharsets.UTF_8));
        byte[] data = d.toByteArray(), type = {'i', 'T', 'X', 't'};
        CRC32 crc = new CRC32(); crc.update(type); crc.update(data);
        int2(o, data.length); o.write(type); o.write(data); int2(o, (int) crc.getValue());
    }

    static void int2(ByteArrayOutputStream o, int v) { o.write(v >>> 24); o.write(v >>> 16); o.write(v >>> 8); o.write(v); }

    /** the text stored under `key` (iTXt without compression, or tEXt), null if absent */
    static String read(byte[] b, String key) {
        int p = 8;
        while (p + 12 <= b.length) {
            int len = ((b[p] & 255) << 24) | ((b[p + 1] & 255) << 16) | ((b[p + 2] & 255) << 8) | (b[p + 3] & 255);
            String type = new String(b, p + 4, 4, StandardCharsets.ISO_8859_1);
            if (len < 0 || p + 12 + len > b.length) return null;
            if (type.equals("IDAT")) return null;   // text chunks come before the picture data
            if (type.equals("iTXt") || type.equals("tEXt")) {
                int s = p + 8, e = s + len, z = s; while (z < e && b[z] != 0) z++;
                if (new String(b, s, z - s, StandardCharsets.UTF_8).equals(key)) {
                    int t = z + 1;
                    if (type.equals("iTXt")) { if (b[t] != 0) return null; t += 2; while (t < e && b[t] != 0) t++; t++; while (t < e && b[t] != 0) t++; t++; }   // flags, language, translated key
                    return new String(b, t, e - t, type.equals("iTXt") ? StandardCharsets.UTF_8 : StandardCharsets.ISO_8859_1);
                }
            }
            p += 12 + len;
        }
        return null;
    }

    /** settings of a picture: our JSON, or parsed from an A1111-style "parameters" text; null if the file has none */
    static JSONObject settings(byte[] png) {
        try {
            String j = read(png, "anima");
            if (j != null) return new JSONObject(j);
            String t = read(png, "parameters");
            if (t == null) return null;
            JSONObject m = new JSONObject();
            int i = t.indexOf("\nSteps:");
            m.put("prompt", (i >= 0 ? t.substring(0, i) : t).trim());
            java.util.regex.Matcher s = java.util.regex.Pattern.compile("Seed: (-?\\d+)").matcher(t), z = java.util.regex.Pattern.compile("Size: (\\d+)x(\\d+)").matcher(t);
            if (s.find()) m.put("seed", s.group(1));
            if (z.find()) { m.put("w", Integer.parseInt(z.group(1))); m.put("h", Integer.parseInt(z.group(2))); }
            return m;
        } catch (Exception e) { return null; }
    }
}
