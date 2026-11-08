package com.hexanima.npu;

import android.content.res.ColorStateList;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.widget.*;

/** Settings: language, theme, auto-load / idle unload of the model, screen on, quality-tag prefix. Everything is stored in the "anima" prefs and applied at once. */
public class Settings extends Base {
    static final String DEFAULT_PREFIX = "best quality, score_7, score_9, very aesthetic, ultra detailed, fair skin, high contrast, anime coloring, ";
    static final int[] IDLE = {0, 3, 5, 10, 30};

    static SharedPreferences prefs(Context c) { return c.getSharedPreferences("anima", 0); }
    static boolean autoload(Context c) { return prefs(c).getBoolean("autoload", false); }
    static int idleMin(Context c) { return prefs(c).getInt("idle", 10); }
    static boolean keepOn(Context c) { return prefs(c).getBoolean("keepon", true); }
    static String rawPrefix(Context c) { return prefs(c).getString("prefix", DEFAULT_PREFIX); }
    /** every step at full resolution (animad otherwise runs the first two of 4 steps on a half-size latent: ~30% faster, slightly less fine detail) */
    static boolean bestQuality(Context c) { return prefs(c).getBoolean("bestq", false); }
    /** sizes above 4096 patches on chips where they are not verified (V79 upstream gave noise above ~4800) */
    static boolean largeSizes(Context c) { return prefs(c).getBoolean("large", false); }
    static final int[] THREADS = {4, 6, 8};
    /** CPU threads of sd.cpp (text encoder and decoder run on the CPU on V81; 8 Elite Gen 5 has 2 prime + 6 performance cores) */
    static int threads(Context c) { return prefs(c).getInt("threads", Compat.qnnCommon() ? 4 : 6); }
    static final int[] WORKERS = {0, 4, 5};
    /** DSP worker threads per block (0 = the default 3, the V69 maximum: V69 has 4 HVX contexts, the main thread needs one) */
    static int dspWorkers(Context c) { return prefs(c).getInt("dspw", 0); }
    /** what goes before the prompt: trimmed, ending with a comma and a space; a single space when the user cleared it (anima.sh treats an empty PREFIX as "use the default") */
    static String prefix(Context c) { String p = rawPrefix(c).trim(); return p.isEmpty() ? " " : (p.endsWith(",") ? p : p + ",") + " "; }

    EditText prefixEdit;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        LinearLayout col = screen(L.s("Настройки", "Settings"), null);
        SharedPreferences p = prefs(this);

        LinearLayout c1 = card(col);
        c1.addView(choice(L.s("Язык", "Language"), L.NAMES[L.index()], v -> {
            builder().setTitle(L.s("Язык", "Language")).setSingleChoiceItems(L.NAMES, L.index(), (d, i) -> { d.dismiss(); L.set(this, L.CODES[i]); recreate(); }).show();
        }));
        String[] tn = {L.s("Как в системе", "Follow system"), L.s("Светлая", "Light"), L.s("Тёмная", "Dark")}, tv = {"system", "light", "dark"};
        int ti = java.util.Arrays.asList(tv).indexOf(p.getString("theme", "system"));
        c1.addView(choice(L.s("Тема", "Theme"), tn[Math.max(ti, 0)], v -> {
            builder().setTitle(L.s("Тема", "Theme")).setSingleChoiceItems(tn, Math.max(ti, 0), (d, i) -> { d.dismiss(); p.edit().putString("theme", tv[i]).apply(); recreate(); }).show();
        }));

        LinearLayout c2 = card(col);
        c2.addView(toggle(L.s("Загружать модель при запуске", "Load the model at start"), L.s("Первая картинка появится быстрее, но модель занимает ОЗУ", "The first picture is faster, but the model holds RAM"), autoload(this), on -> p.edit().putBoolean("autoload", on).apply()));
        String[] in = new String[IDLE.length];
        for (int i = 0; i < IDLE.length; i++) in[i] = IDLE[i] == 0 ? L.s("Никогда", "Never") : L.f("%d мин", "%d min", IDLE[i]);
        int ii = Math.max(0, java.util.Arrays.binarySearch(IDLE, idleMin(this)));
        c2.addView(choice(L.s("Выгружать при простое", "Unload when idle"), in[ii], v -> {
            builder().setTitle(L.s("Выгружать при простое", "Unload when idle")).setSingleChoiceItems(in, ii, (d, i) -> { d.dismiss(); p.edit().putInt("idle", IDLE[i]).apply(); recreate(); }).show();
        }));
        TextView idleNote = text(L.s("Освобождает ОЗУ, если вы долго ничего не генерируете", "Frees RAM when you have not generated anything for a while"), 12, DIM, false); idleNote.setPadding(dp(2), 0, 0, dp(4));
        c2.addView(idleNote);

        LinearLayout cq = card(col);
        TextView qh = text(L.s("Скорость и качество", "Speed and quality"), 15, TXT, true); qh.setPadding(dp(2), dp(6), 0, dp(2)); cq.addView(qh);
        cq.addView(toggle(L.s("Наилучшее качество", "Best quality"), L.s("Все шаги в полном разрешении: чуть больше деталей, примерно на 30% дольше", "Every step at full resolution: a little more fine detail, about 30% slower"), bestQuality(this), on -> p.edit().putBoolean("bestq", on).apply()));
        if (Compat.chipOk() && !Compat.dspArch().equals("v69")) {
            cq.addView(toggle(L.s("Большие размеры (эксперимент)", "Large sizes (experimental)"), L.s("Больше 4096 патчей (больше 1024×1024): на этом чипе не проверено, возможен шум", "Above 4096 patches (larger than 1024×1024): not verified on this chip, may come out as noise"), largeSizes(this), on -> p.edit().putBoolean("large", on).apply()));
            String[] tn2 = new String[THREADS.length]; for (int i = 0; i < THREADS.length; i++) tn2[i] = String.valueOf(THREADS[i]);
            int ti2 = Math.max(0, java.util.Arrays.binarySearch(THREADS, threads(this)));
            cq.addView(choice(L.s("Потоки CPU", "CPU threads"), tn2[ti2], v -> builder().setTitle(L.s("Потоки CPU (текст и декодер)", "CPU threads (text encoder and decoder)")).setSingleChoiceItems(tn2, ti2, (d, i) -> { d.dismiss(); p.edit().putInt("threads", THREADS[i]).apply(); recreate(); }).show()));
            String[] wn = {L.s("3 (по умолчанию)", "3 (default)"), "4", "5"};
            int wi = Math.max(0, java.util.Arrays.binarySearch(WORKERS, dspWorkers(this)));
            cq.addView(choice(L.s("Потоки NPU (эксперимент)", "NPU threads (experimental)"), wn[wi], v -> builder().setTitle(L.s("Потоки NPU на блок", "NPU worker threads per block")).setSingleChoiceItems(wn, wi, (d, i) -> { d.dismiss(); p.edit().putInt("dspw", WORKERS[i]).apply(); recreate(); }).show()));
        }
        TextView qn = text(L.s("Изменения применяются со следующей картинки (модель один раз перезагрузится)", "Changes apply from the next picture (the model reloads once)"), 12, DIM, false); qn.setPadding(dp(2), 0, 0, dp(4)); cq.addView(qn);

        LinearLayout c3 = card(col);
        c3.addView(toggle(L.s("Не гасить экран при генерации", "Keep the screen on while generating"), "", keepOn(this), on -> p.edit().putBoolean("keepon", on).apply()));
        if (Moss.is(p.getString("model", ""))) return;   // MOSS-Image takes plain sentences: the quality-tag prefix is for the Anima models only, so the setting is hidden while MOSS is selected
        TextView pl = text(L.s("Теги качества перед описанием (модели Anima)", "Quality tags before the prompt (Anima models)"), 15, TXT, true); pl.setPadding(dp(2), dp(14), 0, dp(6)); c3.addView(pl);
        prefixEdit = field("");
        prefixEdit.setText(rawPrefix(this)); prefixEdit.setTextSize(13); prefixEdit.setMinLines(2); prefixEdit.setGravity(Gravity.TOP);
        prefixEdit.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        c3.addView(prefixEdit);
        Button reset = button(L.s("Сбросить по умолчанию", "Reset to default"), FIELD, ACC); reset.setTextSize(13);
        reset.setOnClickListener(v -> prefixEdit.setText(DEFAULT_PREFIX));
        LinearLayout.LayoutParams rl = new LinearLayout.LayoutParams(-2, dp(38)); rl.topMargin = dp(8);
        c3.addView(reset, rl);
    }

    @Override protected void onPause() { super.onPause(); if (prefixEdit != null) prefs(this).edit().putString("prefix", prefixEdit.getText().toString()).apply(); }

    View choice(String title, String value, View.OnClickListener l) {
        LinearLayout r = new LinearLayout(this); r.setGravity(Gravity.CENTER_VERTICAL); r.setPadding(dp(2), dp(12), dp(2), dp(12));
        r.addView(text(title, 15, TXT, true), new LinearLayout.LayoutParams(0, -2, 1));
        r.addView(text(value + "  ▾", 14, ACC, false));
        r.setOnClickListener(l);
        return r;
    }

    interface OnToggle { void on(boolean v); }

    View toggle(String title, String sub, boolean on, OnToggle cb) {
        LinearLayout r = new LinearLayout(this); r.setGravity(Gravity.CENTER_VERTICAL); r.setPadding(dp(2), dp(10), dp(2), dp(10));
        LinearLayout t = new LinearLayout(this); t.setOrientation(LinearLayout.VERTICAL);
        t.addView(text(title, 15, TXT, true));
        if (!sub.isEmpty()) { TextView s = text(sub, 12, DIM, false); s.setPadding(0, dp(2), 0, 0); t.addView(s); }
        r.addView(t, new LinearLayout.LayoutParams(0, -2, 1));
        Switch sw = new Switch(this); sw.setChecked(on);
        sw.setThumbTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked}, {}}, new int[]{ACC, DIM}));
        sw.setTrackTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked}, {}}, new int[]{(ACC & 0x00FFFFFF) | 0x66000000, (DIM & 0x00FFFFFF) | 0x44000000}));
        sw.setOnCheckedChangeListener((v, c) -> cb.on(c));
        r.addView(sw);
        r.setOnClickListener(v -> sw.toggle());
        return r;
    }
}
