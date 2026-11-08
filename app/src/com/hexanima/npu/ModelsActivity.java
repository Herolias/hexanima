package com.hexanima.npu;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.WindowManager;
import android.widget.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** Model catalog: a card per model (size, speed at the base resolution, installed state) with download / delete. The download itself runs on a thread with a foreground service,
 *  its progress lives in statics so that leaving this screen does not lose it. The shared files (text encoder, VAE) come with the first model. */
public class ModelsActivity extends Base {
    static volatile boolean downloading;
    static volatile String dlId = "", dlMsg = "";
    static volatile long dlDone, dlTotal;
    static final AtomicBoolean cancel = new AtomicBoolean();
    static volatile String deleted = "";   // last deleted model id: Main unloads the server if it holds it

    Models store;
    final Handler ui = new Handler(Looper.getMainLooper());
    final ExecutorService io = Executors.newSingleThreadExecutor();
    LinearLayout list;
    TextView info;
    EditText url;
    List<Models.Item> items = new ArrayList<>();
    final java.util.Map<String, ProgressBar> bars = new java.util.HashMap<>();
    final java.util.Map<String, TextView> msgs = new java.util.HashMap<>();
    boolean wasDownloading;
    final Runnable tick = new Runnable() {
        @Override public void run() {
            if (downloading) {
                ProgressBar pb = bars.get(dlId); TextView tv = msgs.get(dlId);
                if (pb != null && dlTotal > 0) pb.setProgress((int) (1000 * dlDone / dlTotal));
                if (tv != null) tv.setText(dlMsg);
            } else if (wasDownloading) { wasDownloading = false; getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); render(); }
            if (downloading) wasDownloading = true;
            ui.postDelayed(this, 400);
        }
    };

    String prefBase() { return getSharedPreferences("anima", 0).getString("base", Main.DEFAULT_BASE); }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        store = new Models(getFilesDir());
        Button src = button(L.s("Источник", "Source"), CARD, DIM); src.setTextSize(12);
        LinearLayout col = screen(L.s("Модели", "Models"), src);
        info = text("", 12, DIM, false); info.setPadding(dp(2), 0, 0, dp(10));
        col.addView(info);
        url = field(L.s("Адрес каталога (…/resolve/main/)", "Catalog URL (…/resolve/main/)")); url.setText(prefBase()); url.setTextSize(12); url.setBackground(bg(CARD, 14)); url.setVisibility(View.GONE);
        col.addView(url);
        src.setOnClickListener(v -> { boolean show = url.getVisibility() != View.VISIBLE; url.setVisibility(show ? View.VISIBLE : View.GONE); if (!show) load(); });
        list = new LinearLayout(this); list.setOrientation(LinearLayout.VERTICAL);
        col.addView(list);
        String soc = Compat.soc();
        boolean ok = Compat.chipOk();
        info.setText((ok ? L.f("Чип %s · подходит (%s)", "Chip %s · supported (%s)", soc, Compat.archLabel()) : L.f("Чип %s · проверено только на Snapdragon 8 Gen 1 / 8+ Gen 1", "Chip %s · tested only on Snapdragon 8 Gen 1 / 8+ Gen 1", soc))
            + (ok && !Compat.qnnCommon() ? "\n" + L.s("Текстовый энкодер и декодер работают на CPU, DiT — на NPU", "Text encoder and decoder run on the CPU, the DiT on the NPU") : "")
            + "\n" + L.f("Свободно %.1f ГБ", "Free %.1f GB", getFilesDir().getUsableSpace() / 1e9));
        if (downloading) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        load();
    }

    @Override protected void onResume() { super.onResume(); ui.post(tick); }
    @Override protected void onPause() { super.onPause(); ui.removeCallbacks(tick); }
    @Override protected void onDestroy() { super.onDestroy(); io.shutdownNow(); }

    void load() {
        list.removeAllViews();
        list.addView(text(L.s("Загружаю каталог…", "Loading catalog…"), 13, DIM, false));
        final String base = url.getText().toString().trim().replaceAll("/*$", "") + "/";
        getSharedPreferences("anima", 0).edit().putString("base", base).apply();
        io.submit(() -> {
            try { final List<Models.Item> cat = store.catalog(base); if (Compat.chipOk() && !Compat.mossOk()) cat.removeIf(it -> it.moss); /* MOSS-Image decodes through a QNN context that exists for V69 only */ for (Models.Item it : cat) store.refreshMeta(it); ui.post(() -> { items = cat; render(); }); }
            catch (Exception e) { ui.post(() -> { list.removeAllViews(); list.addView(text(L.f("Каталог недоступен: %s", "Catalog unavailable: %s", e.getMessage()), 13, BAD, false)); }); }
        });
    }

    static String gb(long b) { return String.format("%.1f", b / 1e9); }

    void render() {
        list.removeAllViews(); bars.clear(); msgs.clear();
        final String base = prefBase();
        final boolean needCommonAnima = !store.commonInstalled();
        for (final Models.Item it : items) {
            final boolean needCommon = needCommonAnima && !it.moss;
            final boolean have = store.modelInstalled(it.id);
            final boolean busyThis = downloading && it.id.equals(dlId);
            LinearLayout c = card(list);
            LinearLayout top = new LinearLayout(this); top.setGravity(android.view.Gravity.CENTER_VERTICAL);
            if (have) { TextView ok = text("✓", 18, OK, true); ok.setPadding(0, 0, dp(8), 0); top.addView(ok); }
            top.addView(text(it.name, 17, TXT, true), new LinearLayout.LayoutParams(0, -2, 1));
            c.addView(top);
            String meta = (it.params.isEmpty() ? "" : it.params + (it.quant.isEmpty() ? "" : " " + it.quant) + "  ·  ") + (it.speed > 0 ? L.f("~%d с на 512×768", "~%d s at 512×768", it.speed) + "  ·  " : "") + L.f("%s ГБ", "%s GB", gb(it.size));
            TextView m = text(meta, 13, ACC, false); m.setPadding(0, dp(4), 0, 0); c.addView(m);
            String d = L.s(it.desc, it.descEn);
            if (!d.isEmpty()) { TextView dv = text(d, 13, DIM, false); dv.setPadding(0, dp(4), 0, 0); c.addView(dv); }
            if (!it.author.isEmpty()) { TextView a = text(L.f("Автор: %s", "By %s", it.author), 12, DIM, false); a.setPadding(0, dp(4), 0, 0); c.addView(a); }
            if (busyThis) {
                ProgressBar pb = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); pb.setMax(1000);
                pb.setProgressTintList(android.content.res.ColorStateList.valueOf(ACC)); pb.setProgressBackgroundTintList(android.content.res.ColorStateList.valueOf(FIELD));
                LinearLayout.LayoutParams pl = new LinearLayout.LayoutParams(-1, dp(5)); pl.topMargin = dp(10); c.addView(pb, pl); bars.put(it.id, pb);
                TextView tv = text(dlMsg, 12, DIM, false); tv.setPadding(0, dp(6), 0, 0); c.addView(tv); msgs.put(it.id, tv);
                Button x = button(L.s("Отменить", "Cancel"), FIELD, TXT); x.setOnClickListener(v -> cancel.set(true));
                LinearLayout.LayoutParams xl = new LinearLayout.LayoutParams(-1, dp(42)); xl.topMargin = dp(8); c.addView(x, xl);
            } else {
                String label = have ? L.s("Удалить", "Delete") : L.f("Скачать  %s ГБ", "Download  %s GB", gb(it.size + (needCommon ? store.commonBytes() : 0)));
                Button bt = button(label, have ? FIELD : ACC, have ? BAD : ON_ACC); bt.setEnabled(!downloading); bt.setAlpha(downloading ? .5f : 1f);
                bt.setOnClickListener(v -> {
                    if (have) { store.delete(it.id); deleted = it.id; render(); }
                    else start(base, it, needCommon);
                });
                LinearLayout.LayoutParams bl = new LinearLayout.LayoutParams(-1, dp(44)); bl.topMargin = dp(10);
                c.addView(bt, bl);
            }
        }
        if (items.isEmpty()) list.addView(text(L.s("Каталог пуст", "The catalog is empty"), 13, DIM, false));
        else if (!store.commonInstalled() && hasAnima()) { TextView n = text(L.f("К первой модели добавятся общие файлы (%s ГБ): текстовый энкодер и VAE. Они качаются один раз.", "The first model also fetches the shared files (%s GB): text encoder and VAE, once.", gb(store.commonBytes())), 12, DIM, false); n.setPadding(dp(2), 0, 0, 0); list.addView(n); }
    }

    boolean hasAnima() { for (Models.Item it : items) if (!it.moss) return true; return false; }

    void start(String base, Models.Item it, boolean needCommon) {
        if (!Compat.chipAllowed(this)) {   // an unsupported chip: the models will not run, so ask before 4 GB are downloaded
            builder().setMessage(L.f("Чип %s не поддерживается: модели рассчитаны на Snapdragon 8 Gen 1 / 8+ Gen 1. Всё равно скачать?", "Chip %s is not supported: the models are built for Snapdragon 8 Gen 1 / 8+ Gen 1. Download anyway?", Compat.soc()))
                .setPositiveButton(L.s("Скачать", "Download"), (d, w) -> { Compat.allowChip(this); begin(base, it, needCommon); }).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
            return;
        }
        if (it.moss && Compat.ramGb(this) < 10.5) {   // text encoder 6 GB and the DiT stream through RAM: 12 GB phones only
            builder().setMessage(L.f("В телефоне %.1f ГБ ОЗУ: для этой модели рекомендуется 12 ГБ. Всё равно скачать?", "This phone has %.1f GB of RAM: 12 GB is recommended for this model. Download anyway?", Compat.ramGb(this)))
                .setPositiveButton(L.s("Скачать", "Download"), (d, w) -> begin(base, it, needCommon)).setNegativeButton(L.s("Отмена", "Cancel"), null).show();
            return;
        }
        begin(base, it, needCommon);
    }

    void begin(String base, Models.Item it, boolean needCommon) {
        long need = it.size + (needCommon ? store.commonBytes() : 0);
        if (getFilesDir().getUsableSpace() < need + 300_000_000L) { Toast.makeText(this, L.f("Не хватает места: нужно %s ГБ", "Not enough space: need %s GB", gb(need)), Toast.LENGTH_LONG).show(); return; }
        if (downloading) return;
        downloading = true; dlId = it.id; dlDone = 0; dlTotal = need; dlMsg = L.s("Подключаюсь…", "Connecting…"); cancel.set(false);
        final android.content.Context app = getApplicationContext();
        app.startForegroundService(new Intent(app, DownloadService.class));
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        render();
        final long[] last = {0};
        new Thread(() -> {
            Models.Progress pg = (d, tot, what) -> {
                long now = System.currentTimeMillis(); if (now - last[0] < 300) return; last[0] = now;
                dlDone = d; dlTotal = tot; dlMsg = L.f("%s / %s ГБ  ·  %s", "%s / %s GB  ·  %s", gb(d), gb(tot), what);
            };
            String err = null;
            try { if (!it.moss) store.installCommon(base, pg, cancel); store.installModel(base, it, pg, cancel); }
            catch (java.io.InterruptedIOException e) { err = L.s("Загрузка отменена: файлы сохранены, можно продолжить", "Cancelled: the files are kept, you can resume"); }
            catch (Exception e) { err = L.f("Ошибка: %s", "Error: %s", e.getMessage()); }
            final String msg = err;
            downloading = false;
            app.stopService(new Intent(app, DownloadService.class));
            if (msg != null) ui.post(() -> Toast.makeText(app, msg, Toast.LENGTH_LONG).show());
        }).start();
    }
}
