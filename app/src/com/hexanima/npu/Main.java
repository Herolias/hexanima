package com.hexanima.npu;

import android.app.AlertDialog;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.net.Uri;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Outline;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewOutlineProvider;
import android.view.WindowManager;
import android.widget.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONArray;
import org.json.JSONObject;

/** Anima on the NPU: load/unload the model (animad server via anima.sh) and generate. No root: the bundled native binaries run from the app's own uid. */
public class Main extends Base {
    static final String DEFAULT_BASE = "https://huggingface.co/ayanoby/anima-npu-packs/resolve/main/";
    static final int STEPS = 4;   // the models are distilled for 4 steps (see Info)
    /** popular sizes, all valid for the NPU pipeline (multiples of 16, patch count divisible by 32, at most 6144); grouped by aspect ratio, a portrait and its landscape twin next to each other */
    static final int[][] SIZES = {{512, 512}, {640, 640}, {768, 768}, {1024, 1024}, {512, 768}, {768, 512}, {768, 1152}, {1152, 768}, {1024, 1536}, {1536, 1024}, {768, 1024}, {1024, 768}, {960, 1280}, {1280, 960},
        {576, 1024}, {1024, 576}, {640, 1152}, {1152, 640}, {960, 1536}, {1536, 960}};
    static volatile boolean visible;   // the main screen is in front (otherwise a finished picture is announced by a notification)

    String D, BIN;   // work dir (files/anima) and the native lib dir with the bundled executables
    Models store;
    java.util.List<String> ids = new java.util.ArrayList<>();   // installed model ids
    volatile int resW = 512, resH = 768;
    final Handler ui = new Handler(Looper.getMainLooper());
    final ExecutorService ex = Executors.newSingleThreadExecutor();
    final ExecutorService ex2 = Executors.newSingleThreadExecutor();   // the Stop button
    volatile boolean loaded, busy, cancelled;
    String model = "";   // id of the selected installed model; anima.sh gets MID/MDIR
    int count = 1;       // pictures per press of Generate
    volatile int curIdx, curN = 1;
    boolean statusSeen;
    int tick;

    TextView chipName, chipDot, state, status, saved, placeholder, ramTxt, warnTxt, sizeText;
    LinearLayout customRow, imgBox;
    Button loadBtn, genBtn, countBtn;
    ImageButton dlBtn, shareBtn, favBtn, againBtn;
    MultiAutoCompleteTextView prompt;
    EditText seed, wEdit, hEdit;
    LinearLayout loraBox; TextView loraName, loraInfo, strText; SeekBar strBar; Button loraPick, loraDrop;
    LinearLayout mossBox, fullBox; Button modeFast, modeFull; TextView modeInfo, stepText, cfgText; SeekBar stepBar, cfgBar; EditText negEdit;
    ProgressBar bar, ramBar;
    ImageView img;
    File lastFile;
    boolean customOpen;

    String mp() { return "MID=" + model + " MDIR=m_" + model + " IDLE=" + Settings.idleMin(this) * 60 + " " + Compat.env(this); }

    GradientDrawable gradient(boolean enabled) {
        GradientDrawable g = new GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT, new int[]{enabled ? ACC : 0xFF3A4158, enabled ? ACC2 : 0xFF3A4158});
        g.setCornerRadius(dp(18)); return g;
    }
    SharedPreferences prefs() { return getSharedPreferences("anima", 0); }
    void setFavIcon(boolean on) { favBtn.setImageResource(getResources().getIdentifier(on ? "ic_star" : "ic_star_off", "drawable", getPackageName())); favBtn.setImageTintList(on ? null : ColorStateList.valueOf(TXT)); }

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        D = new File(getFilesDir(), "anima").getPath(); BIN = getApplicationInfo().nativeLibraryDir;
        store = new Models(getFilesDir());
        main.env = pollSh.env = ctlSh.env = new String[]{D, new File(getFilesDir(), "sd").getPath(), BIN};
        installScript();
        getWindow().setNavigationBarColor(CARD);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        LinearLayout frame = new LinearLayout(this);
        frame.setOrientation(LinearLayout.VERTICAL);
        frame.setBackgroundColor(BG);
        frame.setFitsSystemWindows(true);
        ScrollView sv = new ScrollView(this);
        sv.setFillViewport(true);
        sv.setVerticalScrollBarEnabled(false);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(12), dp(16), dp(20));
        sv.addView(root, new ViewGroup.LayoutParams(-1, -2));
        frame.addView(sv, new LinearLayout.LayoutParams(-1, 0, 1));

        // header: title, history, menu
        LinearLayout tr = new LinearLayout(this);
        tr.setGravity(Gravity.CENTER_VERTICAL);
        tr.addView(text("HexAnima", 28, TXT, true), new LinearLayout.LayoutParams(0, -2, 1));
        Button hist = button(L.s("История", "History"), CARD, TXT);
        hist.setTextSize(13);
        hist.setOnClickListener(v -> startActivityForResult(new Intent(this, History.class), 7));
        tr.addView(hist, new LinearLayout.LayoutParams(-2, dp(40)));
        Button more = button("⋯", CARD, TXT);
        more.setTextSize(20);
        more.setOnClickListener(v -> menu(v));
        LinearLayout.LayoutParams ml = new LinearLayout.LayoutParams(dp(48), dp(40)); ml.leftMargin = dp(6);
        tr.addView(more, ml);
        root.addView(tr);
        ramTxt = text("", 12, DIM, false);
        ramTxt.setPadding(dp(2), dp(10), 0, dp(4));
        root.addView(ramTxt);
        ramBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        ramBar.setMax(1000);
        ramBar.setProgressBackgroundTintList(ColorStateList.valueOf(CARD));
        root.addView(ramBar, new LinearLayout.LayoutParams(-1, dp(5)));
        warnTxt = text("", 12, WARN, false);   // thermal / battery / unsupported chip
        warnTxt.setPadding(dp(2), dp(6), 0, 0);
        warnTxt.setVisibility(View.GONE);
        root.addView(warnTxt);

        // model chip + load/unload, then the state line
        LinearLayout head = new LinearLayout(this);
        head.setGravity(Gravity.CENTER_VERTICAL);
        head.setPadding(0, dp(12), 0, 0);
        LinearLayout mc = new LinearLayout(this);
        mc.setGravity(Gravity.CENTER_VERTICAL);
        mc.setBackground(bg(CARD, 20));
        mc.setPadding(dp(14), dp(10), dp(14), dp(10));
        chipDot = text("●", 12, DIM, false);
        chipName = text("…", 14, TXT, true);
        chipName.setPadding(dp(8), 0, dp(6), 0); chipName.setMaxLines(2); chipName.setEllipsize(android.text.TextUtils.TruncateAt.END);
        mc.addView(chipDot); mc.addView(chipName, new LinearLayout.LayoutParams(0, -2, 1)); mc.addView(text("▾", 12, DIM, false));
        mc.setOnClickListener(v -> pickModel());
        head.addView(mc, new LinearLayout.LayoutParams(0, -2, 1));
        loadBtn = button(L.s("Загрузить", "Load"), CARD, TXT);
        loadBtn.setTextSize(13);
        loadBtn.setOnClickListener(v -> toggleModel());
        LinearLayout.LayoutParams lb = new LinearLayout.LayoutParams(-2, dp(42)); lb.leftMargin = dp(8);
        head.addView(loadBtn, lb);
        root.addView(head);
        state = text(L.s("Проверяю…", "Checking…"), 12, DIM, false);
        state.setPadding(dp(4), dp(6), 0, dp(12));
        root.addView(state);

        // result card
        imgBox = new LinearLayout(this);
        imgBox.setOrientation(LinearLayout.VERTICAL);
        imgBox.setGravity(Gravity.CENTER);
        imgBox.setBackground(bg(CARD, 24));
        imgBox.setClipToOutline(true);
        imgBox.setOutlineProvider(new ViewOutlineProvider() {
            @Override public void getOutline(View v, Outline o) { o.setRoundRect(0, 0, v.getWidth(), v.getHeight(), dp(24)); }
        });
        placeholder = text(L.s("Здесь появится картинка", "Your picture will appear here"), 14, DIM, false);
        placeholder.setGravity(Gravity.CENTER);
        placeholder.setPadding(dp(24), dp(64), dp(24), dp(64));
        imgBox.addView(placeholder, new LinearLayout.LayoutParams(-1, -2));
        img = new ImageView(this);
        img.setAdjustViewBounds(true);
        img.setScaleType(ImageView.ScaleType.FIT_CENTER);
        img.setVisibility(View.GONE);
        img.setOnClickListener(v -> { if (lastFile != null) Gallery.view(this, lastFile); });
        imgBox.addView(img, new LinearLayout.LayoutParams(-1, -2));
        root.addView(imgBox, new LinearLayout.LayoutParams(-1, -2));
        LinearLayout resRow = new LinearLayout(this);
        resRow.setGravity(Gravity.CENTER_VERTICAL);
        saved = text("", 12, DIM, false);
        saved.setPadding(dp(4), dp(8), dp(8), 0);
        resRow.addView(saved, new LinearLayout.LayoutParams(0, -2, 1));
        againBtn = icon("ic_refresh", CARD);
        againBtn.setOnClickListener(v -> generate(true));   // the same prompt with a new random seed
        favBtn = icon("ic_star_off", CARD);
        favBtn.setOnClickListener(v -> { if (lastFile == null) return; boolean nv = !Gallery.fav(lastFile); Gallery.setFav(lastFile, nv); setFavIcon(nv); });
        dlBtn = icon("ic_download", CARD);
        dlBtn.setOnClickListener(v -> { if (lastFile == null) return; try { Gallery.export(this, lastFile); Toast.makeText(this, L.s("Сохранено в галерею: Pictures/Anima", "Saved to gallery: Pictures/Anima"), Toast.LENGTH_SHORT).show(); } catch (Exception e) { Toast.makeText(this, e.toString(), Toast.LENGTH_LONG).show(); } });
        shareBtn = icon("ic_share", CARD);
        shareBtn.setOnClickListener(v -> { if (lastFile != null) Gallery.share(this, lastFile); });
        for (ImageButton x : new ImageButton[]{againBtn, favBtn, dlBtn, shareBtn}) {
            x.setVisibility(View.GONE);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(46), dp(40)); lp.leftMargin = dp(6);
            resRow.addView(x, lp);
        }
        root.addView(resRow);

        // prompt
        LinearLayout pl = new LinearLayout(this);
        pl.setGravity(Gravity.CENTER_VERTICAL);
        pl.addView(label(L.s("ОПИСАНИЕ", "PROMPT")), new LinearLayout.LayoutParams(0, -2, 1));
        TextView tpl = text(L.s("Шаблоны", "Templates"), 12, ACC, true);
        tpl.setPadding(dp(8), dp(14), dp(8), dp(6));
        tpl.setOnClickListener(v -> templates());
        pl.addView(tpl);
        TextView clr = text(L.s("Очистить", "Clear"), 12, ACC, true);
        clr.setPadding(dp(8), dp(14), dp(2), dp(6));
        clr.setOnClickListener(v -> prompt.setText(""));
        pl.addView(clr);
        root.addView(pl);
        prompt = new MultiAutoCompleteTextView(this);   // Danbooru tag suggestions while typing, comma separated
        prompt.setHint("1girl, smile, outdoors, cherry blossoms…");
        prompt.setHintTextColor(DIM); prompt.setTextColor(TXT); prompt.setTextSize(16);
        prompt.setMinLines(3);
        prompt.setGravity(Gravity.TOP);
        prompt.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        prompt.setBackground(bg(CARD, 18));
        prompt.setPadding(dp(16), dp(14), dp(16), dp(14));
        prompt.setThreshold(2);
        prompt.setTokenizer(new MultiAutoCompleteTextView.CommaTokenizer());
        prompt.setDropDownBackgroundDrawable(bg(CARD, 12));
        prompt.setDropDownHeight(dp(240));
        root.addView(prompt, new LinearLayout.LayoutParams(-1, -2));
        loadTags();

        // size: a drop-down button that opens a two-column picker, and the portrait <-> landscape swap
        root.addView(label(L.s("РАЗМЕР", "SIZE")));
        LinearLayout sizeRow = new LinearLayout(this);
        sizeRow.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout sizeBtn = new LinearLayout(this);
        sizeBtn.setGravity(Gravity.CENTER_VERTICAL); sizeBtn.setBackground(bg(CARD, 16)); sizeBtn.setPadding(dp(16), 0, dp(16), 0);
        sizeText = text("", 15, TXT, true);
        sizeBtn.addView(sizeText, new LinearLayout.LayoutParams(0, -2, 1)); sizeBtn.addView(text("▾", 14, DIM, false));
        sizeBtn.setOnClickListener(v -> pickSize());
        sizeRow.addView(sizeBtn, new LinearLayout.LayoutParams(0, dp(48), 1));
        Button swap = button("⇄", CARD, TXT);
        swap.setTextSize(18);
        swap.setOnClickListener(v -> {
            int w = resW, h = resH;
            if (customRow.getVisibility() == View.VISIBLE) try { w = Integer.parseInt(wEdit.getText().toString().trim()); h = Integer.parseInt(hEdit.getText().toString().trim()); } catch (NumberFormatException e) { }
            setRes(h, w);
        });
        LinearLayout.LayoutParams sw = new LinearLayout.LayoutParams(dp(56), dp(48)); sw.leftMargin = dp(8);
        sizeRow.addView(swap, sw);
        root.addView(sizeRow);
        customRow = new LinearLayout(this);
        customRow.setGravity(Gravity.CENTER_VERTICAL);
        customRow.setPadding(0, dp(10), 0, 0);
        wEdit = field(L.s("ширина", "width")); hEdit = field(L.s("высота", "height"));
        wEdit.setInputType(InputType.TYPE_CLASS_NUMBER); hEdit.setInputType(InputType.TYPE_CLASS_NUMBER);
        wEdit.setBackground(bg(CARD, 14)); hEdit.setBackground(bg(CARD, 14));
        customRow.addView(wEdit, new LinearLayout.LayoutParams(0, -2, 1));
        TextView times = text("×", 18, DIM, false); times.setGravity(Gravity.CENTER);
        customRow.addView(times, new LinearLayout.LayoutParams(dp(36), -2));
        customRow.addView(hEdit, new LinearLayout.LayoutParams(0, -2, 1));
        customRow.setVisibility(View.GONE);
        root.addView(customRow);

        // seed
        root.addView(label(L.s("SEED  ·  -1 = случайный", "SEED  ·  -1 = random")));
        LinearLayout sr = new LinearLayout(this);
        sr.setGravity(Gravity.CENTER_VERTICAL);
        seed = field("-1");
        seed.setText("-1");
        seed.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED);
        seed.setBackground(bg(CARD, 14));
        sr.addView(seed, new LinearLayout.LayoutParams(0, -2, 1));
        Button dice = button("🎲", CARD, TXT);
        dice.setOnClickListener(v -> seed.setText(String.valueOf(new Random().nextInt(1 << 30))));
        LinearLayout.LayoutParams dl = new LinearLayout.LayoutParams(dp(52), dp(44)); dl.leftMargin = dp(8);
        sr.addView(dice, dl);
        root.addView(sr);

        // MOSS-Image sampler: Fast = the rapid-moss 4-step adapter (distilled: no CFG, no negative prompt); Full = the model itself with the official sampler (steps, CFG, negative prompt)
        mossBox = new LinearLayout(this); mossBox.setOrientation(LinearLayout.VERTICAL);
        mossBox.addView(label(L.s("РЕЖИМ", "MODE")));
        LinearLayout mr = new LinearLayout(this);
        modeFast = button(L.s("Быстрый · 4 шага", "Fast · 4 steps"), CARD, TXT); modeFast.setOnClickListener(v -> setMossFull(false));
        modeFull = button(L.s("Полная модель", "Full model"), CARD, TXT); modeFull.setOnClickListener(v -> setMossFull(true));
        mr.addView(modeFast, new LinearLayout.LayoutParams(0, dp(44), 1));
        LinearLayout.LayoutParams mfl = new LinearLayout.LayoutParams(0, dp(44), 1); mfl.leftMargin = dp(8); mr.addView(modeFull, mfl);
        mossBox.addView(mr);
        modeInfo = text("", 12, DIM, false); modeInfo.setPadding(dp(4), dp(6), dp(4), 0); mossBox.addView(modeInfo);
        fullBox = new LinearLayout(this); fullBox.setOrientation(LinearLayout.VERTICAL);
        stepBar = new SeekBar(this); stepBar.setMax(Moss.FULL_STEPS_MAX - Moss.FULL_STEPS_MIN);
        stepBar.setProgress(Math.max(0, Math.min(stepBar.getMax(), prefs().getInt("moss_steps", Moss.FULL_STEPS_DEF) - Moss.FULL_STEPS_MIN)));
        stepText = text("", 14, TXT, true); fullBox.addView(seekRow(L.s("Шаги", "Steps"), stepBar, stepText));
        cfgBar = new SeekBar(this); cfgBar.setMax(28);   // CFG 1.0 .. 8.0 in 0.25
        cfgBar.setProgress(Math.max(0, Math.min(28, Math.round((prefs().getFloat("moss_cfg", Moss.CFG_DEF) - 1f) * 4))));
        cfgText = text("", 14, TXT, true); fullBox.addView(seekRow("CFG", cfgBar, cfgText));
        SeekBar.OnSeekBarChangeListener sl = new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar b, int p, boolean u) { showFullValues(); if (u) prefs().edit().putInt("moss_steps", mossSteps()).putFloat("moss_cfg", mossCfg()).apply(); }
            public void onStartTrackingTouch(SeekBar b) { }
            public void onStopTrackingTouch(SeekBar b) { }
        };
        stepBar.setOnSeekBarChangeListener(sl); cfgBar.setOnSeekBarChangeListener(sl);
        fullBox.addView(label(L.s("НЕГАТИВНОЕ ОПИСАНИЕ  ·  необязательно", "NEGATIVE PROMPT  ·  optional")));
        negEdit = field(L.s("например: размыто, лишние пальцы, текст", "e.g. blurry, extra fingers, text"));
        negEdit.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE); negEdit.setMinLines(2); negEdit.setGravity(Gravity.TOP);
        negEdit.setBackground(bg(CARD, 14)); negEdit.setText(prefs().getString("moss_neg", ""));
        fullBox.addView(negEdit, new LinearLayout.LayoutParams(-1, -2));
        mossBox.addView(fullBox);
        root.addView(mossBox);
        setMossFull(prefs().getBoolean("moss_full", false));

        // MOSS-Image: user LoRA (picked file is checked at once; strength slider; folded with the turbo adapter on first use)
        loraBox = new LinearLayout(this); loraBox.setOrientation(LinearLayout.VERTICAL);
        loraBox.addView(label(L.s("LORA  ·  необязательно", "LORA  ·  optional")));
        LinearLayout lr = new LinearLayout(this); lr.setGravity(Gravity.CENTER_VERTICAL);
        loraName = text(L.s("не выбрана", "none"), 14, TXT, false); loraName.setSingleLine(true); loraName.setEllipsize(android.text.TextUtils.TruncateAt.MIDDLE);
        loraName.setBackground(bg(CARD, 16)); loraName.setPadding(dp(16), dp(12), dp(16), dp(12));
        lr.addView(loraName, new LinearLayout.LayoutParams(0, -2, 1));
        loraPick = button(L.s("Выбрать", "Choose"), CARD, TXT); loraPick.setOnClickListener(v -> pickLora());
        LinearLayout.LayoutParams pl2 = new LinearLayout.LayoutParams(-2, dp(44)); pl2.leftMargin = dp(8); lr.addView(loraPick, pl2);
        loraDrop = button("✕", CARD, TXT); loraDrop.setOnClickListener(v -> dropLora());
        LinearLayout.LayoutParams dl2 = new LinearLayout.LayoutParams(dp(48), dp(44)); dl2.leftMargin = dp(8); lr.addView(loraDrop, dl2);
        loraBox.addView(lr);
        loraInfo = text("", 12, DIM, false); loraInfo.setPadding(dp(4), dp(6), dp(4), 0); loraBox.addView(loraInfo);
        LinearLayout sr2 = new LinearLayout(this); sr2.setGravity(Gravity.CENTER_VERTICAL); sr2.setPadding(0, dp(6), 0, 0);
        sr2.addView(text(L.s("Сила", "Strength"), 14, TXT, false));
        strBar = new SeekBar(this); strBar.setMax(40); strBar.setProgress(Math.round(prefs().getFloat("moss_lora_str", 1f) * 20));
        sr2.addView(strBar, new LinearLayout.LayoutParams(0, -2, 1));
        strText = text("", 14, TXT, true); strText.setMinWidth(dp(44)); strText.setGravity(Gravity.END); sr2.addView(strText);
        strBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar b, int p, boolean u) { strText.setText(String.format(java.util.Locale.US, "%.2f", p * 0.05f)); if (u) prefs().edit().putFloat("moss_lora_str", p * 0.05f).apply(); }
            public void onStartTrackingTouch(SeekBar b) { }
            public void onStopTrackingTouch(SeekBar b) { }
        });
        strText.setText(String.format(java.util.Locale.US, "%.2f", strBar.getProgress() * 0.05f));
        loraBox.addView(sr2);
        root.addView(loraBox);
        refreshLora();

        // sticky bottom bar: progress + status + [pictures count] [Generate / Stop]
        LinearLayout bottom = new LinearLayout(this);
        bottom.setOrientation(LinearLayout.VERTICAL);
        bottom.setBackgroundColor(CARD);
        bottom.setPadding(dp(16), dp(10), dp(16), dp(10));
        bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(100);
        bar.setProgressTintList(ColorStateList.valueOf(ACC));
        bar.setProgressBackgroundTintList(ColorStateList.valueOf(FIELD));
        bar.setVisibility(View.GONE);
        LinearLayout.LayoutParams bl = new LinearLayout.LayoutParams(-1, dp(5)); bl.bottomMargin = dp(6);
        bottom.addView(bar, bl);
        status = text("", 13, DIM, false);
        status.setVisibility(View.GONE);
        status.setPadding(dp(2), 0, 0, dp(6));
        bottom.addView(status);
        LinearLayout brow = new LinearLayout(this);
        count = Math.max(1, Math.min(4, prefs().getInt("count", 1)));
        countBtn = button("×" + count, FIELD, TXT);
        countBtn.setTextSize(16);
        countBtn.setOnClickListener(v -> { count = count % 4 + 1; prefs().edit().putInt("count", count).apply(); countBtn.setText("×" + count); genBtn.setText(genLabel()); });
        LinearLayout.LayoutParams cb = new LinearLayout.LayoutParams(dp(64), dp(54)); cb.rightMargin = dp(8);
        brow.addView(countBtn, cb);
        genBtn = button("", ACC, ON_ACC);
        genBtn.setTextSize(17);
        genBtn.setOnClickListener(v -> { if (busy) cancel(); else generate(false); });
        brow.addView(genBtn, new LinearLayout.LayoutParams(0, dp(54), 1));
        bottom.addView(brow);
        frame.addView(bottom, new LinearLayout.LayoutParams(-1, -2));

        setContentView(frame);
        SharedPreferences p = prefs();   // restore the last session's inputs
        prompt.setText(p.getString("prompt", "")); seed.setText(p.getString("seed", "-1")); model = p.getString("model", "");
        refreshModels();
        setRes(p.getInt("w", 512), p.getInt("h", 768));
        refreshState();
        if (!p.getBoolean("welcomed", false)) startActivity(new Intent(this, Welcome.class));
    }

    /** the tag list for the suggestions is read in the background (8000 lines in assets/tags.txt) */
    void loadTags() {
        new Thread(() -> {
            final ArrayList<String> tags = new ArrayList<>();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(getAssets().open("tags.txt"), "UTF-8"))) { for (String l; (l = r.readLine()) != null; ) if (!l.isEmpty()) tags.add(l); } catch (IOException ignored) { }
            ui.post(() -> prompt.setAdapter(new ArrayAdapter<String>(this, android.R.layout.simple_dropdown_item_1line, tags) {
                @Override public View getView(int i, View v, ViewGroup p) { View r = super.getView(i, v, p); ((TextView) r).setTextColor(TXT); ((TextView) r).setTextSize(15); r.setBackgroundColor(CARD); return r; }
            }));
        }).start();
    }

    @Override protected void onPause() {
        super.onPause();
        visible = false;
        ui.removeCallbacks(ramTick);
        prefs().edit().putString("prompt", prompt.getText().toString()).putString("seed", seed.getText().toString()).putString("model", model).putInt("w", resW).putInt("h", resH)
            .putString("moss_neg", negEdit != null ? negEdit.getText().toString() : prefs().getString("moss_neg", "")).apply();
    }

    @Override protected void onResume() {
        super.onResume();
        visible = true;
        ((NotificationManager) getSystemService(NOTIFICATION_SERVICE)).cancel(2);
        String del = ModelsActivity.deleted; ModelsActivity.deleted = "";
        if (!del.isEmpty() && del.equals(model) && loaded && !busy) { loaded = false; ex.submit(() -> su("sh " + D + "/anima.sh stop < /dev/null >/dev/null 2>&1", false)); }
        if (genBtn != null) { refreshModels(); ui.post(ramTick); }
        if (android.os.Build.VERSION.SDK_INT >= 33 && prefs().getBoolean("welcomed", false) && !prefs().getBoolean("asked_notif", false)) {   // once, after the welcome screen
            prefs().edit().putBoolean("asked_notif", true).apply();
            if (checkSelfPermission("android.permission.POST_NOTIFICATIONS") != android.content.pm.PackageManager.PERMISSION_GRANTED) requestPermissions(new String[]{"android.permission.POST_NOTIFICATIONS"}, 5);
        }
    }

    /** every 2 s: RAM strip, thermal / battery / chip warnings; every 10 s the model state (the idle watchdog may have unloaded it) */
    final Runnable ramTick = new Runnable() {
        @Override public void run() {
            android.app.ActivityManager.MemoryInfo mi = new android.app.ActivityManager.MemoryInfo();
            ((android.app.ActivityManager) getSystemService(ACTIVITY_SERVICE)).getMemoryInfo(mi);
            double g = 1024.0 * 1024 * 1024, tot = mi.totalMem / g, free = mi.availMem / g, used = tot - free;
            ramTxt.setText(L.f("ОЗУ  всего %.1f  ·  занято %.1f  ·  свободно %.1f ГБ", "RAM  total %.1f  ·  used %.1f  ·  free %.1f GB", tot, used, free));
            ramBar.setProgress((int) (1000 * used / tot));
            ramBar.setProgressTintList(ColorStateList.valueOf(free < 0.8 ? BAD : free < 1.5 ? WARN : ACC));
            List<String> w = new ArrayList<>();
            if (!Compat.chipAllowed(Main.this)) w.add(L.f("Чип %s не поддерживается", "Chip %s is not supported", Compat.soc()));
            PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
            if (pm.getCurrentThermalStatus() >= PowerManager.THERMAL_STATUS_MODERATE) w.add(L.s("Телефон горячий: скорость снижена", "The phone is hot: speed is reduced"));
            if (pm.isPowerSaveMode()) w.add(L.s("Экономия заряда замедляет NPU", "Battery saver slows the NPU"));
            Intent bat = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
            if (bat != null) { int lvl = bat.getIntExtra("level", -1) * 100 / Math.max(1, bat.getIntExtra("scale", 100)), st = bat.getIntExtra("status", 0); if (lvl >= 0 && lvl < 15 && st != 2 && st != 5) w.add(L.f("Низкий заряд батареи (%d%%)", "Low battery (%d%%)", lvl)); }
            warnTxt.setVisibility(w.isEmpty() ? View.GONE : View.VISIBLE);
            warnTxt.setText("⚠  " + android.text.TextUtils.join("  ·  ", w));
            if (++tick % 5 == 0 && !busy) refreshState();
            ui.postDelayed(this, 2000);
        }
    };

    @Override protected void onActivityResult(int req, int res, Intent d) {
        if (req == 9 && res == RESULT_OK && d != null && d.getData() != null) { importLora(d.getData()); return; }
        if (req == 7 && res == RESULT_OK && d != null) {
            prompt.setText(d.getStringExtra("prompt")); seed.setText(d.getStringExtra("seed"));
            customOpen = false;
            setRes(d.getIntExtra("w", 512), d.getIntExtra("h", 768));
        } else if (req == 8 && res == RESULT_OK && d != null && d.getData() != null) {
            try (InputStream in = getContentResolver().openInputStream(d.getData()); ByteArrayOutputStream bo = new ByteArrayOutputStream()) {
                byte[] b = new byte[1 << 16]; int total = 0;
                for (int n; (n = in.read(b)) > 0 && total < (40 << 20); ) { bo.write(b, 0, n); total += n; }
                JSONObject m = PngMeta.settings(bo.toByteArray());
                if (m == null || m.optString("prompt").isEmpty()) { Toast.makeText(this, L.s("В этом файле нет настроек генерации", "This file has no generation settings"), Toast.LENGTH_LONG).show(); return; }
                prompt.setText(m.optString("prompt")); seed.setText(m.optString("seed", "-1"));
                customOpen = false; if (m.optInt("w") > 0) setRes(m.optInt("w"), m.optInt("h"));
                if (m.has("mode")) {   // MOSS-Image sampler settings
                    if (m.has("steps")) stepBar.setProgress(m.optInt("steps", Moss.FULL_STEPS_DEF) - Moss.FULL_STEPS_MIN);
                    if (m.has("cfg")) cfgBar.setProgress(Math.round(((float) m.optDouble("cfg", Moss.CFG_DEF) - 1f) * 4));
                    negEdit.setText(m.optString("negative", "")); setMossFull(m.optString("mode").equals("full"));
                    prefs().edit().putInt("moss_steps", mossSteps()).putFloat("moss_cfg", mossCfg()).putString("moss_neg", negEdit.getText().toString()).apply();
                }
                Toast.makeText(this, L.s("Настройки загружены из файла", "Settings loaded from the file"), Toast.LENGTH_SHORT).show();
            } catch (Exception e) { Toast.makeText(this, e.toString(), Toast.LENGTH_LONG).show(); }
        }
    }


    // ---- MOSS-Image sampler mode --------------------------------------------------------------------
    boolean mossFull() { return modeFull != null && prefs().getBoolean("moss_full", false); }
    int mossSteps() { return stepBar == null ? Moss.FULL_STEPS_DEF : Moss.FULL_STEPS_MIN + stepBar.getProgress(); }
    float mossCfg() { return cfgBar == null ? Moss.CFG_DEF : 1f + cfgBar.getProgress() * 0.25f; }
    LinearLayout seekRow(String name, SeekBar sb, TextView val) {
        LinearLayout r = new LinearLayout(this); r.setGravity(Gravity.CENTER_VERTICAL); r.setPadding(0, dp(8), 0, 0);
        TextView n = text(name, 14, TXT, false); n.setMinWidth(dp(56)); r.addView(n);
        r.addView(sb, new LinearLayout.LayoutParams(0, -2, 1));
        val.setMinWidth(dp(44)); val.setGravity(Gravity.END); r.addView(val);
        return r;
    }
    void showFullValues() {
        stepText.setText(String.valueOf(mossSteps())); cfgText.setText(String.format(java.util.Locale.US, "%.2f", mossCfg()));
        if (genBtn != null && !busy) genBtn.setText(genLabel());
    }
    void setMossFull(boolean on) {
        if (busy) return;
        prefs().edit().putBoolean("moss_full", on).apply();
        modeFast.setBackground(bg(on ? CARD : ACC, 14)); modeFast.setTextColor(on ? TXT : ON_ACC);
        modeFull.setBackground(bg(on ? ACC : CARD, 14)); modeFull.setTextColor(on ? ON_ACC : TXT);
        fullBox.setVisibility(on ? View.VISIBLE : View.GONE);
        modeInfo.setText(on ? L.s("Сама модель с официальным сэмплером (Euler, CFG): лучше анатомия и следование описанию, но каждый шаг с CFG > 1 считает модель дважды. Официально: 50 шагов, CFG 3.",
                "The model itself with the official sampler (Euler, CFG): better anatomy and prompt following, but every step with CFG above 1 runs the model twice. Official settings: 50 steps, CFG 3.")
            : L.s("Турбо-адаптер rapid-moss: 4 шага, без CFG и негативного описания (дистиллированная модель).", "The rapid-moss turbo adapter: 4 steps, no CFG or negative prompt (a distilled model)."));
        showFullValues();
    }

    // ---- MOSS-Image user LoRA ---------------------------------------------------------------------
    String loraId() { return prefs().getString("moss_lora_id", ""); }
    boolean loraOn() { return isMoss() && !loraId().isEmpty() && MossLora.src(getFilesDir()).exists() && strBar.getProgress() > 0; }
    void refreshLora() {
        boolean has = !loraId().isEmpty() && MossLora.src(getFilesDir()).exists();
        loraName.setText(has ? prefs().getString("moss_lora_name", "LoRA") : L.s("не выбрана", "none"));
        loraInfo.setText(has ? prefs().getString("moss_lora_info", "") : L.s("Файл .safetensors для MOSS-Image: проверяется сразу после выбора", "A .safetensors file for MOSS-Image: it is checked as soon as you pick it"));
        loraInfo.setTextColor(has ? OK : DIM);
        loraDrop.setVisibility(has ? View.VISIBLE : View.GONE); strBar.setEnabled(has); strBar.setAlpha(has ? 1f : .4f);
    }
    void pickLora() {
        if (busy) { Toast.makeText(this, L.s("Дождитесь окончания генерации", "Wait for the generation to finish"), Toast.LENGTH_SHORT).show(); return; }
        startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("*/*").addCategory(Intent.CATEGORY_OPENABLE), 9);
    }
    void dropLora() {
        if (busy) return;
        prefs().edit().remove("moss_lora_id").remove("moss_lora_name").remove("moss_lora_info").apply();
        ex2.submit(() -> { MossLora.deleteTree(MossLora.dir(getFilesDir())); ui.post(this::refreshLora); });
    }
    void importLora(Uri uri) {
        String nm = "LoRA";
        try (android.database.Cursor c = getContentResolver().query(uri, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (i >= 0) nm = c.getString(i); } } catch (Exception ignored) { }
        final String name = nm;
        loraInfo.setTextColor(WARN); loraInfo.setText(L.s("Проверяю файл…", "Checking the file…"));
        ex2.submit(() -> {
            final File dir = MossLora.dir(getFilesDir()); dir.mkdirs(); final File tmp = new File(dir, "new.safetensors");
            try (InputStream in = getContentResolver().openInputStream(uri); FileOutputStream o = new FileOutputStream(tmp)) {
                byte[] b = new byte[1 << 16]; long total = 0;
                for (int n; (n = in.read(b)) > 0; ) { o.write(b, 0, n); total += n; if (total > (1L << 30)) throw new IOException(L.s("файл больше 1 ГБ", "the file is larger than 1 GB")); }
            } catch (Exception e) { tmp.delete(); ui.post(() -> { refreshLora(); loraFail(name, L.f("Не удалось прочитать файл: %s", "Cannot read the file: %s", e.getMessage())); }); return; }
            MossLora.Check c = MossLora.check(BIN, tmp);
            if (!c.ok) { tmp.delete(); ui.post(() -> { refreshLora(); loraFail(name, c.message()); }); return; }
            File[] old = dir.listFiles(); if (old != null) for (File f : old) if (!f.equals(tmp)) MossLora.deleteTree(f);   // the old LoRA and its folded patches go
            tmp.renameTo(MossLora.src(getFilesDir()));
            prefs().edit().putString("moss_lora_id", Long.toHexString(System.currentTimeMillis())).putString("moss_lora_name", name).putString("moss_lora_info", "✓ " + c.message()).apply();
            ui.post(this::refreshLora);
        });
    }
    void loraFail(String name, String why) {
        builder().setTitle(L.f("LoRA «%s» не подходит", "LoRA \"%s\" does not fit", name)).setMessage(why).setPositiveButton("OK", null).show();
    }

    // ---- menu, dialogs ------------------------------------------------------------------------
    void menu(View anchor) {
        PopupMenu pm = new PopupMenu(this, anchor);
        pm.getMenu().add(0, 1, 0, L.s("Модели", "Models"));
        pm.getMenu().add(0, 2, 1, L.s("Открыть настройки из PNG", "Load settings from a PNG"));
        pm.getMenu().add(0, 3, 2, L.s("Настройки", "Settings"));
        pm.getMenu().add(0, 4, 3, L.s("Инфо", "Info"));
        pm.setOnMenuItemClickListener(i -> {
            if (i.getItemId() == 1) openModels();
            else if (i.getItemId() == 2) startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("image/png").addCategory(Intent.CATEGORY_OPENABLE), 8);
            else if (i.getItemId() == 3) startActivity(new Intent(this, Settings.class));
            else startActivity(new Intent(this, Info.class));
            return true;
        });
        pm.show();
    }

    void openModels() { if (busy) { Toast.makeText(this, L.s("Дождитесь окончания генерации", "Wait for the generation to finish"), Toast.LENGTH_SHORT).show(); return; } startActivity(new Intent(this, ModelsActivity.class)); }

    /** saved and recent prompts: tap to use; saved ones can be removed with ✕ */
    void templates() {
        final LinearLayout box = new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL); box.setPadding(dp(14), dp(14), dp(14), dp(10));
        final AlertDialog[] dlg = new AlertDialog[1];
        final Runnable[] fill = new Runnable[1];
        fill[0] = () -> {
            box.removeAllViews();
            Button save = button(L.s("Сохранить текущее описание", "Save the current prompt"), ACC, ON_ACC); save.setTextSize(14);
            save.setOnClickListener(v -> {
                String t = prompt.getText().toString().trim();
                if (t.isEmpty()) { Toast.makeText(this, L.s("Введите описание", "Enter a prompt"), Toast.LENGTH_SHORT).show(); return; }
                List<String> l = savedPrompts(); l.remove(t); l.add(0, t); while (l.size() > 30) l.remove(l.size() - 1);
                prefs().edit().putString("saved", new JSONArray(l).toString()).apply(); fill[0].run();
            });
            box.addView(save, new LinearLayout.LayoutParams(-1, dp(44)));
            List<String> sv = savedPrompts();
            if (!sv.isEmpty()) { box.addView(sect(L.s("Сохранённые", "Saved"))); for (String t : sv) box.addView(promptRow(t, true, dlg, fill)); }
            List<String> rc = recentPrompts();
            if (!rc.isEmpty()) { box.addView(sect(L.s("Недавние", "Recent"))); for (String t : rc) box.addView(promptRow(t, false, dlg, fill)); }
        };
        fill[0].run();
        ScrollView s = new ScrollView(this); s.addView(box);
        dlg[0] = dialog(s);
    }

    TextView sect(String s) { TextView t = text(s, 12, DIM, true); t.setPadding(dp(4), dp(14), 0, dp(6)); return t; }

    View promptRow(String t, boolean removable, AlertDialog[] dlg, Runnable[] fill) {
        LinearLayout r = new LinearLayout(this); r.setGravity(Gravity.CENTER_VERTICAL); r.setBackground(bg(FIELD, 12)); r.setPadding(dp(12), dp(10), dp(8), dp(10));
        TextView tv = text(t, 13, TXT, false); tv.setMaxLines(2); tv.setEllipsize(android.text.TextUtils.TruncateAt.END);
        r.addView(tv, new LinearLayout.LayoutParams(0, -2, 1));
        if (removable) {
            TextView x = text("✕", 16, DIM, false); x.setPadding(dp(12), dp(4), dp(8), dp(4));
            x.setOnClickListener(v -> { List<String> l = savedPrompts(); l.remove(t); prefs().edit().putString("saved", new JSONArray(l).toString()).apply(); fill[0].run(); });
            r.addView(x);
        }
        r.setOnClickListener(v -> { prompt.setText(t); prompt.setSelection(t.length()); dlg[0].dismiss(); });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.topMargin = dp(4);
        r.setLayoutParams(lp);
        return r;
    }

    List<String> savedPrompts() {
        List<String> l = new ArrayList<>();
        try { JSONArray a = new JSONArray(prefs().getString("saved", "[]")); for (int i = 0; i < a.length(); i++) l.add(a.getString(i)); } catch (Exception ignored) { }
        return l;
    }

    List<String> recentPrompts() {
        List<String> l = new ArrayList<>();
        File[] fs = Gallery.list(this, Gallery.ALL);
        for (int i = 0; i < fs.length && l.size() < 10 && i < 60; i++) { String p = Gallery.meta(fs[i]).optString("prompt").trim(); if (!p.isEmpty() && !l.contains(p)) l.add(p); }
        return l;
    }

    static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
    static String sizeLabel(int w, int h) { int g = gcd(w, h); return w + "×" + h + "  ·  " + w / g + ":" + h / g; }

    /** Size picker: the portrait sizes in the left column, their landscape twins in the right one, squares below, then "Custom". */
    void pickSize() {
        final AlertDialog[] dlg = new AlertDialog[1];
        LinearLayout box = new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL); box.setPadding(dp(14), dp(14), dp(14), dp(10));
        java.util.List<int[]> por = new java.util.ArrayList<>(), lan = new java.util.ArrayList<>(), sq = new java.util.ArrayList<>();
        for (int[] z : sizeList()) (z[1] > z[0] ? por : z[0] > z[1] ? lan : sq).add(z);
        box.addView(cols(text(L.s("Портрет", "Portrait"), 12, DIM, true), text(L.s("Альбом", "Landscape"), 12, DIM, true), true));
        for (int i = 0; i < Math.max(por.size(), lan.size()); i++) box.addView(cols(i < por.size() ? cell(por.get(i), dlg) : new View(this), i < lan.size() ? cell(lan.get(i), dlg) : new View(this), false));
        TextView sqh = text(L.s("Квадрат", "Square"), 12, DIM, true); sqh.setPadding(dp(4), dp(14), 0, dp(6)); box.addView(sqh);
        for (int i = 0; i < sq.size(); i += 2) box.addView(cols(cell(sq.get(i), dlg), i + 1 < sq.size() ? cell(sq.get(i + 1), dlg) : new View(this), false));
        Button custom = button(L.s("Свой", "Custom"), FIELD, customOpen ? ACC : TXT);
        custom.setOnClickListener(v -> { customOpen = true; setRes(resW, resH); dlg[0].dismiss(); });
        LinearLayout.LayoutParams cl = new LinearLayout.LayoutParams(-1, dp(44)); cl.topMargin = dp(12);
        box.addView(custom, cl);
        ScrollView sv = new ScrollView(this); sv.addView(box);
        dlg[0] = dialog(sv);
    }

    LinearLayout cols(View l, View r, boolean header) {
        LinearLayout row = new LinearLayout(this); row.setPadding(0, dp(3), 0, dp(3));
        LinearLayout.LayoutParams a = new LinearLayout.LayoutParams(0, -2, 1); a.rightMargin = dp(4);
        LinearLayout.LayoutParams b = new LinearLayout.LayoutParams(0, -2, 1); b.leftMargin = dp(4);
        if (header) { l.setPadding(dp(4), 0, 0, dp(2)); r.setPadding(dp(4), 0, 0, dp(2)); }
        row.addView(l, a); row.addView(r, b);
        return row;
    }

    TextView cell(int[] z, AlertDialog[] dlg) {
        boolean on = !customOpen && z[0] == resW && z[1] == resH;
        TextView t = text(sizeLabel(z[0], z[1]), 14, on ? ON_ACC : TXT, on);
        t.setGravity(Gravity.CENTER); t.setPadding(dp(6), dp(12), dp(6), dp(12)); t.setBackground(bg(on ? ACC : FIELD, 14));
        t.setOnClickListener(v -> { customOpen = false; setRes(z[0], z[1]); dlg[0].dismiss(); });
        return t;
    }

    boolean isMoss() { return Moss.is(model); }
    /** the size presets of the selected model: Anima's list, or MOSS-Image's own presets (any size with sides multiple of 16 works) */
    java.util.List<int[]> sizeList() {
        if (isMoss()) return java.util.Arrays.asList(Moss.PRESETS);
        java.util.List<int[]> l = new java.util.ArrayList<>(); int max = Compat.maxPatches(this);   // chips other than V69: only the verified sizes unless "large sizes" is on (Settings)
        for (int[] z : SIZES) if ((z[0] / 16) * (z[1] / 16) <= max) l.add(z);
        return l;
    }
    String mossSizeError(int w, int h) { return Moss.sizeError(w, h); }

    // ---- model state --------------------------------------------------------------------------
    String modelName() { return model.isEmpty() ? L.s("нет модели", "no model") : store.name(model); }

    void pickModel() {
        if (busy) { Toast.makeText(this, L.s("Дождитесь окончания генерации", "Wait for the generation to finish"), Toast.LENGTH_SHORT).show(); return; }
        if (ids.isEmpty()) { openModels(); return; }
        String[] names = new String[ids.size() + 1];
        for (int i = 0; i < ids.size(); i++) names[i] = (ids.get(i).equals(model) ? "✓  " : "      ") + store.name(ids.get(i));
        names[ids.size()] = "＋  " + L.s("Каталог моделей…", "Model catalog…");
        builder().setTitle(L.s("Модель", "Model")).setItems(names, (d, i) -> { if (i < ids.size()) setModel(ids.get(i)); else openModels(); }).show();
    }

    /** The NPU pipeline needs width/height multiples of 16, patch count divisible by 32 and at most 6144 patches (megarun TMAX; 4096 on unverified chips unless enabled in Settings). */
    String sizeError(int w, int h) {
        if (w < 256 || h < 256) return L.s("Размер: минимум 256 по каждой стороне", "Size: at least 256 on each side");
        if (w % 16 != 0 || h % 16 != 0) return L.s("Ширина и высота должны быть кратны 16", "Width and height must be multiples of 16");
        int t = (w / 16) * (h / 16);
        if (t % 32 != 0) return L.f("Число патчей (%d) должно делиться на 32, например 640×640, 1024×1024, 704×832", "Patch count (%d) must be divisible by 32, e.g. 640×640, 1024×1024, 704×832", t);
        if (t > 6144) return L.s("Слишком большой размер: максимум 6144 патча (например 1536×960)", "Too large: at most 6144 patches (e.g. 1536×960)");
        if (t > Compat.maxPatches(this)) return L.f("На этом чипе проверено до %d патчей (например 1024×1024); большие размеры включаются в настройках", "On this chip sizes are verified up to %d patches (e.g. 1024×1024); larger sizes can be enabled in Settings", Compat.maxPatches(this));
        return null;
    }

    void setRes(int w, int h) {
        resW = w; resH = h;
        boolean preset = false;
        for (int[] z : sizeList()) if (z[0] == w && z[1] == h) preset = true;
        if (sizeText != null) sizeText.setText(sizeLabel(w, h) + (preset && !customOpen ? "" : "  ·  " + L.s("Свой", "Custom")));
        if (customRow != null) { customRow.setVisibility(customOpen || !preset ? View.VISIBLE : View.GONE); if (!wEdit.getText().toString().equals(String.valueOf(w)) || !hEdit.getText().toString().equals(String.valueOf(h))) { wEdit.setText(String.valueOf(w)); hEdit.setText(String.valueOf(h)); } }
        if (chipName != null) chipName.setText(modelName());
        if (genBtn != null && !busy) genBtn.setText(genLabel());
    }

    /** rough time to a picture: the catalog speed at 512x768 scaled by the patch count, times the number of pictures */
    String genLabel() {
        if (model.isEmpty()) return L.s("Скачать модель", "Get a model");
        long est = Math.max(3, Moss.is(model) ? Moss.estimate(store.speed(model), resW, resH, mossFull(), mossSteps(), mossCfg()) : Math.round(store.speed(model) * ((resW / 16) * (resH / 16)) / 1536.0)) * count;
        String t = est >= 120 ? L.f("~%d мин", "~%d min", Math.round(est / 60.0)) : L.f("~%d с", "~%d s", est);
        return count > 1 ? L.f("Генерировать ×%d  ·  %s", "Generate ×%d  ·  %s", count, t) : L.f("Генерировать  ·  %s", "Generate  ·  %s", t);
    }

    /** Share of the DiT time spent after `st` steps: the first two of the 4 steps use a half-size latent and cost ~0.3 (512x768) / ~0.22 (960x1536) of a full step. */
    double stepShare(int st) {
        double lo = resW * resH < 500000 ? 0.3 : 0.22;
        double done = 0, total = 0;
        for (int i = 0; i < STEPS; i++) { double c = i < 2 ? lo : 1.0; total += c; if (i < st) done += c; }
        return done / total;
    }

    void setModelUi(boolean isLoaded, boolean working, String msg) {
        ui.post(() -> {
            int c = working ? WARN : isLoaded ? OK : DIM;
            chipDot.setTextColor(c);
            if (loraBox != null) loraBox.setVisibility(isMoss() ? View.VISIBLE : View.GONE);
            if (mossBox != null) { mossBox.setVisibility(isMoss() ? View.VISIBLE : View.GONE); modeFast.setEnabled(!busy); modeFull.setEnabled(!busy); }
            state.setText(working ? msg : isMoss() ? L.s("MOSS-Image: текст, DiT и VAE работают на NPU; веса читаются с накопителя при каждой картинке", "MOSS-Image: text encoder, DiT and VAE run on the NPU; the weights are streamed from storage for every picture") : isLoaded ? L.s("Модель загружена в NPU, генерация стартует сразу", "Model is loaded on the NPU, generation starts at once")
                : L.s("Модель выгружена: при генерации загрузится сама (~5 с)", "Model is unloaded: it loads itself when you generate (~5 s)"));
            loadBtn.setText(isLoaded ? L.s("Выгрузить", "Unload") : L.s("Загрузить", "Load"));
            loadBtn.setEnabled(!working && !busy && !model.isEmpty() && !isMoss());
            loadBtn.setAlpha(loadBtn.isEnabled() ? 1f : .5f);
            countBtn.setEnabled(!busy); countBtn.setAlpha(busy ? .5f : 1f);
            boolean can = busy ? !cancelled : !working;   // while a picture is made the main button is "Stop"
            genBtn.setEnabled(can);
            genBtn.setBackground(busy ? bg(BAD, 18) : gradient(can));
            genBtn.setTextColor(busy ? 0xFFFFFFFF : can ? ON_ACC : DIM);
            genBtn.setText(busy ? (cancelled ? L.s("Останавливаю…", "Stopping…") : L.s("Остановить", "Stop")) : genLabel());
        });
    }

    void refreshState() {
        if (isMoss()) { setModelUi(false, false, ""); return; }
        ex.submit(() -> {
            String s = su("sh " + D + "/anima.sh status", true);
            if (s.startsWith("ERR") || s.isEmpty()) {
                ui.post(() -> { state.setText(L.f("Не запускается оболочка: %s", "Shell does not start: %s", s)); chipDot.setTextColor(BAD); });
                return;
            }
            if (busy) return;
            loaded = s.contains("unloaded") ? false : s.contains("loaded");
            setModelUi(loaded, false, "");
            if (!statusSeen) { statusSeen = true; if (!loaded && Settings.autoload(this) && !model.isEmpty() && Compat.chipAllowed(this)) ui.post(this::toggleModel); }   // "load at start"
        });
    }

    void setModel(String m) {
        if (m.equals(model) || busy) return;   // no model switch mid-generation
        model = m;
        setRes(resW, resH);
        if (loaded) {   // the loaded server holds the other model's weights
            setModelUi(true, true, L.s("Меняю модель…", "Switching model…"));
            ex.submit(() -> { su("sh " + D + "/anima.sh stop < /dev/null >/dev/null 2>&1", false); loaded = false; setModelUi(false, false, ""); });
        } else setModelUi(false, false, "");
    }

    void toggleModel() {
        if (isMoss()) return;   // nothing to preload
        final boolean want = !loaded;
        setModelUi(loaded, true, want ? L.s("Загружаю модель…", "Loading the model…") : L.s("Выгружаю модель…", "Unloading the model…"));
        ex.submit(() -> {
            su("rm -f " + D + "/cancel; " + mp() + "sh " + D + "/anima.sh " + (want ? "load" : "stop") + " < /dev/null >/dev/null 2>&1", false);
            loaded = want;
            setModelUi(loaded, false, "");
        });
    }

    /** anima.sh ships in assets and is rewritten at every start (it is a script, so it may live in the app's data dir) */
    void installScript() {
        for (String nm : new String[]{"anima.sh", "moss.sh"})
            try (InputStream in = getAssets().open(nm); FileOutputStream o = new FileOutputStream(new File(D, nm))) {
                byte[] b = new byte[8192]; for (int n; (n = in.read(b)) > 0; ) o.write(b, 0, n);
            } catch (IOException e) { Toast.makeText(this, nm + ": " + e, Toast.LENGTH_LONG).show(); }
    }

    void refreshModels() {
        ids = store.installed();
        model = ids.isEmpty() ? "" : ids.contains(model) ? model : ids.get(0);
        setRes(resW, resH);
        setModelUi(loaded, false, "");
    }

    // ---- shell --------------------------------------------------------------------------------
    /** One long-lived /system/bin/sh with A, S, BIN in its environment: the helper binaries are started from it with the app's own uid. */
    static class Sh {
        String[] env; Process p; BufferedWriter w; BufferedReader r;
        synchronized String run(String cmd) {
            try {
                if (p == null) {
                    ProcessBuilder pb = new ProcessBuilder("/system/bin/sh").redirectErrorStream(true);
                    pb.environment().put("A", env[0]); pb.environment().put("S", env[1]); pb.environment().put("BIN", env[2]); pb.directory(new File(env[0]));
                    p = pb.start();
                    w = new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));
                    r = new BufferedReader(new InputStreamReader(p.getInputStream()));
                }
                w.write(cmd + "\necho __END__\n");
                w.flush();
                StringBuilder sb = new StringBuilder();
                for (String l; (l = r.readLine()) != null && !l.equals("__END__"); ) sb.append(l).append('\n');
                return sb.toString().trim();
            } catch (Exception e) {
                p = null;
                return "ERR " + e;
            }
        }
    }

    /** main: model load / generation; pollSh: progress and status polling; ctlSh: the Stop button (it must not wait for a running generation) */
    final Sh main = new Sh(), pollSh = new Sh(), ctlSh = new Sh();

    /** Runs a command in the main shell (blocking) and returns its stdout; `poll` uses the second shell so progress polling never waits for a running generation. */
    String su(String cmd, boolean poll) { return (poll ? pollSh : main).run(cmd); }

    static String q(String s) { return "'" + s.replace("'", "'\\''") + "'"; }

    // ---- generation ---------------------------------------------------------------------------
    /** Stop: the server is stopped (the model unloads; the next picture reloads it in ~5 s), anima.sh sees the cancel marker and leaves its waiting loops */
    void cancel() {
        if (!busy || cancelled) return;
        cancelled = true;
        setModelUi(loaded, false, "");
        if (isMoss()) ex2.submit(() -> ctlSh.run("sh " + D + "/moss.sh stop < /dev/null >/dev/null 2>&1"));
        else ex2.submit(() -> ctlSh.run("touch " + D + "/cancel; sh " + D + "/anima.sh stop < /dev/null >/dev/null 2>&1"));
    }

    void generate(boolean newSeed) {
        if (model.isEmpty()) { openModels(); return; }
        int cw = 0, ch = 0;
        try { cw = Integer.parseInt(wEdit.getText().toString().trim()); ch = Integer.parseInt(hEdit.getText().toString().trim()); } catch (NumberFormatException e) { }
        String bad = isMoss() ? mossSizeError(cw, ch) : sizeError(cw, ch);
        if (bad != null) { Toast.makeText(this, bad, Toast.LENGTH_LONG).show(); return; }
        if (cw != resW || ch != resH) setRes(cw, ch);
        final String pr = prompt.getText().toString().trim();
        if (pr.isEmpty()) { Toast.makeText(this, L.s("Введите описание", "Enter a prompt"), Toast.LENGTH_SHORT).show(); return; }
        String sd = seed.getText().toString().trim();
        final boolean randomSeed = newSeed || sd.isEmpty() || sd.equals("-1");   // -1 = random; the seed used is shown under the picture
        long base0 = 0; try { base0 = Long.parseLong(sd); } catch (NumberFormatException ignored) { }
        final long baseSeed = base0;
        final int n = count, w = resW, h = resH;
        busy = true; cancelled = false; curIdx = 0; curN = n;
        setModelUi(loaded, false, "");
        View f0 = getCurrentFocus(); if (f0 != null) ((android.view.inputmethod.InputMethodManager) getSystemService(INPUT_METHOD_SERVICE)).hideSoftInputFromWindow(f0.getWindowToken(), 0);
        bar.setProgress(0);
        bar.setVisibility(View.VISIBLE);
        status.setVisibility(View.VISIBLE);
        status.setText(L.s("Запуск…", "Starting…"));
        if (Settings.keepOn(this)) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (((PowerManager) getSystemService(POWER_SERVICE)).getCurrentThermalStatus() >= PowerManager.THERMAL_STATUS_SEVERE) Toast.makeText(this, L.s("Телефон сильно нагрелся: генерация будет медленной", "The phone is very hot: generation will be slow"), Toast.LENGTH_LONG).show();
        final android.content.Context app = getApplicationContext();
        app.startForegroundService(new Intent(app, DownloadService.class).putExtra("mode", "gen"));   // keeps the process alive when the app goes to the background
        final long t0 = System.currentTimeMillis();
        ex.submit(() -> {
            int made = 0; Bitmap lastBm = null;
            for (int i = 0; i < n && !cancelled; i++) {
                curIdx = i;
                final String seedS = String.valueOf(randomSeed ? new Random().nextInt(1 << 30) : baseSeed + i);
                final File f = runOne(pr, seedS, w, h, t0);
                if (f == null) break;
                made++;
                final Bitmap bm = BitmapFactory.decodeFile(f.getPath());
                lastBm = bm;
                ui.post(() -> {
                    lastFile = f; img.setImageBitmap(bm); img.setVisibility(View.VISIBLE); placeholder.setVisibility(View.GONE);
                    setFavIcon(false);
                    for (ImageButton x : new ImageButton[]{againBtn, favBtn, dlBtn, shareBtn}) x.setVisibility(View.VISIBLE);
                    final ScrollView svv = (ScrollView) imgBox.getParent().getParent(); svv.post(() -> svv.smoothScrollTo(0, 0));   // after the image has been laid out
                    saved.setText("seed " + seedS);
                });
            }
            final double sec = (System.currentTimeMillis() - t0) / 1000.0;
            final boolean wasCancelled = cancelled, okAny = made > 0;
            final int total = made;
            if (wasCancelled) loaded = false;   // Stop shut the server down
            busy = false; cancelled = false;
            app.stopService(new Intent(app, DownloadService.class));
            final Bitmap nb = lastBm;
            ui.post(() -> {
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                bar.setVisibility(View.GONE);
                if (wasCancelled) status.setText(L.s("Остановлено", "Stopped"));
                else if (!okAny) status.setText(L.s("Ошибка генерации (лог: files/anima/animad.log)", "Generation failed (log: files/anima/animad.log)"));
                else status.setText(L.f("Готово за %.1f с", "Done in %.1f s", sec));
                setModelUi(loaded, false, "");
                if (okAny && !visible) notifyDone(app, nb, pr, total);
            });
        });
    }

    /** one picture: loads the model if needed, runs anima.sh, stores the PNG (with the settings inside) in the history; null on failure or Stop */
    File runOne(String pr, String seedS, int w, int h, long t0) {
        if (isMoss()) return runOneMoss(pr, seedS, w, h, t0);
        if (!loaded) {
            setProgressUi(0, L.s("Загружаю модель…", "Loading the model…"), t0);
            su("rm -f " + D + "/cancel; " + mp() + "sh " + D + "/anima.sh load < /dev/null >/dev/null 2>&1", false);
            if (cancelled) return null;
            loaded = true;
        }
        final String f = D + "/last.png";
        final boolean[] done = {false};
        final int[] base = count();
        Thread poll = new Thread(() -> {
            while (!done[0]) {
                try { Thread.sleep(700); } catch (InterruptedException e) { return; }
                int[] c = count();
                int st = c[0] - base[0];
                if (c[1] > base[1]) setProgressUi(97, L.s("Сохраняю…", "Saving…"), t0);
                else if (st >= STEPS) setProgressUi(92, L.s("Декодирование…", "Decoding…"), t0);
                else if (c[2] > base[2]) setProgressUi(15 + (int) (76 * stepShare(st)), L.f("Шаг %d/%d", "Step %d/%d", st + 1, STEPS), t0);  // adapter line = step 1 started
                else setProgressUi(8, L.s("Кодирование текста…", "Encoding the text…"), t0);
            }
        });
        poll.start();
        su("cd " + D + " && rm -f last.png && " + mp() + "PREFIX=" + q(Settings.prefix(this)) + " STEPS=" + STEPS + " W=" + w + " H=" + h + " LOGRE='learned_condition|completed in' sh ./anima.sh " + q(pr) + " " + seedS + " " + f + " < /dev/null >> gen.log 2>&1", false);
        done[0] = true;
        try { poll.join(); } catch (InterruptedException ignored) { }
        if (cancelled) return null;
        byte[] png = new byte[0];
        try { png = Models.readBytes(new File(f)); } catch (Exception ignored) { }
        if (png.length == 0 || BitmapFactory.decodeByteArray(png, 0, png.length) == null) return null;
        try {
            JSONObject meta = new JSONObject().put("prompt", pr).put("seed", seedS).put("w", w).put("h", h).put("model", modelName()).put("id", model).put("sec", (System.currentTimeMillis() - t0) / 1000.0);
            return Gallery.save(this, PngMeta.embed(png, meta), meta);
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, e.toString(), Toast.LENGTH_LONG).show()); return null; }
    }

    /** one MOSS-Image picture: tokenise here, text encoder + DiT + VAE on the NPU through moss.sh, PNG with the settings inside */
    File runOneMoss(String pr, String seedS, int w, int h, long t0) {
        final File home = getFilesDir(), M = Moss.dir(home), ids = new File(M, "ids.i32"), raw = new File(M, "img.f32");
        setProgressUi(3, L.s("Токенизация…", "Tokenising…"), t0);
        final String ve = Moss.vaeError(home);
        if (ve != null) { ui.post(() -> Toast.makeText(this, ve, Toast.LENGTH_LONG).show()); return null; }
        try { Moss.writeIds(home, pr, ids); } catch (Exception e) { ui.post(() -> Toast.makeText(this, "tokenizer: " + e, Toast.LENGTH_LONG).show()); return null; }
        final boolean full = mossFull(); final int steps = full ? mossSteps() : STEPS; final float cfg = mossCfg(); final String neg = negEdit.getText().toString().trim();
        String fullEnv = "";
        if (full) {
            prefs().edit().putString("moss_neg", neg).apply();
            final String fe = Moss.fullError(home);
            if (fe != null) { ui.post(() -> Toast.makeText(this, fe, Toast.LENGTH_LONG).show()); return null; }
            final File nf = new File(M, "neg.i32");
            try { Moss.writeIds(home, neg, nf); } catch (Exception e) { ui.post(() -> Toast.makeText(this, "tokenizer: " + e, Toast.LENGTH_LONG).show()); return null; }
            fullEnv = "MOSS_FULL=" + q("full " + String.format(java.util.Locale.US, "%.2f", cfg) + " " + nf.getPath()) + " ";
        }
        String loraEnv = ""; final boolean useLora = loraOn(); final float lstr = strBar.getProgress() * 0.05f;
        if (useLora) {
            final String[] msg = {""};
            File ovl = MossLora.ensure(home, BIN, loraId(), lstr, (d, t) -> setProgressUi(1 + 5 * d / t, L.f("Подготовка LoRA… %d/%d", "Preparing the LoRA… %d/%d", d, t), t0), () -> cancelled, msg);
            if (ovl == null) { if (!cancelled) ui.post(() -> Toast.makeText(this, L.f("Не удалось подготовить LoRA: %s", "Could not prepare the LoRA: %s", msg[0]), Toast.LENGTH_LONG).show()); return null; }
            loraEnv = "MOSS_LORA=" + q(ovl.getPath()) + " ";
        }
        final boolean[] done = {false};
        Thread poll = new Thread(() -> {
            while (!done[0]) {
                try { Thread.sleep(700); } catch (InterruptedException e) { return; }
                String s = su("echo $(grep -c '^te:' " + M + "/moss.log) $(grep -c '^refiner:' " + M + "/moss.log) $(grep -c '^step' " + M + "/moss.log) $(grep -c '^done' " + M + "/moss.log) $(sed -n 's/^schedule //p' " + M + "/moss.log)", true);
                try {
                    String[] a = s.trim().split("\\s+"); int te = Integer.parseInt(a[0]), rf = Integer.parseInt(a[1]), st = Integer.parseInt(a[2]), dn = Integer.parseInt(a[3]);
                    final int all = a.length > 4 ? Integer.parseInt(a[4]) : steps;   // full mode: the schedule mossrun settled on
                    if (dn > 0) setProgressUi(94, L.s("Декодирование…", "Decoding…"), t0);
                    else if (rf > 0) setProgressUi(22 + 70 * st / all, L.f("Шаг %d/%d", "Step %d/%d", Math.min(st + 1, all), all), t0);
                    else if (te > 0) setProgressUi(20, L.s("Подготовка текста…", "Preparing the text…"), t0);
                    else setProgressUi(8, L.s("Кодирование текста на NPU…", "Encoding the text on the NPU…"), t0);
                } catch (Exception ignored) { }
            }
        });
        poll.start();
        su("rm -f " + q(raw.getPath()) + "; " + loraEnv + fullEnv + Moss.env(this) + "M=" + q(M.getPath()) + " sh " + D + "/moss.sh " + q(ids.getPath()) + " " + w + " " + h + " " + seedS + " " + steps + " " + q(raw.getPath()) + " < /dev/null >> " + D + "/gen.log 2>&1", false);
        done[0] = true;
        try { poll.join(); } catch (InterruptedException ignored) { }
        if (cancelled) return null;
        try {
            byte[] png = Moss.png(raw, w, h); raw.delete();
            JSONObject meta = new JSONObject().put("prompt", pr).put("seed", seedS).put("w", w).put("h", h).put("model", modelName()).put("id", model).put("sec", (System.currentTimeMillis() - t0) / 1000.0);
            if (useLora) meta.put("lora", prefs().getString("moss_lora_name", "LoRA")).put("lora_strength", lstr);
            meta.put("mode", full ? "full" : "fast"); if (full) meta.put("steps", steps).put("cfg", cfg).put("negative", neg);
            return Gallery.save(this, PngMeta.embed(png, meta), meta);
        } catch (Exception e) { ui.post(() -> Toast.makeText(this, L.f("Ошибка генерации: %s (лог files/moss/moss.log)", "Generation failed: %s (log files/moss/moss.log)", e.getMessage()), Toast.LENGTH_LONG).show()); return null; }
    }

    /** "Done" notification when the app is not on screen; tapping it opens the app */
    void notifyDone(android.content.Context app, Bitmap bm, String pr, int made) {
        try {
            NotificationManager nm = (NotificationManager) app.getSystemService(NOTIFICATION_SERVICE);
            nm.createNotificationChannel(new NotificationChannel("done", L.s("Готовые картинки", "Finished pictures"), NotificationManager.IMPORTANCE_DEFAULT));
            Notification.Builder nb = new Notification.Builder(app, "done").setSmallIcon(android.R.drawable.ic_menu_gallery).setAutoCancel(true)
                .setContentTitle(made > 1 ? L.f("Готово: %d картинки", "Done: %d pictures", made) : L.s("Картинка готова", "Picture is ready")).setContentText(pr)
                .setContentIntent(PendingIntent.getActivity(app, 0, new Intent(app, Main.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP), PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT));
            if (bm != null) nb.setLargeIcon(bm).setStyle(new Notification.BigPictureStyle().bigPicture(bm));
            nm.notify(2, nb.build());
        } catch (Exception ignored) { }
    }

    /** [steps done, vae runs, adapter runs (text encoder finished)] from the animad log. */
    int[] count() {
        String s = su("echo $(grep -c 'animad: step' " + D + "/animad.log) $(grep -c 'animad: vae' " + D + "/animad.log) $(grep -c 'animad: adapter' " + D + "/animad.log)", true);
        try {
            String[] a = s.split("\\s+");
            return new int[]{Integer.parseInt(a[0]), Integer.parseInt(a[1]), Integer.parseInt(a[2])};
        } catch (Exception e) {
            return new int[]{0, 0, 0};
        }
    }

    /** pct is the progress of the current picture; with several pictures the bar and the text cover the whole batch */
    void setProgressUi(int pct, String msg, long t0) {
        final int all = (curIdx * 100 + pct) / curN;
        final String pre = curN > 1 ? "(" + (curIdx + 1) + "/" + curN + ")  " : "";
        final String t = L.f("%s  ·  %.0f с", "%s  ·  %.0f s", pre + msg, (System.currentTimeMillis() - t0) / 1000.0);
        ui.post(() -> { bar.setProgress(all); status.setText(t); });
    }
}
