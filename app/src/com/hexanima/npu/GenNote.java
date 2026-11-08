package com.hexanima.npu;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;

/** The generation notification (the one the foreground service shows): the step ("Step 5/20") and the time left, kept current while pictures are made.
 *  Android 16+: a Live Update (ProgressStyle + "request promoted ongoing" + short critical text "5/20 · 4m"). HyperOS 3+ shows a promoted Live Update
 *  in the Super Island (small icon + the short text) for any app. Xiaomi's own focus template (extras miui.focus.param) is not used: SystemUI accepts
 *  it only from apps whose signature Xiaomi registered (measured on HyperOS OS4.0: canShowFocus = true, yet the template fails with
 *  "Authentication failed because scope mismatch", and every update triggers that auth request again).
 *  The time left also counts down in the notification itself (chronometer). Older Android: an ordinary progress notification. */
final class GenNote {
    static final int ID = 1;   // the foreground service's notification id: notify(ID) replaces it in place
    static volatile boolean live;   // the generation service runs: updates after it stopped would leave an ongoing notification behind
    private static int shownStep = -2; private static long shownEnd, shownAt; private static String shownTitle = "", shownLeft = "";
    private static boolean promoLogged;

    static void channel(Context c) {
        c.getSystemService(NotificationManager.class).createNotificationChannel(new NotificationChannel("gen", L.s("Генерация", "Generation"), NotificationManager.IMPORTANCE_LOW));
    }

    /** title e.g. "Step 5/20"; pct 0..100 (-1 = indeterminate); step/total: the step shown (1-based) or -1 outside the sampling loop; endMs: when the batch should be done (0 = unknown) */
    static Notification build(Context c, String title, int pct, int step, int total, long endMs) {
        final long now = System.currentTimeMillis();
        final boolean eta = endMs > now + 1000;
        final String left = eta ? left(endMs - now) : L.s("Генерирую картинку…", "Generating a picture…");
        PendingIntent open = PendingIntent.getActivity(c, 1, new Intent(c, Main.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP), PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        int si = c.getResources().getIdentifier("ic_stat", "drawable", c.getPackageName());
        Notification.Builder b = new Notification.Builder(c, "gen").setSmallIcon(si != 0 ? si : android.R.drawable.ic_menu_gallery).setContentTitle(title).setContentText(left)
            .setOngoing(true).setOnlyAlertOnce(true).setCategory(Notification.CATEGORY_PROGRESS).setContentIntent(open);
        if (eta) b.setWhen(endMs).setShowWhen(true).setUsesChronometer(true).setChronometerCountDown(true);   // SystemUI counts down to the estimated end
        if (Build.VERSION.SDK_INT >= 31) b.setForegroundServiceBehavior(Notification.FOREGROUND_SERVICE_IMMEDIATE);
        if (Build.VERSION.SDK_INT >= 36) {   // Live Update: expanded at the top of the shade, a status bar chip / the HyperOS Super Island
            b.setStyle(new Notification.ProgressStyle().setProgress(Math.max(pct, 0)).setProgressIndeterminate(pct < 0));
            String chip = step > 0 ? step + "/" + total + (eta ? " · " + shortLeft(endMs - now) : "") : eta ? shortLeft(endMs - now) : null;
            if (chip != null) b.setShortCriticalText(chip);
            Bundle ex = new Bundle(); ex.putBoolean("android.requestPromotedOngoing", true); b.addExtras(ex);   // the extra NotificationCompat writes; the builder method exists from API 36.1
            try { Notification.Builder.class.getMethod("setRequestPromotedOngoing", boolean.class).invoke(b, true); } catch (Exception ignored) { }
        } else b.setProgress(100, Math.max(pct, 0), pct < 0);
        Notification n = b.build();
        if (Build.VERSION.SDK_INT >= 36 && !promoLogged) {   // once per process: why a Live Update would not be shown
            promoLogged = true;
            Log.i("HexAnima", "Live Update: canPostPromotedNotifications=" + c.getSystemService(NotificationManager.class).canPostPromotedNotifications() + " promotable=" + n.hasPromotableCharacteristics());
        }
        return n;
    }

    static String shortLeft(long ms) {   // for the chip: "4m" / "40s"
        long s = Math.max(1, ms / 1000);
        return s >= 90 ? L.f("%dм", "%dm", (s + 30) / 60) : L.f("%dс", "%ds", Math.max(5, (s + 4) / 5 * 5));
    }

    static String left(long ms) {
        long s = Math.max(1, ms / 1000);
        return s >= 90 ? L.f("≈ %d мин осталось", "about %d min left", (s + 30) / 60) : L.f("≈ %d с осталось", "about %d s left", Math.max(5, (s + 4) / 5 * 5));
    }

    /** replaces the notification: at once when the step or title changes, otherwise at most every 2 s and only when the estimate or the chip text changed */
    static synchronized void update(Context c, String title, int pct, int step, int total, long endMs) {
        if (!live) return;
        long now = System.currentTimeMillis();
        String sl = endMs > now + 1000 ? shortLeft(endMs - now) : "";   // the chip text ticks down between steps
        boolean changed = step != shownStep || !title.equals(shownTitle), drift = Math.abs(endMs - shownEnd) >= 5000 || !sl.equals(shownLeft);
        if (!changed && (now - shownAt < 2000 || !drift)) return;
        shownStep = step; shownTitle = title; shownEnd = endMs; shownAt = now; shownLeft = sl;
        try { c.getSystemService(NotificationManager.class).notify(ID, build(c, title, pct, step, total, endMs)); } catch (Exception e) { Log.w("HexAnima", "gen notification: " + e); }
    }

    static synchronized void reset() { shownStep = -2; shownTitle = ""; shownLeft = ""; shownEnd = 0; shownAt = 0; }
}
