package com.hexanima.npu;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.os.PowerManager;

/** Keeps the process alive (foreground service, type dataSync) while model files are downloaded or pictures are generated; the work itself runs in Main.
 *  Generation also holds a partial wake lock: with the screen off the CPU would otherwise suspend between the NPU calls and the picture would stall.
 *  (On HyperOS a foreground service alone does not stop the system freezer: Main asks for "no battery restrictions", see Main.askBackground.) */
public class DownloadService extends Service {
    private PowerManager.WakeLock wl;

    @Override public int onStartCommand(Intent i, int f, int id) {
        L.init(this);
        NotificationManager nm = getSystemService(NotificationManager.class);
        boolean gen = i != null && "gen".equals(i.getStringExtra("mode"));
        if (gen) {
            GenNote.channel(this); GenNote.reset();
            startForeground(GenNote.ID, GenNote.build(this, L.s("Запуск…", "Starting…"), -1, -1, 0, 0));
            GenNote.live = true;
            if (wl == null) { wl = getSystemService(PowerManager.class).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "hexanima:generate"); wl.setReferenceCounted(false); }
            wl.acquire(4 * 3600 * 1000L);   // a safety timeout; released when Main stops the service
            return START_NOT_STICKY;
        }
        nm.createNotificationChannel(new NotificationChannel("dl", L.s("Загрузка моделей", "Model downloads"), NotificationManager.IMPORTANCE_LOW));
        Notification n = new Notification.Builder(this, "dl").setSmallIcon(android.R.drawable.stat_sys_download).setContentTitle("HexAnima")
            .setContentText(L.s("Скачиваю файлы модели…", "Downloading model files…")).setOngoing(true).build();
        startForeground(1, n);
        return START_NOT_STICKY;
    }

    @Override public void onDestroy() {
        GenNote.live = false;
        if (wl != null && wl.isHeld()) wl.release();
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent i) { return null; }
}
