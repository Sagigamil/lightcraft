package com.lightcraft.android;

import android.app.NativeActivity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.DocumentsContract;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;

/** Only platform services live here. All editing and UI code lives in Rust. */
public final class MainActivity extends NativeActivity {
    private static final int OPEN = 101, DESTINATION = 102, SAVE = 103;
    private final ConcurrentLinkedQueue<JSONObject> events = new ConcurrentLinkedQueue<>();
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Object sourceLock = new Object();
    private final Object modelLock = new Object();
    private volatile boolean recursive = false;
    /** The album an "open" was for (an album's Add Photos from Folder...), -1 = none. */
    private volatile long openAlbum = -1;
    private volatile String lastExport = "", savePending = "";
    private SharedPreferences prefs;
    private volatile CountDownLatch saveResult;
    private volatile Uri saveUri;
    private static final Set<String> EXTENSIONS = new HashSet<>(Arrays.asList(
        "jpg","jpeg","png","tif","tiff","webp","dng","cr2","cr3","nef","nrw","arw","raf","orf","rw2","rwl","raw","pef","psd","jxl","gif","bmp","heic","avif"));

    @Override public void onCreate(Bundle state) {
        prefs = getSharedPreferences("folders", MODE_PRIVATE);
        // Discard only our abandoned per-read temporary files after process death.
        File[] stale = getCacheDir().listFiles((dir, name) -> name.startsWith("source-") && name.endsWith(".bin"));
        if (stale != null) for (File file : stale) file.delete();
        super.onCreate(state);
        // NativeActivity surfaces otherwise lay out behind the transparent system
        // bars on some Android 10+ devices. Keep the egui toolbar below the clock
        // and navigation affordances so its hit targets match what is drawn.
        if (android.os.Build.VERSION.SDK_INT >= 30) getWindow().setDecorFitsSystemWindows(true);
        getWindow().getDecorView().setSystemUiVisibility(0);
        getWindow().setStatusBarColor(0xff191919);
        getWindow().setNavigationBarColor(0xff191919);
        // the soft keyboard's target (see ImeView): without it, Backspace is often dropped
        ImeView ime = new ImeView(this);
        addContentView(ime, new android.view.ViewGroup.LayoutParams(1, 1));
        ime.requestFocus();
    }
    @Override protected void onPause() { event("save", ""); super.onPause(); }
    @Override public void onTrimMemory(int level) { super.onTrimMemory(level); event("memory", Integer.toString(level)); }
    @Override public void onBackPressed() { event("back", ""); }

    private void event(String kind, String value) {
        try { events.add(new JSONObject().put("kind", kind).put("value", value)); }
        catch (org.json.JSONException ignored) { android.util.Log.e("LightCraft", "Could not queue event"); }
    }
    private void picker(int code) {
        runOnUiThread(() -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
            startActivityForResult(intent, code);
        });
    }
    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == SAVE) {
            saveUri = result == RESULT_OK && data != null ? data.getData() : null;
            CountDownLatch waiting = saveResult;
            if (waiting != null) waiting.countDown();
            return;
        }
        // the album travels with this pick only: a cancelled pick or a later Refresh has none
        long album = request == OPEN ? openAlbum : -1;
        if (request == OPEN) openAlbum = -1;
        if (result != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        try {
            int flags = data.getFlags() & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            getContentResolver().takePersistableUriPermission(uri, flags);
            prefs.edit().putString(request == OPEN ? "source" : "destination", uri.toString()).commit();
            if (request == OPEN) scan(uri, album); else {
                prefs.edit().putBoolean("createDocument", false).apply();
                event("destination", uri.toString());
                event("notice", "Export folder selected. Open File > Export to save photos.");
            }
        } catch (Exception e) { event("error", "Folder permission: " + e); }
    }
    private void scan(Uri root, long album) {
        worker.execute(() -> {
            try {
                JSONArray files = new JSONArray();
                walk(root, DocumentsContract.getTreeDocumentId(root), files, new HashSet<>(), 0);
                JSONObject found = new JSONObject().put("kind", "files").put("files", files).put("tree", root.toString());
                if (album >= 0) found.put("album", album);
                events.add(found);
            } catch (Exception e) { event("error", "Reopen Folder to authorize access: " + e); }
        });
    }
    private void walk(Uri tree, String parent, JSONArray files, Set<String> seen, int depth) throws Exception {
        if (depth > 64 || !seen.add(parent)) return;
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parent);
        String[] cols = {DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE, DocumentsContract.Document.COLUMN_SIZE, DocumentsContract.Document.COLUMN_LAST_MODIFIED};
        try (Cursor c = getContentResolver().query(children, cols, null, null, null)) {
            if (c == null) throw new IOException("Provider returned no directory listing");
            while (c.moveToNext()) {
                String id = c.getString(0), name = c.getString(1), mime = c.getString(2);
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) {
                    if (recursive) walk(tree, id, files, seen, depth + 1);
                } else if (name != null && EXTENSIONS.contains(name.substring(name.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT))) {
                    if (files.length() >= 100000) throw new IOException("Folder scan limit is 100000 photos");
                    files.put(new JSONObject().put("uri", DocumentsContract.buildDocumentUriUsingTree(tree, id).toString())
                        .put("name", name).put("size", c.getLong(3)).put("modified", c.getLong(4)));
                }
            }
        }
    }
    private File materialize(Uri uri) throws Exception {
        // Providers may supply pipes. Sequential streams work for both pipes and regular files.
        synchronized (sourceLock) {
            File f = File.createTempFile("source-", ".bin", getCacheDir());
            try (InputStream in = getContentResolver().openInputStream(uri); FileOutputStream out = new FileOutputStream(f)) {
                if (in == null) throw new IOException("Provider returned no image stream");
                byte[] buffer = new byte[65536]; long size = 0; int n;
                while ((n = in.read(buffer)) != -1) {
                    size += n;
                    if (size > 512L * 1024 * 1024) throw new IOException("Image exceeds the 512 MiB source limit");
                    out.write(buffer, 0, n);
                }
                return f;
            } catch (Exception e) { f.delete(); throw e; }
        }
    }
    private void copyToDocument(File source, Uri target) throws Exception {
        try (InputStream in = new FileInputStream(source); OutputStream out = getContentResolver().openOutputStream(target, "w")) {
            if (out == null) throw new IOException("Provider returned no export stream");
            byte[] buf = new byte[65536]; int n;
            while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
            out.flush();
        }
    }
    private Uri export(File file, String name) throws Exception {
        if (prefs.getBoolean("createDocument", false)) {
            saveUri = null;
            CountDownLatch waiting = new CountDownLatch(1);
            saveResult = waiting;
            runOnUiThread(() -> startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT)
                .addCategory(Intent.CATEGORY_OPENABLE).setType(mime(name)).putExtra(Intent.EXTRA_TITLE, name), SAVE));
            if (!waiting.await(10, TimeUnit.MINUTES) || saveUri == null) throw new IOException("Export document selection cancelled");
            Uri target = saveUri;
            saveResult = null;
            // ACTION_CREATE_DOCUMENT creates a new document; Android resolves collisions.
            copyToDocument(file, target);
            lastExport = target.toString();
            return target;
        }
        String selected = prefs.getString("destination", "");
        if (selected.isEmpty()) throw new IOException("Choose Export Folder before exporting");
        Uri tree = Uri.parse(selected);
        Uri parent = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree));
        Uri document = findExport(name);
        boolean created = document == null;
        if (document != null && !prefs.getStringSet("exports", Collections.emptySet()).contains(document.toString()))
            throw new IOException("Refusing to overwrite a source or a file not exported by LightCraft; choose Add number");
        if (created) document = DocumentsContract.createDocument(getContentResolver(), parent, mime(name), name);
        if (document == null) throw new IOException("Provider could not create export");
        try { copyToDocument(file, document); }
        catch (Exception e) { if (created) try { DocumentsContract.deleteDocument(getContentResolver(), document); } catch (Exception ignored) {} throw e; }
        lastExport = document.toString();
        Set<String> exported = new HashSet<>(prefs.getStringSet("exports", Collections.emptySet()));
        exported.add(lastExport);
        prefs.edit().putStringSet("exports", exported).commit();
        return document;
    }
    private static String mime(String name) {
        String ext = name.substring(name.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT);
        return ext.equals("jpg") || ext.equals("jpeg") ? "image/jpeg" : ext.equals("png") ? "image/png" :
            ext.equals("tif") || ext.equals("tiff") ? "image/tiff" : "application/octet-stream";
    }
    private Uri findExport(String name) throws Exception {
        if (prefs.getBoolean("createDocument", false)) return null;
        String selected = prefs.getString("destination", "");
        if (selected.isEmpty()) return null;
        Uri tree = Uri.parse(selected);
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree));
        try (Cursor c = getContentResolver().query(children, new String[]{DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME}, null, null, null)) {
            if (c == null) throw new IOException("Destination listing unavailable");
            while (c.moveToNext()) if (name.equals(c.getString(1))) return DocumentsContract.buildDocumentUriUsingTree(tree, c.getString(0));
        }
        return null;
    }

    /**
     * Copy the embedded SAM 3 point-prompt assets out of the APK once. Every file is
     * size- and SHA-256-verified against the packaged manifest; a .ready marker names
     * the model hash so later launches skip the copy. Nothing is downloaded.
     */
    private JSONObject extractModel() throws Exception {
        synchronized (modelLock) {
            File dir = new File(getFilesDir(), "models/sam3");
            if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException("Could not create " + dir);
            JSONObject manifest = new JSONObject(new String(readAll(getAssets().open("sam3/manifest.json")), StandardCharsets.UTF_8));
            JSONObject files = manifest.getJSONObject("files");
            String modelSha = files.getJSONObject("model.safetensors").getString("sha256");
            File ready = new File(dir, ".ready");
            boolean current = ready.isFile() && modelSha.equals(new String(readAll(new FileInputStream(ready)), StandardCharsets.UTF_8).trim());
            String[] required = {"manifest.json", "model.safetensors", "vocab.json", "merges.txt", "LICENSE"};
            JSONArray copied = new JSONArray(), skipped = new JSONArray();
            for (String name : required) {
                File target = new File(dir, name);
                JSONObject info = name.equals("manifest.json") ? null : files.getJSONObject(name);
                if (current && target.isFile() && (info == null || target.length() == info.getLong("size"))) {
                    skipped.put(name);
                    continue;
                }
                copyAsset("sam3/" + name, target, info == null ? null : info.getString("sha256"));
                copied.put(name);
            }
            Files.write(ready.toPath(), modelSha.getBytes(StandardCharsets.UTF_8));
            android.util.Log.i("LightCraft", "AI model assets: " + copied.length() + " copied, " + skipped.length() + " already present");
            return new JSONObject().put("files", copied).put("skipped", skipped);
        }
    }

    private void copyAsset(String asset, File target, String expectedSha) throws Exception {
        File part = new File(target.getParentFile(), target.getName() + ".part");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = getAssets().open(asset); OutputStream out = new FileOutputStream(part)) {
            byte[] buffer = new byte[1 << 20];
            int n;
            while ((n = in.read(buffer)) != -1) {
                out.write(buffer, 0, n);
                digest.update(buffer, 0, n);
            }
        } catch (Exception e) {
            part.delete();
            throw e;
        }
        String got = hex(digest.digest());
        if (expectedSha != null && !expectedSha.equals(got)) {
            part.delete();
            throw new IOException("Embedded model file " + asset + " is damaged (SHA-256 mismatch)");
        }
        Files.move(part.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
    }

    private static byte[] readAll(InputStream in) throws IOException {
        try (InputStream input = in; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int n;
            while ((n = input.read(buffer)) != -1) out.write(buffer, 0, n);
            return out.toByteArray();
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder s = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) s.append(String.format("%02x", b & 0xff));
        return s.toString();
    }

    /** Called by attached Rust worker threads; never blocks Android's main thread on IO. */
    public String bridge(String request) {
        try {
            JSONObject q = new JSONObject(request), r = new JSONObject();
            switch (q.getString("op")) {
                case "poll": { JSONArray a = new JSONArray(); JSONObject e; while ((e = events.poll()) != null) a.put(e); r.put("events", a); break; }
                case "open": recursive = q.optBoolean("recursive"); openAlbum = q.optLong("album", -1); picker(OPEN); break;
                case "destination": picker(DESTINATION); break;
                case "exportDestination": r.put("value", prefs.getBoolean("createDocument", false) ? "Android Save As" : prefs.getString("destination", "")); break;
                case "documentMode":
                    prefs.edit().putBoolean("createDocument", true).apply();
                    event("destination", "Android Save As");
                    event("notice", "Exports will use Android Save As"); break;
                case "sourceDestination": {
                    String source = prefs.getString("source", "");
                    if (source.isEmpty()) throw new IOException("Open a folder first");
                    prefs.edit().putString("destination", source).putBoolean("createDocument", false).apply();
                    event("destination", source); break;
                }
                case "exportExists": r.put("exists", findExport(q.getString("name")) != null); break;
                case "refresh": {
                    recursive = q.optBoolean("recursive");
                    String root = prefs.getString("source", ""); if (!root.isEmpty()) scan(Uri.parse(root), -1); break;
                }
                case "read": r.put("path", materialize(Uri.parse(q.getString("uri"))).getAbsolutePath()); break;
                case "exists": {
                    try (Cursor c = getContentResolver().query(Uri.parse(q.getString("uri")), new String[]{DocumentsContract.Document.COLUMN_DOCUMENT_ID}, null, null, null)) {
                        r.put("exists", c != null && c.moveToFirst());
                    } break;
                }
                case "export": r.put("uri", export(new File(q.getString("path")), q.getString("name")).toString()); break;
                case "share": {
                    if (lastExport.isEmpty()) throw new IOException("Export a photo before sharing");
                    Uri uri = Uri.parse(lastExport);
                    runOnUiThread(() -> {
                        Intent share = new Intent(Intent.ACTION_SEND).setType(getContentResolver().getType(uri));
                        share.putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                        startActivity(Intent.createChooser(share, "Share edited photo"));
                    }); break;
                }
                case "extractModel": r.put("result", extractModel()); break;
                case "finish": runOnUiThread(this::finish); break;
                default: throw new IOException("Unknown Android operation");
            }
            return r.toString();
        } catch (Exception e) {
            android.util.Log.e("LightCraft", "Platform operation failed", e);
            try { return new JSONObject().put("error", e.toString()).toString(); }
            catch (Exception ignored) { return "{\"error\":\"Android platform failure\"}"; }
        }
    }
}
