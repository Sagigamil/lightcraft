package com.lightcraft.android;

/**
 * MobileSAM on the Hexagon NPU (AI edition only).
 *
 * <p>The image encoder runs on the HTP and the small prompt decoder on the QNN CPU backend (the
 * HTP decoder is not accurate for this model). The contexts are extracted from the APK on first
 * use; {@code encode} writes the embeddings for the cached photo, {@code decode} runs one
 * multi-point prompt. Both return their time in milliseconds, or a negative error code.
 */
final class MobileSam {
    static {
        System.loadLibrary("mobilesam");
    }

    private MobileSam() {}

    /** {@code input}: 1024x1024x3 float32 image; {@code output}: 64x64x256 float32 embeddings. */
    static native long encode(String dir, String libDir, String input, String output);

    /** {@code points}: n x 2 float32, {@code labels}: n float32; {@code mask}: 256x256 logits, {@code score}: one float. */
    static native long decode(String dir, String libDir, String embeddings, String points, String labels, String mask, String score);
}
