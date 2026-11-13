package com.lightcraft.android;

/**
 * Runs SAM 3's vision backbone on the Hexagon NPU through QNN (AI edition only).
 *
 * <p>The native library loads the chained QNN context binaries (stem, two 16-block chunks)
 * from the app's private storage, passes the preprocessed image through them on the HTP and
 * writes the 72×72×1024 token tensor. The neck, prompt encoder and mask decoder still run in
 * the Rust engine on the CPU, so the CPU path remains the fallback.
 */
final class QnnEncoder {
    static {
        System.loadLibrary("qnn_chain");
    }

    private QnnEncoder() {}

    /**
     * Executes the chain and returns its time in milliseconds, or a negative failure code.
     *
     * @param dir      directory with the context binaries (and, for a fallback lookup, the libs)
     * @param libDir   where to look for libQnnHtp.so if the loader path does not have it
     * @param spec     one stage per line: context|graph|input|dims|inputId|output|dims|outputId
     * @param input    preprocessed image, float32 NCHW [1,3,1008,1008]
     * @param output   receives float32 tokens NHWC [1,72,72,1024]
     */
    static native long encode(String dir, String libDir, String spec, String input, String output);
}
