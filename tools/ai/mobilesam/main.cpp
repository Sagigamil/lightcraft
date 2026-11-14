// On-device test driver for the MobileSAM QNN runner.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

extern "C" int mobilesam_run_points(const char* lib_dir, const char* dir, const float* image, size_t image_floats,
                                    const float* points, size_t num_points, const float* labels, float* masks, float* scores,
                                    long* encoder_ms, long* decoder_ms);

static std::vector<float> read_f32(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", path);
        return {};
    }
    auto n = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> v(n / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    return v;
}

static bool write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return static_cast<bool>(f);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: mobilesam_test <libdir> <ctxdir> <image.raw> <points.raw> <labels.raw> <outdir>\n");
        return 2;
    }
    auto image = read_f32(argv[3]);
    auto points = read_f32(argv[4]);
    auto labels = read_f32(argv[5]);
    if (image.empty() || points.empty() || points.size() % 2 != 0 || labels.size() != points.size() / 2) {
        std::fprintf(stderr, "bad inputs: image %zu floats, points %zu, labels %zu\n", image.size(), points.size(), labels.size());
        return 2;
    }
    const size_t n = labels.size();
    std::vector<float> masks(n * 256 * 256), scores(n);
    long enc_ms = 0, dec_ms = 0;
    int rc = mobilesam_run_points(argv[1], argv[2], image.data(), image.size(), points.data(), n, labels.data(), masks.data(),
                                  scores.data(), &enc_ms, &dec_ms);
    std::printf("rc=%d points=%zu encoder_ms=%ld decoder_total_ms=%ld\n", rc, n, enc_ms, dec_ms);
    if (rc != 0) return rc;
    for (size_t p = 0; p < n; ++p) {
        const float* m = masks.data() + p * 256 * 256;
        float lo = m[0], hi = m[0];
        size_t positive = 0;
        for (size_t i = 0; i < 256 * 256; ++i) {
            lo = m[i] < lo ? m[i] : lo;
            hi = m[i] > hi ? m[i] : hi;
            positive += m[i] > 0.0f ? 1 : 0;
        }
        std::printf("point %zu (%.1f, %.1f) label %.0f: score=%.4f mask min=%.4f max=%.4f positive=%zu\n", p, points[2 * p],
                    points[2 * p + 1], labels[p], scores[p], lo, hi, positive);
    }
    if (!write_f32(std::string(argv[6]) + "/masks.raw", masks)) return 3;
    if (!write_f32(std::string(argv[6]) + "/scores.raw", scores)) return 3;
    return 0;
}
