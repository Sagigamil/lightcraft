// On-device test driver for the MobileSAM QNN runner (4 mask options per prompt).

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" int mobilesam_run_points(const char* lib_dir, const char* dir, const float* image, size_t image_floats,
                                    const float* points, size_t num_points, const float* labels, float* masks, float* scores,
                                    long* encoder_ms, long* decoder_ms);
extern "C" int mobilesam_run_prompt(const char* lib_dir, const char* dir, const float* image, size_t image_floats,
                                    const float* coords, const float* labels, size_t num_points, float* masks, float* scores,
                                    long* encoder_ms, long* decoder_ms);

constexpr size_t kOptions = 4;
constexpr size_t kMask = 256 * 256;

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

static void report(const char* label, const float* mask, float score) {
    float lo = mask[0], hi = mask[0];
    size_t positive = 0;
    for (size_t i = 0; i < kMask; ++i) {
        lo = mask[i] < lo ? mask[i] : lo;
        hi = mask[i] > hi ? mask[i] : hi;
        positive += mask[i] > 0.0f ? 1 : 0;
    }
    std::printf("%s score=%.4f mask min=%.4f max=%.4f positive=%zu\n", label, score, lo, hi, positive);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: mobilesam_test <libdir> <ctxdir> <image.raw> <points.raw> <labels.raw> <outdir> [prompt|points]\n");
        return 2;
    }
    const bool prompt = argc > 7 && std::strcmp(argv[7], "prompt") == 0;
    auto image = read_f32(argv[3]);
    auto points = read_f32(argv[4]);
    auto labels = read_f32(argv[5]);
    if (image.empty() || points.empty() || points.size() % 2 != 0 || labels.size() != points.size() / 2) {
        std::fprintf(stderr, "bad inputs: image %zu floats, points %zu, labels %zu\n", image.size(), points.size(), labels.size());
        return 2;
    }
    const size_t n = labels.size();
    long enc_ms = 0, dec_ms = 0;
    int rc = 0;
    if (prompt) {
        std::vector<float> masks(kOptions * kMask), scores(kOptions);
        rc = mobilesam_run_prompt(argv[1], argv[2], image.data(), image.size(), points.data(), labels.data(), n, masks.data(),
                                  scores.data(), &enc_ms, &dec_ms);
        std::printf("rc=%d prompt points=%zu encoder_ms=%ld decoder_ms=%ld\n", rc, n, enc_ms, dec_ms);
        if (rc != 0) return rc;
        for (size_t i = 0; i < kOptions; ++i) {
            char label[32];
            std::snprintf(label, sizeof(label), "mask %zu", i);
            report(label, masks.data() + i * kMask, scores[i]);
        }
        if (!write_f32(std::string(argv[6]) + "/masks.raw", masks)) return 3;
        if (!write_f32(std::string(argv[6]) + "/scores.raw", scores)) return 3;
        return 0;
    }
    std::vector<float> masks(n * kOptions * kMask), scores(n * kOptions);
    rc = mobilesam_run_points(argv[1], argv[2], image.data(), image.size(), points.data(), n, labels.data(), masks.data(),
                              scores.data(), &enc_ms, &dec_ms);
    std::printf("rc=%d points=%zu encoder_ms=%ld decoder_total_ms=%ld\n", rc, n, enc_ms, dec_ms);
    if (rc != 0) return rc;
    for (size_t p = 0; p < n; ++p) {
        for (size_t i = 0; i < kOptions; ++i) {
            char label[64];
            std::snprintf(label, sizeof(label), "point %zu (%.1f, %.1f) label %.0f mask %zu", p, points[2 * p], points[2 * p + 1],
                          labels[p], i);
            report(label, masks.data() + (p * kOptions + i) * kMask, scores[p * kOptions + i]);
        }
    }
    if (!write_f32(std::string(argv[6]) + "/masks.raw", masks)) return 3;
    if (!write_f32(std::string(argv[6]) + "/scores.raw", scores)) return 3;
    return 0;
}
