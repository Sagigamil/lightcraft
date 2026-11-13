// On-device test driver for the chained QNN encoder.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

extern "C" int qnn_run_chain(const char* dir, const char* spec, const float* input, size_t input_floats, const char* output_path,
                             long* encoder_ms);

static std::string read_text(const char* path) {
    std::ifstream f(path);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: qnn_chain_test <dir> <spec.txt> <input.raw> <output.raw>\n");
        return 2;
    }
    std::string spec = read_text(argv[2]);
    std::ifstream in(argv[3], std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[3]);
        return 2;
    }
    auto bytes = static_cast<size_t>(in.tellg());
    in.seekg(0);
    std::vector<float> data(bytes / sizeof(float));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    long ms = 0;
    int rc = qnn_run_chain(argv[1], spec.c_str(), data.data(), data.size(), argv[4], &ms);
    std::printf("rc=%d encoder_ms=%ld\n", rc, ms);
    return rc;
}
