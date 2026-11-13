// Chained QNN HTP encoder runner for Lightcraft.
//
// Loads several QNN context binaries (stem, ViT block chunks) and executes them in sequence on
// the Hexagon NPU, passing the intermediate token tensor from one graph to the next.  This is the
// same flow `qnn-net-run` uses, but in-process and without files between stages.
//
// The QNN SDK headers are used at compile time; the runtime libraries (libQnnHtp.so, the v81
// stub/skel) are packaged in the AI flavor and loaded with dlopen.

#include <dlfcn.h>
#include <android/log.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <QnnInterface.h>
#include <QnnTypes.h>
#include <QnnContext.h>
#include <QnnGraph.h>
#include <QnnTensor.h>
#include <QnnBackend.h>
#include <QnnDevice.h>

#define LOG_TAG "LightCraftQnn"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Stage {
    std::string context;
    std::string graph;
    std::string input;
    std::vector<uint32_t> input_dims;
    uint32_t input_id = 0;
    std::string output;
    std::vector<uint32_t> output_dims;
    uint32_t output_id = 0;
};

struct Engine {
    void* lib = nullptr;
    const QnnInterface_t* iface = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    bool ready = false;
    std::string error;
};

Engine g_engine;

using QnnVTable = QNN_INTERFACE_VER_TYPE;
using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

const QnnVTable* fn() {
    return g_engine.iface ? &g_engine.iface->QNN_INTERFACE_VER_NAME : nullptr;
}

bool init_engine(Engine& e) {
    if (e.ready) return true;
    e.lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (!e.lib) {
        e.error = std::string("dlopen libQnnHtp.so: ") + dlerror();
        return false;
    }
    auto get_providers = reinterpret_cast<GetProvidersFn>(dlsym(e.lib, "QnnInterface_getProviders"));
    if (!get_providers) {
        e.error = "QnnInterface_getProviders not found";
        return false;
    }
    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    if (get_providers(&providers, &count) != QNN_SUCCESS || !providers || count == 0) {
        e.error = "QnnInterface_getProviders failed";
        return false;
    }
    e.iface = providers[0];
    const auto* q = &e.iface->QNN_INTERFACE_VER_NAME;
    if (q->backendCreate(nullptr, nullptr, &e.backend) != QNN_SUCCESS || !e.backend) {
        e.error = "QnnBackend_create failed";
        return false;
    }
    if (q->deviceCreate(nullptr, nullptr, &e.device) != QNN_SUCCESS || !e.device) {
        e.error = "QnnDevice_create failed";
        return false;
    }
    e.ready = true;
    LOGI("HTP backend ready (QNN API %d.%d)", QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR);
    return true;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto n = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> b(static_cast<size_t>(n));
    if (!f.read(reinterpret_cast<char*>(b.data()), n)) return {};
    return b;
}

bool write_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(f);
}

// "context|graph|input|dims|input_id|output|dims|output_id"
Stage parse_stage(const std::string& line) {
    Stage s;
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= line.size()) {
        auto p = line.find('|', start);
        parts.push_back(line.substr(start, p == std::string::npos ? std::string::npos : p - start));
        if (p == std::string::npos) break;
        start = p + 1;
    }
    if (parts.size() < 8) return s;
    s.context = parts[0];
    s.graph = parts[1];
    s.input = parts[2];
    s.input_id = static_cast<uint32_t>(std::stoul(parts[4]));
    s.output = parts[5];
    s.output_id = static_cast<uint32_t>(std::stoul(parts[7]));
    auto dims = [](const std::string& text, std::vector<uint32_t>& out) {
        size_t i = 0;
        while (i <= text.size()) {
            auto c = text.find(',', i);
            auto v = text.substr(i, c == std::string::npos ? std::string::npos : c - i);
            if (!v.empty()) out.push_back(static_cast<uint32_t>(std::stoul(v)));
            if (c == std::string::npos) break;
            i = c + 1;
        }
    };
    dims(parts[3], s.input_dims);
    dims(parts[6], s.output_dims);
    return s;
}

Qnn_Tensor_t make_tensor(uint32_t id, const std::string& name, const std::vector<uint32_t>& dims, Qnn_TensorType_t type, void* data,
                         size_t bytes) {
    Qnn_Tensor_t t{};  // zero every field, including the v2 quantize params
    t.version = QNN_TENSOR_VERSION_2;
    t.v2.id = id;
    t.v2.name = name.c_str();
    t.v2.type = type;
    t.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v2.dataType = QNN_DATATYPE_FLOAT_32;
    t.v2.rank = static_cast<uint32_t>(dims.size());
    t.v2.dimensions = const_cast<uint32_t*>(dims.data());
    t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    t.v2.clientBuf.data = data;
    t.v2.clientBuf.dataSize = bytes;
    return t;
}

}  // namespace

extern "C" int qnn_run_chain(const char* dir, const char* spec, const float* input, size_t input_floats, const char* output_path,
                             long* encoder_ms) {
    if (!init_engine(g_engine)) {
        std::fprintf(stderr, "init failed: %s\n", g_engine.error.c_str());
        return -1;
    }
    const auto* q = fn();
    std::vector<Stage> stages;
    {
        std::string text(spec);
        size_t start = 0;
        while (start < text.size()) {
            auto nl = text.find('\n', start);
            auto line = text.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
            if (!line.empty()) stages.push_back(parse_stage(line));
            if (nl == std::string::npos) break;
            start = nl + 1;
        }
    }
    if (stages.empty()) {
        std::fprintf(stderr, "empty stage list\n");
        return -1;
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> current(input, input + input_floats);
    for (size_t i = 0; i < stages.size(); ++i) {
        const Stage& s = stages[i];
        std::string path = std::string(dir) + "/" + s.context;
        std::vector<uint8_t> binary = read_file(path);
        if (binary.empty()) {
            std::fprintf(stderr, "cannot read %s\n", path.c_str());
            return -2;
        }
        Qnn_ContextHandle_t ctx = nullptr;
        if (q->contextCreateFromBinary(g_engine.backend, g_engine.device, nullptr, binary.data(), binary.size(), &ctx, nullptr) != QNN_SUCCESS) {
            std::fprintf(stderr, "contextCreateFromBinary failed for %s\n", s.context.c_str());
            return -3;
        }
        Qnn_GraphHandle_t graph = nullptr;
        if (q->graphRetrieve(ctx, s.graph.c_str(), &graph) != QNN_SUCCESS || !graph) {
            std::fprintf(stderr, "graphRetrieve(%s) failed\n", s.graph.c_str());
            return -4;
        }

        size_t in_floats = 1;
        for (auto d : s.input_dims) in_floats *= d;
        size_t out_floats = 1;
        for (auto d : s.output_dims) out_floats *= d;
        if (in_floats != current.size()) {
            std::fprintf(stderr, "stage %zu input mismatch: spec %zu, data %zu\n", i, in_floats, current.size());
            return -5;
        }
        std::vector<float> out(out_floats);

        // The context already created the graph's tensors; bind our buffers by id/name and execute.
        Qnn_Tensor_t tin = make_tensor(s.input_id, s.input, s.input_dims, QNN_TENSOR_TYPE_APP_WRITE, current.data(), current.size() * sizeof(float));
        Qnn_Tensor_t tout = make_tensor(s.output_id, s.output, s.output_dims, QNN_TENSOR_TYPE_APP_READ, out.data(), out.size() * sizeof(float));

        auto ts = std::chrono::steady_clock::now();
        Qnn_ErrorHandle_t gerr = q->graphExecute(graph, &tin, 1, &tout, 1, nullptr, nullptr);
        if (gerr != QNN_SUCCESS) {
            std::fprintf(stderr, "graphExecute failed (stage %zu) err %llu\n", i, static_cast<unsigned long long>(gerr));
            return -8;
        }
        auto te = std::chrono::steady_clock::now();
        std::fprintf(stderr, "stage %zu (%s) %.1f ms\n", i, s.graph.c_str(), std::chrono::duration<double, std::milli>(te - ts).count());
        current.swap(out);
        q->contextFree(ctx, nullptr);
    }
    auto t1 = std::chrono::steady_clock::now();
    if (encoder_ms) *encoder_ms = static_cast<long>(std::chrono::duration<double, std::milli>(t1 - t0).count());
    if (!write_file(output_path, current.data(), current.size() * sizeof(float))) {
        std::fprintf(stderr, "cannot write %s\n", output_path);
        return -9;
    }
    std::fprintf(stderr, "chain done in %ld ms\n", encoder_ms ? *encoder_ms : 0);
    return 0;
}

extern "C" const char* qnn_backend_name() {
    return init_engine(g_engine) ? "Qualcomm Hexagon HTP (QNN)" : "unavailable";
}
