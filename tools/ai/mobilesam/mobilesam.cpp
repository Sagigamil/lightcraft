// MobileSAM encoder + point decoder on the Hexagon HTP, through two QNN context binaries.
//
// The contexts are produced on the device from Qualcomm's float DLCs:
//   qnn-context-binary-generator --dlc_path encoder.dlc --backend libQnnHtp.so ...
// The encoder takes [1,1024,1024,3] NHWC float32 in 0..1 (mean/std are inside the graph) and
// returns [1,64,64,256]; the decoder takes that plus one point ([1,1,2] coords in 0..1024 and
// [1,1] labels) and returns [1,256,256,1] mask logits and [1,1] score.

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

#define LOG_TAG "LightCraftMobileSam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Engine {
    void* lib = nullptr;
    const QnnInterface_t* iface = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
};

Engine g_engine;
using QnnVTable = QNN_INTERFACE_VER_TYPE;
using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

const QnnVTable* fn() {
    return g_engine.iface ? &g_engine.iface->QNN_INTERFACE_VER_NAME : nullptr;
}

bool init_engine(const char* lib_dir) {
    if (g_engine.device) return true;
    const std::string dir = lib_dir ? lib_dir : "";
    auto open_lib = [&dir](const char* name, int flags) -> void* {
        void* handle = nullptr;
        if (!dir.empty() && dir[0] == '/') handle = dlopen((dir + "/" + name).c_str(), flags);
        if (!handle) handle = dlopen(name, flags);
        if (handle) {
            LOGI("loaded %s", name);
        } else {
            const char* why = dlerror();
            LOGE("cannot load %s: %s", name, why ? why : "unknown error");
        }
        return handle;
    };
    open_lib("libQnnHtpV81Stub.so", RTLD_NOW | RTLD_LOCAL);
    g_engine.lib = open_lib("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_engine.lib) return false;
    auto get_providers = reinterpret_cast<GetProvidersFn>(dlsym(g_engine.lib, "QnnInterface_getProviders"));
    if (!get_providers) return false;
    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    if (get_providers(&providers, &count) != QNN_SUCCESS || !providers || count == 0) return false;
    g_engine.iface = providers[0];
    const auto* q = fn();
    if (q->backendCreate(nullptr, nullptr, &g_engine.backend) != QNN_SUCCESS || !g_engine.backend) return false;
    if (q->deviceCreate(nullptr, nullptr, &g_engine.device) != QNN_SUCCESS || !g_engine.device) return false;
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

Qnn_Tensor_t tensor(uint32_t id, const char* name, const std::vector<uint32_t>& dims, Qnn_TensorType_t type, void* data, size_t bytes) {
    Qnn_Tensor_t t{};
    t.version = QNN_TENSOR_VERSION_2;
    t.v2.id = id;
    t.v2.name = name;
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

struct Input {
    uint32_t id;
    const char* name;
    std::vector<uint32_t> dims;
    void* data;
    size_t bytes;
};

struct Output {
    uint32_t id;
    const char* name;
    std::vector<uint32_t> dims;
    void* data;
    size_t bytes;
};

struct Context {
    Qnn_ContextHandle_t handle = nullptr;
};

bool open_context(const std::string& dir, const char* file, Context& ctx) {
    const auto* q = fn();
    std::vector<uint8_t> binary = read_file(dir + "/" + file);
    if (binary.empty()) {
        LOGE("cannot read %s/%s", dir.c_str(), file);
        return false;
    }
    if (q->contextCreateFromBinary(g_engine.backend, g_engine.device, nullptr, binary.data(), binary.size(), &ctx.handle, nullptr) !=
        QNN_SUCCESS) {
        LOGE("contextCreateFromBinary failed for %s", file);
        return false;
    }
    return true;
}

bool run_graph(Context& ctx, const char* graph_name, std::vector<Input>& inputs, std::vector<Output>& outputs, double* ms) {
    const auto* q = fn();
    Qnn_GraphHandle_t graph = nullptr;
    if (q->graphRetrieve(ctx.handle, graph_name, &graph) != QNN_SUCCESS || !graph) {
        LOGE("graphRetrieve(%s) failed", graph_name);
        return false;
    }
    std::vector<Qnn_Tensor_t> tin, tout;
    for (auto& i : inputs) tin.push_back(tensor(i.id, i.name, i.dims, QNN_TENSOR_TYPE_APP_WRITE, i.data, i.bytes));
    for (auto& o : outputs) tout.push_back(tensor(o.id, o.name, o.dims, QNN_TENSOR_TYPE_APP_READ, o.data, o.bytes));
    auto t0 = std::chrono::steady_clock::now();
    Qnn_ErrorHandle_t err = q->graphExecute(graph, tin.data(), static_cast<uint32_t>(tin.size()), tout.data(),
                                            static_cast<uint32_t>(tout.size()), nullptr, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    if (err != QNN_SUCCESS) {
        LOGE("graphExecute(%s) failed: %llu", graph_name, static_cast<unsigned long long>(err));
        return false;
    }
    if (ms) *ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

}  // namespace

// Encoder: image (float32 NHWC 1024x1024x3, 0..1) -> embeddings (float32 64x64x256).
// Decoder: embeddings + coords [1,1,2] + labels [1,1] -> masks [1,256,256,1] + scores [1,1].
//
// `points` is n x 2 (x, y in the encoder's 1024 frame); `labels` is n. Each point produces one
// 256x256 mask (row-major, float logits) and one score. The encoder runs once.
extern "C" int mobilesam_run_points(const char* lib_dir, const char* dir, const float* image, size_t image_floats,
                                    const float* points, size_t num_points, const float* labels, float* masks, float* scores,
                                    long* encoder_ms, long* decoder_ms) {
    if (!init_engine(lib_dir)) {
        LOGE("HTP init failed");
        return -1;
    }
    const size_t embedding_floats = 64 * 64 * 256;
    if (image_floats != 1024 * 1024 * 3 || num_points == 0) {
        LOGE("buffer size mismatch");
        return -2;
    }
    Context enc, dec;
    if (!open_context(dir, "encoder_ctx.bin", enc)) return -3;
    if (!open_context(dir, "decoder_ctx.bin", dec)) return -4;

    std::vector<float> embeddings(embedding_floats);
    {
        std::vector<Input> ins = {{1, "image", {1, 1024, 1024, 3}, const_cast<float*>(image), image_floats * sizeof(float)}};
        std::vector<Output> outs = {{881, "image_embeddings", {1, 64, 64, 256}, embeddings.data(), embedding_floats * sizeof(float)}};
        double ms = 0;
        if (!run_graph(enc, "graph_xqc9gm1i", ins, outs, &ms)) return -5;
        LOGI("encoder %.1f ms", ms);
        if (encoder_ms) *encoder_ms = static_cast<long>(ms);
    }
    long total_decoder_ms = 0;
    for (size_t p = 0; p < num_points; ++p) {
        float coords[2] = {points[2 * p], points[2 * p + 1]};
        float label = labels[p];
        std::vector<Input> ins = {
            {1, "point_coords", {1, 1, 2}, coords, sizeof(coords)},
            {12, "point_labels", {1, 1}, &label, sizeof(label)},
            {51, "image_embeddings", {1, 64, 64, 256}, embeddings.data(), embedding_floats * sizeof(float)},
        };
        std::vector<Output> outs = {
            {1951, "masks", {1, 256, 256, 1}, masks + p * 256 * 256, 256 * 256 * sizeof(float)},
            {1954, "scores", {1, 1}, scores + p, sizeof(float)},
        };
        double ms = 0;
        if (!run_graph(dec, "graph_l76ln4dr", ins, outs, &ms)) return -6;
        total_decoder_ms += static_cast<long>(ms);
        LOGI("decoder point %zu: %.1f ms", p, ms);
    }
    if (decoder_ms) *decoder_ms = total_decoder_ms;
    return 0;
}

extern "C" int mobilesam_run(const char* lib_dir, const char* dir, const float* image, size_t image_floats, const float* coords,
                             size_t coords_floats, const float* labels, size_t labels_floats, float* masks, size_t masks_floats,
                             float* scores, size_t scores_floats, long* encoder_ms, long* decoder_ms) {
    if (coords_floats != 2 || labels_floats != 1 || masks_floats != 256 * 256 || scores_floats != 1) {
        LOGE("buffer size mismatch");
        return -2;
    }
    return mobilesam_run_points(lib_dir, dir, image, image_floats, coords, 1, labels, masks, scores, encoder_ms, decoder_ms);
}
