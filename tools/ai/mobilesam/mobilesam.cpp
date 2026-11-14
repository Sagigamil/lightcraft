// MobileSAM on the Hexagon HTP (encoder) and the QNN CPU backend (decoder).
//
// The encoder runs in 2.5 s on the HTP and matches the CPU backend; the tiny prompt decoder is
// numerically unreliable on the HTP (masks differ from the CPU, sometimes empty), so it runs on
// the CPU backend in ~54 ms. Both contexts are produced from Qualcomm's `mobilesam-qnn_dlc-float`
// DLCs (the decoder re-exported with 8 prompt slots).
//
// Encoder: image (float32 NHWC 1024x1024x3, 0..1) -> image_embeddings (float32 64x64x256).
// Decoder: embeddings + point_coords [1,8,2] + point_labels [1,8] -> masks [1,256,256,1] + scores [1,1].
// Unused prompt slots are padded with label -1.

#include <dlfcn.h>
#include <jni.h>
#include <android/log.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
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

constexpr size_t kPromptSlots = 8;
constexpr size_t kEmbeddingFloats = 64 * 64 * 256;
constexpr size_t kMaskFloats = 256 * 256;

struct Engine {
    void* lib = nullptr;
    const QnnInterface_t* iface = nullptr;
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
};

Engine g_encoder;
Engine g_decoder;
using QnnVTable = QNN_INTERFACE_VER_TYPE;
using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

const QnnVTable* fn(const Engine& e) {
    return e.iface ? &e.iface->QNN_INTERFACE_VER_NAME : nullptr;
}

bool init_engine(Engine& e, const char* lib_dir, const char* lib_name, bool htp) {
    if (e.backend) return true;
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
    if (htp) {
        // The app stages the QNN libs (and the skel) in its native library directory and cannot
        // set ADSP_LIBRARY_PATH at exec time; the FastRPC client also searches "." for the skel.
        if (!getenv("ADSP_LIBRARY_PATH")) {
            std::string path;
            if (!dir.empty() && dir[0] == '/') path = dir + ":";
            path += "/vendor/lib/rfsa/adsp:/vendor/lib64";
            setenv("ADSP_LIBRARY_PATH", path.c_str(), 1);
        }
        if (!dir.empty() && dir[0] == '/') {
            std::string skel = dir + "/libQnnHtpV81Skel.so";
            if (access(skel.c_str(), R_OK) == 0 && chdir(dir.c_str()) == 0) LOGI("using skel from %s", dir.c_str());
        }
        open_lib("libQnnHtpV81Stub.so", RTLD_NOW | RTLD_LOCAL);
    }
    e.lib = open_lib(lib_name, RTLD_NOW | RTLD_GLOBAL);
    if (!e.lib) return false;
    auto get_providers = reinterpret_cast<GetProvidersFn>(dlsym(e.lib, "QnnInterface_getProviders"));
    if (!get_providers) return false;
    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    if (get_providers(&providers, &count) != QNN_SUCCESS || !providers || count == 0) return false;
    e.iface = providers[0];
    const auto* q = fn(e);
    if (q->backendCreate(nullptr, nullptr, &e.backend) != QNN_SUCCESS || !e.backend) return false;
    // The CPU backend has no device: QnnDevice_create is optional/unsupported there.
    if (q->deviceCreate(nullptr, nullptr, &e.device) != QNN_SUCCESS || !e.device) {
        if (htp) return false;
        LOGI("%s has no device (CPU backend)", lib_name);
        e.device = nullptr;
    }
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

Qnn_Tensor_t tensor(uint32_t id, const char* name, const std::vector<uint32_t>& dims, Qnn_TensorType_t type, void* data,
                    size_t bytes) {
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

bool open_context(Engine& e, const std::string& dir, const char* file, Context& ctx) {
    const auto* q = fn(e);
    std::vector<uint8_t> binary = read_file(dir + "/" + file);
    if (binary.empty()) {
        LOGE("cannot read %s/%s", dir.c_str(), file);
        return false;
    }
    if (q->contextCreateFromBinary(e.backend, e.device, nullptr, binary.data(), binary.size(), &ctx.handle, nullptr) !=
        QNN_SUCCESS) {
        LOGE("contextCreateFromBinary failed for %s", file);
        return false;
    }
    return true;
}

bool run_graph(Engine& engine, Context& ctx, const char* graph_name, std::vector<Input>& inputs, std::vector<Output>& outputs,
               double* ms) {
    const auto* q = fn(engine);
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

// Encodes one image and leaves the embeddings for the decoder calls.
bool encode(Engine& enc_engine, Context& enc_ctx, const float* image, size_t image_floats, std::vector<float>& embeddings,
            long* encoder_ms) {
    embeddings.assign(kEmbeddingFloats, 0.0f);
    std::vector<Input> ins = {{1, "image", {1, 1024, 1024, 3}, const_cast<float*>(image), image_floats * sizeof(float)}};
    std::vector<Output> outs = {{881, "image_embeddings", {1, 64, 64, 256}, embeddings.data(), kEmbeddingFloats * sizeof(float)}};
    double ms = 0;
    if (!run_graph(enc_engine, enc_ctx, "graph_xqc9gm1i", ins, outs, &ms)) return false;
    LOGI("encoder %.1f ms", ms);
    if (encoder_ms) *encoder_ms = static_cast<long>(ms);
    return true;
}

// Runs the prompt decoder with exactly one prompt set (up to 8 slots, padded with label -1).
bool decode(Engine& dec_engine, Context& dec_ctx, const std::vector<float>& embeddings, const float* coords, const float* labels,
            size_t num_points, float* masks, float* scores, long* decoder_ms) {
    if (num_points == 0 || num_points > kPromptSlots) return false;
    std::vector<float> c(kPromptSlots * 2, 0.0f);
    std::vector<float> l(kPromptSlots, -1.0f);
    for (size_t i = 0; i < num_points; ++i) {
        c[2 * i] = coords[2 * i];
        c[2 * i + 1] = coords[2 * i + 1];
        l[i] = labels[i];
    }
    std::vector<Input> ins = {
        {1, "image_embeddings", {1, 64, 64, 256}, const_cast<float*>(embeddings.data()), kEmbeddingFloats * sizeof(float)},
        {4, "point_coords", {1, 8, 2}, c.data(), c.size() * sizeof(float)},
        {22, "point_labels", {1, 8}, l.data(), l.size() * sizeof(float)},
    };
    std::vector<Output> outs = {
        {486, "scores", {1, 1}, scores, sizeof(float)},
        {487, "masks", {1, 256, 256, 1}, masks, kMaskFloats * sizeof(float)},
    };
    double ms = 0;
    if (!run_graph(dec_engine, dec_ctx, "decoder8", ins, outs, &ms)) return false;
    LOGI("decoder %.1f ms", ms);
    if (decoder_ms) *decoder_ms = static_cast<long>(ms);
    return true;
}

}  // namespace

namespace {

/// The app keeps one session: both contexts stay loaded, and the embeddings of the last encoded
/// photo live here between clicks.
struct Session {
    bool ready = false;
    Context enc_ctx;
    Context dec_ctx;
    std::vector<float> embeddings;
};

Session g_session;

bool ensure_session(const char* lib_dir, const char* dir) {
    if (g_session.ready) return true;
    if (!init_engine(g_encoder, lib_dir, "libQnnHtp.so", true)) {
        LOGE("HTP init failed");
        return false;
    }
    if (!init_engine(g_decoder, lib_dir, "libQnnCpu.so", false)) {
        LOGE("CPU init failed");
        return false;
    }
    if (!open_context(g_encoder, dir, "encoder_ctx.bin", g_session.enc_ctx)) return false;
    if (!open_context(g_decoder, dir, "decoder8_cpu.bin", g_session.dec_ctx)) return false;
    g_session.embeddings.assign(kEmbeddingFloats, 0.0f);
    g_session.ready = true;
    LOGI("MobileSAM ready (HTP encoder + CPU decoder)");
    return true;
}

std::string jstring_to_utf8(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    if (!c) return {};
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

bool read_f32_file(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto n = static_cast<size_t>(f.tellg());
    f.seekg(0);
    out.resize(n / sizeof(float));
    return static_cast<bool>(f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n)));
}

bool write_f32_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(f);
}

}  // namespace

// One prompt with up to 8 positive/negative points -> one mask and one score.
extern "C" int mobilesam_run_prompt(const char* lib_dir, const char* dir, const float* image, size_t image_floats,
                                    const float* coords, const float* labels, size_t num_points, float* masks, float* scores,
                                    long* encoder_ms, long* decoder_ms) {
    if (!ensure_session(lib_dir, dir)) return -1;
    if (image_floats != 1024 * 1024 * 3 || num_points == 0 || num_points > kPromptSlots) return -2;
    if (!encode(g_encoder, g_session.enc_ctx, image, image_floats, g_session.embeddings, encoder_ms)) return -5;
    if (!decode(g_decoder, g_session.dec_ctx, g_session.embeddings, coords, labels, num_points, masks, scores, decoder_ms)) return -6;
    return 0;
}

// Test helper: one prompt per point (each with a single active slot), one encode for all.
extern "C" int mobilesam_run_points(const char* lib_dir, const char* dir, const float* image, size_t image_floats,
                                    const float* points, size_t num_points, const float* labels, float* masks, float* scores,
                                    long* encoder_ms, long* decoder_ms) {
    if (!ensure_session(lib_dir, dir)) return -1;
    if (image_floats != 1024 * 1024 * 3 || num_points == 0) return -2;
    if (!encode(g_encoder, g_session.enc_ctx, image, image_floats, g_session.embeddings, encoder_ms)) return -5;
    long total = 0;
    for (size_t p = 0; p < num_points; ++p) {
        float label = labels[p];
        long ms = 0;
        if (!decode(g_decoder, g_session.dec_ctx, g_session.embeddings, &points[2 * p], &label, 1, masks + p * kMaskFloats,
                    scores + p, &ms)) {
            return -6;
        }
        total += ms;
    }
    if (decoder_ms) *decoder_ms = total;
    return 0;
}

// JNI: encode the image file and write the embeddings file. Returns ms, or a negative error.
extern "C" JNIEXPORT jlong JNICALL Java_com_lightcraft_android_MobileSam_encode(JNIEnv* env, jclass, jstring jdir, jstring jlib,
                                                                                jstring jinput, jstring joutput) {
    const std::string dir = jstring_to_utf8(env, jdir);
    const std::string lib = jstring_to_utf8(env, jlib);
    const std::string input = jstring_to_utf8(env, jinput);
    const std::string output = jstring_to_utf8(env, joutput);
    if (dir.empty() || lib.empty() || input.empty() || output.empty()) return -10;
    std::vector<float> image;
    if (!read_f32_file(input, image) || image.size() != 1024 * 1024 * 3) {
        LOGE("cannot read the image tensor %s", input.c_str());
        return -11;
    }
    if (!ensure_session(lib.c_str(), dir.c_str())) return -12;
    long ms = 0;
    if (!encode(g_encoder, g_session.enc_ctx, image.data(), image.size(), g_session.embeddings, &ms)) return -13;
    if (!write_f32_file(output, g_session.embeddings.data(), g_session.embeddings.size() * sizeof(float))) return -14;
    return static_cast<jlong>(ms);
}

// JNI: decode with prompts. Reads the embeddings file, writes the mask and score files. Returns ms.
extern "C" JNIEXPORT jlong JNICALL Java_com_lightcraft_android_MobileSam_decode(
    JNIEnv* env, jclass, jstring jdir, jstring jlib, jstring jembeddings, jstring jpoints, jstring jlabels, jstring jmask,
    jstring jscore) {
    const std::string dir = jstring_to_utf8(env, jdir);
    const std::string lib = jstring_to_utf8(env, jlib);
    const std::string embeddings_path = jstring_to_utf8(env, jembeddings);
    const std::string points_path = jstring_to_utf8(env, jpoints);
    const std::string labels_path = jstring_to_utf8(env, jlabels);
    const std::string mask_path = jstring_to_utf8(env, jmask);
    const std::string score_path = jstring_to_utf8(env, jscore);
    if (dir.empty() || lib.empty() || embeddings_path.empty() || points_path.empty() || labels_path.empty()) return -20;
    std::vector<float> embeddings, points, labels;
    if (!read_f32_file(embeddings_path, embeddings) || embeddings.size() != kEmbeddingFloats) return -21;
    if (!read_f32_file(points_path, points) || points.empty() || points.size() % 2 != 0) return -22;
    if (!read_f32_file(labels_path, labels) || labels.size() != points.size() / 2) return -23;
    if (!ensure_session(lib.c_str(), dir.c_str())) return -24;
    g_session.embeddings = std::move(embeddings);
    std::vector<float> masks(kMaskFloats), scores(1);
    long ms = 0;
    if (!decode(g_decoder, g_session.dec_ctx, g_session.embeddings, points.data(), labels.data(), labels.size(), masks.data(),
                scores.data(), &ms)) {
        return -25;
    }
    if (!write_f32_file(mask_path, masks.data(), masks.size() * sizeof(float))) return -26;
    if (!write_f32_file(score_path, scores.data(), sizeof(float))) return -27;
    return static_cast<jlong>(ms);
}
