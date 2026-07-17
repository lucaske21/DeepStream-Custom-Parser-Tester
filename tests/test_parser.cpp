/**
 * test_parser.cpp
 *
 * Unit tests for NvDsInferParseYolo26InstanceMask using synthetic tensors.
 *
 * Compile standalone (no ONNX Runtime needed):
 *   g++ -std=c++11 -I../include -I../parser \
 *       test_parser.cpp ../parser/yolo26_parser.cpp -o test_parser
 *
 * Run:
 *   ./test_parser
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cassert>
#include <string>

#include "fake_nvdsinfer.h"
#include "yolo26_parser.h"

/* -------------------------------------------------------------------------- */
/* Tiny test framework                                                         */
/* -------------------------------------------------------------------------- */

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << (msg) << "\n"; \
            ++g_failed; \
        } else { \
            std::cout << "  [PASS] " << (msg) << "\n"; \
            ++g_passed; \
        } \
    } while (0)

static float approxEq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

/* -------------------------------------------------------------------------- */
/* Helper: build a minimal NvDsInferLayerInfo from a float buffer.            */
/* -------------------------------------------------------------------------- */
static NvDsInferLayerInfo makeLayer(const char* name, void* buf,
                                    unsigned int d0, unsigned int d1,
                                    unsigned int d2 = 0)
{
    NvDsInferLayerInfo l;
    l.dataType     = FLOAT;
    l.isInput      = false;
    l.bindingIndex = 0;
    l.layerName    = name;
    l.buffer       = buf;

    if (d2 == 0) {
        l.inferDims.numDims    = 2;
        l.inferDims.d[0]       = d0;
        l.inferDims.d[1]       = d1;
        l.inferDims.numElements = d0 * d1;
    } else {
        l.inferDims.numDims    = 3;
        l.inferDims.d[0]       = d0;
        l.inferDims.d[1]       = d1;
        l.inferDims.d[2]       = d2;
        l.inferDims.numElements = d0 * d1 * d2;
    }
    return l;
}

/* -------------------------------------------------------------------------- */
/* Test 1: basic detection above threshold                                     */
/* -------------------------------------------------------------------------- */
static void test_basic_detection() {
    std::cout << "\n[test_basic_detection]\n";

    /* output0: [1, 300, 38] → dims without batch = [300, 38] */
    const int NUM_DETS = 300;
    const int COLS     = 38;
    std::vector<float> det0(NUM_DETS * COLS, 0.0f);

    /* Place one valid detection at row 0. */
    det0[0 * COLS + 0] = 50.0f;   /* x1 */
    det0[0 * COLS + 1] = 60.0f;   /* y1 */
    det0[0 * COLS + 2] = 250.0f;  /* x2 */
    det0[0 * COLS + 3] = 300.0f;  /* y2 */
    det0[0 * COLS + 4] = 0.91f;   /* confidence */
    det0[0 * COLS + 5] = 1.0f;    /* class_id */
    /* cols 6-37: mask coefficients – leave as 0 */

    /* output1: [1, 32, 160, 160] → dims without batch = [32, 160, 160] */
    const int NUM_PROTO = 32;
    const int PH = 160, PW = 160;
    std::vector<float> proto(NUM_PROTO * PH * PW, 0.0f);

    std::vector<NvDsInferLayerInfo> layers;
    layers.push_back(makeLayer("output0", det0.data(), NUM_DETS, COLS));
    layers.push_back(makeLayer("output1", proto.data(), NUM_PROTO, PH, PW));

    NvDsInferNetworkInfo netInfo;
    netInfo.width    = 640;
    netInfo.height   = 640;
    netInfo.channels = 3;

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 2;
    params.perClassPreclusterThreshold  = {0.25f, 0.25f};
    params.perClassPostclusterThreshold = {0.25f, 0.25f};

    std::vector<NvDsInferInstanceMaskInfo> objs;
    bool ok = NvDsInferParseYolo26InstanceMask(layers, netInfo, params, objs);

    CHECK(ok,                         "parser returned true");
    CHECK(objs.size() == 1,           "exactly 1 detection");

    if (!objs.empty()) {
        CHECK(approxEq(objs[0].left,   50.0f),  "bbox x1");
        CHECK(approxEq(objs[0].top,    60.0f),  "bbox y1");
        CHECK(approxEq(objs[0].width,  200.0f), "bbox w");
        CHECK(approxEq(objs[0].height, 240.0f), "bbox h");
        CHECK(approxEq(objs[0].detectionConfidence, 0.91f), "confidence");
        CHECK(objs[0].classId == 1,              "class id");
        CHECK(objs[0].mask != nullptr,           "mask buffer not null");
        CHECK(objs[0].mask_width  == PW,         "mask width");
        CHECK(objs[0].mask_height == PH,         "mask height");
    }

    /* Clean up parser-allocated mask buffers. */
    for (auto& o : objs) { delete[] o.mask; o.mask = nullptr; }
}

/* -------------------------------------------------------------------------- */
/* Test 2: detection below threshold is filtered out                           */
/* -------------------------------------------------------------------------- */
static void test_below_threshold() {
    std::cout << "\n[test_below_threshold]\n";

    const int NUM_DETS = 10, COLS = 38;
    std::vector<float> det0(NUM_DETS * COLS, 0.0f);
    det0[0 * COLS + 0] = 10.0f;
    det0[0 * COLS + 1] = 10.0f;
    det0[0 * COLS + 2] = 100.0f;
    det0[0 * COLS + 3] = 100.0f;
    det0[0 * COLS + 4] = 0.10f;   /* below 0.25 */
    det0[0 * COLS + 5] = 0.0f;

    std::vector<float> proto(32 * 160 * 160, 0.0f);

    std::vector<NvDsInferLayerInfo> layers;
    layers.push_back(makeLayer("output0", det0.data(), NUM_DETS, COLS));
    layers.push_back(makeLayer("output1", proto.data(), 32, 160, 160));

    NvDsInferNetworkInfo netInfo = {640, 640, 3};

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold  = {0.25f};
    params.perClassPostclusterThreshold = {0.25f};

    std::vector<NvDsInferInstanceMaskInfo> objs;
    bool ok = NvDsInferParseYolo26InstanceMask(layers, netInfo, params, objs);

    CHECK(ok,               "parser returned true");
    CHECK(objs.empty(),     "0 detections (filtered by threshold)");

    for (auto& o : objs) { delete[] o.mask; o.mask = nullptr; }
}

/* -------------------------------------------------------------------------- */
/* Test 3: missing output layer → parser returns false                         */
/* -------------------------------------------------------------------------- */
static void test_missing_output() {
    std::cout << "\n[test_missing_output]\n";

    std::vector<float> det0(300 * 38, 0.0f);
    std::vector<NvDsInferLayerInfo> layers;
    /* Only output0, no output1. */
    layers.push_back(makeLayer("output0", det0.data(), 300, 38));

    NvDsInferNetworkInfo netInfo = {640, 640, 3};
    NvDsInferParseDetectionParams params;

    std::vector<NvDsInferInstanceMaskInfo> objs;
    bool ok = NvDsInferParseYolo26InstanceMask(layers, netInfo, params, objs);

    CHECK(!ok, "parser returns false when output1 is absent");

    for (auto& o : objs) { delete[] o.mask; o.mask = nullptr; }
}

/* -------------------------------------------------------------------------- */
/* Test 4: degenerate bbox (x2 <= x1) is ignored                              */
/* -------------------------------------------------------------------------- */
static void test_degenerate_bbox() {
    std::cout << "\n[test_degenerate_bbox]\n";

    const int NUM_DETS = 5, COLS = 38;
    std::vector<float> det0(NUM_DETS * COLS, 0.0f);

    /* det 0: degenerate bbox */
    det0[0 * COLS + 0] = 200.0f;  /* x1 */
    det0[0 * COLS + 1] =  50.0f;  /* y1 */
    det0[0 * COLS + 2] = 100.0f;  /* x2 < x1 → degenerate */
    det0[0 * COLS + 3] = 150.0f;  /* y2 */
    det0[0 * COLS + 4] = 0.95f;
    det0[0 * COLS + 5] = 0.0f;

    /* det 1: valid */
    det0[1 * COLS + 0] = 10.0f;
    det0[1 * COLS + 1] = 20.0f;
    det0[1 * COLS + 2] = 110.0f;
    det0[1 * COLS + 3] = 120.0f;
    det0[1 * COLS + 4] = 0.80f;
    det0[1 * COLS + 5] = 0.0f;

    std::vector<float> proto(32 * 160 * 160, 0.0f);

    std::vector<NvDsInferLayerInfo> layers;
    layers.push_back(makeLayer("output0", det0.data(), NUM_DETS, COLS));
    layers.push_back(makeLayer("output1", proto.data(), 32, 160, 160));

    NvDsInferNetworkInfo netInfo = {640, 640, 3};
    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold  = {0.25f};
    params.perClassPostclusterThreshold = {0.25f};

    std::vector<NvDsInferInstanceMaskInfo> objs;
    bool ok = NvDsInferParseYolo26InstanceMask(layers, netInfo, params, objs);

    CHECK(ok,               "parser returned true");
    CHECK(objs.size() == 1, "only the valid bbox is kept (degenerate filtered)");

    for (auto& o : objs) { delete[] o.mask; o.mask = nullptr; }
}

/* -------------------------------------------------------------------------- */
/* Test 5: mask sigmoid values are in (0, 1)                                  */
/* -------------------------------------------------------------------------- */
static void test_mask_values() {
    std::cout << "\n[test_mask_values]\n";

    const int NUM_DETS = 1, COLS = 38;
    std::vector<float> det0(NUM_DETS * COLS, 0.0f);
    det0[0] = 0.0f; det0[1] = 0.0f; det0[2] = 320.0f; det0[3] = 320.0f;
    det0[4] = 0.99f; det0[5] = 0.0f;
    /* Put non-zero coefficients to get non-trivial mask values. */
    for (int k = 6; k < 38; ++k) det0[k] = (k % 3 == 0) ? 1.0f : -0.5f;

    std::vector<float> proto(32 * 8 * 8, 0.5f);  /* use small proto for speed */

    std::vector<NvDsInferLayerInfo> layers;
    layers.push_back(makeLayer("output0", det0.data(), NUM_DETS, COLS));
    layers.push_back(makeLayer("output1", proto.data(), 32, 8, 8));

    NvDsInferNetworkInfo netInfo = {640, 640, 3};
    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold  = {0.25f};
    params.perClassPostclusterThreshold = {0.25f};

    std::vector<NvDsInferInstanceMaskInfo> objs;
    bool ok = NvDsInferParseYolo26InstanceMask(layers, netInfo, params, objs);

    CHECK(ok && !objs.empty(), "got detection");
    if (!objs.empty()) {
        bool allInRange = true;
        for (int i = 0; i < objs[0].mask_width * objs[0].mask_height; ++i) {
            const float v = objs[0].mask[i];
            if (v <= 0.0f || v >= 1.0f) { allInRange = false; break; }
        }
        CHECK(allInRange, "all mask values in (0,1) – sigmoid applied");
    }

    for (auto& o : objs) { delete[] o.mask; o.mask = nullptr; }
}

/* -------------------------------------------------------------------------- */
/* Test 6: name-based layer lookup falls back to index                        */
/* -------------------------------------------------------------------------- */
static void test_fallback_layer_name() {
    std::cout << "\n[test_fallback_layer_name]\n";

    const int NUM_DETS = 5, COLS = 38;
    std::vector<float> det0(NUM_DETS * COLS, 0.0f);
    det0[0 * COLS + 2] = 100.0f;
    det0[0 * COLS + 3] = 100.0f;
    det0[0 * COLS + 4] = 0.80f;

    std::vector<float> proto(32 * 160 * 160, 0.0f);

    std::vector<NvDsInferLayerInfo> layers;
    /* Use non-standard names – parser should fall back to index 0 and 1. */
    layers.push_back(makeLayer("pred_boxes", det0.data(), NUM_DETS, COLS));
    layers.push_back(makeLayer("proto_masks", proto.data(), 32, 160, 160));

    NvDsInferNetworkInfo netInfo = {640, 640, 3};
    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold  = {0.25f};
    params.perClassPostclusterThreshold = {0.25f};

    std::vector<NvDsInferInstanceMaskInfo> objs;
    bool ok = NvDsInferParseYolo26InstanceMask(layers, netInfo, params, objs);

    CHECK(ok, "parser returns true with fallback layer names");

    for (auto& o : objs) { delete[] o.mask; o.mask = nullptr; }
}

/* -------------------------------------------------------------------------- */
/* main                                                                        */
/* -------------------------------------------------------------------------- */
int main()
{
    std::cout << "=== yolo26_parser unit tests ===\n";

    test_basic_detection();
    test_below_threshold();
    test_missing_output();
    test_degenerate_bbox();
    test_mask_values();
    test_fallback_layer_name();

    std::cout << "\n=============================\n";
    std::cout << "Passed: " << g_passed << "\n";
    std::cout << "Failed: " << g_failed << "\n";

    return (g_failed == 0) ? 0 : 1;
}
