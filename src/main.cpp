/**
 * main.cpp
 *
 * DeepStream Custom Parser Tester — standalone CLI tool.
 *
 * Usage
 * -----
 *   parser_tester --image <img.jpg> --model <model.onnx> [options]
 *
 * Options
 *   --image <path>            Input image (required).
 *   --model <path>            ONNX model file (required).
 *   --parser-so <path>        Parser shared library (.so) (required).
 *   --parser-func <name>      C-linkage function name inside .so (required).
 *   --output <path>           Save annotated result image (default: output/result.jpg).
 *   --conf <float>            Confidence threshold (default: 0.25).
 *   --classes <names...>      Class names, space-separated (default: "object").
 *   --width <int>             Network input width  (default: 640).
 *   --height <int>            Network input height (default: 640).
 *   --dump-tensor             Save raw output tensors to output/output*.bin.
 *   --dump-mask               Save intermediate mask images to output/.
 *   --no-display              Skip cv::imshow() even when DISPLAY is available.
 *   --compare <expected.json> Run golden regression test.
 *
 * Data flow
 * ---------
 *   Image → letterbox preprocess → OrtRunner → TensorInfo[]
 *       → TensorAdapter → NvDsInferLayerInfo[]
 *       → parser .so (loaded via dlopen/dlsym)
 *       → NvDsInferInstanceMaskInfo[]
 *       → Visualizer → annotated image + JSON results
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <dlfcn.h>

#include <nlohmann/json.hpp>

#include "fake_nvdsinfer.h"
#include "ort_runner.h"
#include "tensor_adapter.h"
#include "visualizer.h"

using json = nlohmann::json;

/* -------------------------------------------------------------------------- */
/* Parser function type                                                        */
/* -------------------------------------------------------------------------- */

/**
 * Function pointer type matching the DeepStream custom-parser C ABI.
 * The .so must export the function with extern "C" linkage.
 */
typedef bool (*NvDsParserFunc)(
    const std::vector<NvDsInferLayerInfo>&,
    const NvDsInferNetworkInfo&,
    const NvDsInferParseDetectionParams&,
    std::vector<NvDsInferInstanceMaskInfo>&);

/* -------------------------------------------------------------------------- */
/* CLI helpers                                                                 */
/* -------------------------------------------------------------------------- */

static void printUsage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " --image <img.jpg> --model <model.onnx> [options]\n\n"
        "  --image       <path>        Input image (required)\n"
        "  --model       <path>        ONNX model (required)\n"
        "  --parser-so   <path>        Parser .so path (required)\n"
        "  --parser-func <name>        Parser function name (required)\n"
        "  --output      <path>        Annotated output image (default: output/result.jpg)\n"
        "  --conf        <float>       Confidence threshold (default: 0.25)\n"
        "  --classes     <n0 n1 ...>   Class names (default: \"object\")\n"
        "  --width       <int>         Network input width  (default: 640)\n"
        "  --height      <int>         Network input height (default: 640)\n"
        "  --dump-tensor               Save raw output tensors\n"
        "  --dump-mask                 Save intermediate mask images\n"
        "  --no-display                Skip imshow()\n"
        "  --compare     <expected.json>  Golden regression test\n";
}

static bool getArg(const std::vector<std::string>& args,
                   const std::string& flag,
                   std::string& value)
{
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag) { value = args[i + 1]; return true; }
    }
    return false;
}

static bool hasFlag(const std::vector<std::string>& args, const std::string& flag) {
    for (const auto& a : args) { if (a == flag) return true; }
    return false;
}

/* Known CLI flags – used to detect the end of a --classes list. */
static const char* const KNOWN_FLAGS[] = {
    "--image", "--model", "--output", "--conf",
    "--classes", "--width", "--height",
    "--dump-tensor", "--dump-mask", "--no-display",
    "--compare", "--parser-so", "--parser-func",
    "--help", "-h", nullptr
};

static bool isKnownFlag(const std::string& s) {
    for (int i = 0; KNOWN_FLAGS[i] != nullptr; ++i) {
        if (s == KNOWN_FLAGS[i]) return true;
    }
    return false;
}

/* Collect all tokens after --classes until the next known flag. */
static std::vector<std::string> getClassNames(const std::vector<std::string>& args) {
    std::vector<std::string> names;
    bool collecting = false;
    for (const auto& a : args) {
        if (a == "--classes") { collecting = true; continue; }
        if (collecting) {
            if (isKnownFlag(a)) break;
            names.push_back(a);
        }
    }
    return names;
}

/* -------------------------------------------------------------------------- */
/* Tensor dump                                                                 */
/* -------------------------------------------------------------------------- */

static void dumpTensors(const std::vector<TensorInfo>& tensors,
                        const std::string& outDir)
{
    for (const auto& ti : tensors) {
        std::string path = outDir + "/" + ti.name + ".bin";
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "[dump] Cannot open " << path << "\n"; continue; }
        f.write(reinterpret_cast<const char*>(ti.data.data()),
                static_cast<std::streamsize>(ti.data.size() * sizeof(float)));
        std::cout << "[dump] Tensor '" << ti.name
                  << "' -> " << path
                  << "  (" << ti.data.size() << " floats)\n";
    }
}

/* -------------------------------------------------------------------------- */
/* JSON export                                                                 */
/* -------------------------------------------------------------------------- */

static json detectionsToJson(
    const std::vector<NvDsInferInstanceMaskInfo>& objs)
{
    json arr = json::array();
    for (const auto& o : objs) {
        json det;
        det["class_id"]   = o.classId;
        det["confidence"] = static_cast<double>(o.detectionConfidence);
        det["bbox"]       = { o.left, o.top, o.width, o.height };
        det["mask_width"]  = o.mask_width;
        det["mask_height"] = o.mask_height;
        arr.push_back(det);
    }
    return arr;
}

/* -------------------------------------------------------------------------- */
/* Golden regression test                                                      */
/* -------------------------------------------------------------------------- */

static float bboxIoU(float ax1, float ay1, float aw, float ah,
                     float bx1, float by1, float bw, float bh)
{
    const float ax2 = ax1 + aw, ay2 = ay1 + ah;
    const float bx2 = bx1 + bw, by2 = by1 + bh;

    const float ix1 = std::max(ax1, bx1);
    const float iy1 = std::max(ay1, by1);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);

    const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    const float aArea = aw * ah;
    const float bArea = bw * bh;
    const float uni   = aArea + bArea - inter;
    return (uni > 0.0f) ? inter / uni : 0.0f;
}

static bool runGoldenTest(
    const std::string& expectedPath,
    const std::vector<NvDsInferInstanceMaskInfo>& actual,
    float iouThresh    = 0.5f,
    float confMaxDiff  = 0.1f)
{
    std::ifstream f(expectedPath);
    if (!f) {
        std::cerr << "[golden] Cannot open " << expectedPath << "\n";
        return false;
    }
    json expected;
    try { f >> expected; }
    catch (const json::exception& e) {
        std::cerr << "[golden] JSON parse error: " << e.what() << "\n";
        return false;
    }

    bool allPassed = true;
    std::cout << "\n=== Golden Regression Test ===\n";
    std::cout << "Expected: " << expected.size()
              << "  Actual: " << actual.size() << "\n";

    /* For each expected detection, find the best-matching actual detection. */
    std::vector<bool> matched(actual.size(), false);

    for (size_t ei = 0; ei < expected.size(); ++ei) {
        const auto& exp = expected[ei];
        const int   expCls  = exp.value("class_id", -1);
        const float expConf = static_cast<float>(exp.value("confidence", 0.0));
        const auto& expBbox = exp["bbox"];
        const float ex = expBbox[0], ey = expBbox[1],
                    ew = expBbox[2], eh = expBbox[3];

        float bestIoU = 0.0f;
        int   bestIdx = -1;
        for (size_t ai = 0; ai < actual.size(); ++ai) {
            if (matched[ai]) continue;
            const auto& a = actual[ai];
            if (static_cast<int>(a.classId) != expCls) continue;
            const float iou = bboxIoU(ex, ey, ew, eh,
                                      a.left, a.top, a.width, a.height);
            if (iou > bestIoU) { bestIoU = iou; bestIdx = static_cast<int>(ai); }
        }

        bool pass = (bestIdx >= 0 && bestIoU >= iouThresh);
        bool confPass = true;
        if (pass) {
            matched[static_cast<size_t>(bestIdx)] = true;
            const float confDiff =
                std::fabs(actual[static_cast<size_t>(bestIdx)].detectionConfidence
                          - expConf);
            confPass = (confDiff <= confMaxDiff);
        }

        const bool rowPass = pass && confPass;
        allPassed = allPassed && rowPass;

        std::cout << "  [" << (rowPass ? "PASS" : "FAIL") << "]"
                  << " expected cls=" << expCls
                  << " conf=" << expConf
                  << " bbox=[" << ex<<","<<ey<<","<<ew<<","<<eh<<"]"
                  << " -> IoU=" << bestIoU
                  << (pass ? "" : " (no match)")
                  << ((!pass || confPass) ? "" : " (conf diff too large)")
                  << "\n";
    }

    /* Unmatched actual detections count as false positives. */
    for (size_t ai = 0; ai < actual.size(); ++ai) {
        if (!matched[ai]) {
            allPassed = false;
            std::cout << "  [FP]  actual cls=" << actual[ai].classId
                      << " conf=" << actual[ai].detectionConfidence << "\n";
        }
    }

    std::cout << "Result: " << (allPassed ? "PASSED" : "FAILED") << "\n\n";
    return allPassed;
}

/* -------------------------------------------------------------------------- */
/* Free parser-allocated mask buffers                                          */
/* -------------------------------------------------------------------------- */

static void freeMasks(std::vector<NvDsInferInstanceMaskInfo>& objs) {
    for (auto& o : objs) {
        delete[] o.mask;
        o.mask = nullptr;
    }
}

/* -------------------------------------------------------------------------- */
/* main                                                                        */
/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
    if (argc < 2) { printUsage(argv[0]); return 1; }

    const std::vector<std::string> args(argv + 1, argv + argc);

    if (hasFlag(args, "--help") || hasFlag(args, "-h")) {
        printUsage(argv[0]); return 0;
    }

    /* ------------------------------------------------------------------ */
    /* Parse CLI arguments.                                                 */
    /* ------------------------------------------------------------------ */
    std::string imagePath, modelPath, outputPath, comparePath;
    std::string confStr, widthStr, heightStr;
    std::string parserSoPath, parserFuncName;
    bool dumpTensor = hasFlag(args, "--dump-tensor");
    bool dumpMask   = hasFlag(args, "--dump-mask");
    bool noDisplay  = hasFlag(args, "--no-display");

    if (!getArg(args, "--image",   imagePath)  ||
        !getArg(args, "--model",   modelPath)) {
        std::cerr << "Error: --image and --model are required.\n";
        printUsage(argv[0]);
        return 1;
    }

    getArg(args, "--output",      outputPath);
    getArg(args, "--compare",     comparePath);
    getArg(args, "--conf",        confStr);
    getArg(args, "--width",       widthStr);
    getArg(args, "--height",      heightStr);
    getArg(args, "--parser-so",   parserSoPath);
    getArg(args, "--parser-func", parserFuncName);

    if (outputPath.empty()) outputPath = "output/result.jpg";

    /* Parse --conf with explicit error checking. */
    float confThresh = 0.25f;
    if (!confStr.empty()) {
        char* endPtr = nullptr;
        const float parsed = static_cast<float>(std::strtod(confStr.c_str(), &endPtr));
        if (endPtr == confStr.c_str() || *endPtr != '\0' || parsed < 0.0f || parsed > 1.0f) {
            std::cerr << "Error: --conf must be a float in [0,1], got: " << confStr << "\n";
            return 1;
        }
        confThresh = parsed;
    }

    /* Parse --width with explicit error checking. */
    int netW = 640;
    if (!widthStr.empty()) {
        char* endPtr = nullptr;
        const long parsed = std::strtol(widthStr.c_str(), &endPtr, 10);
        if (endPtr == widthStr.c_str() || *endPtr != '\0' || parsed <= 0) {
            std::cerr << "Error: --width must be a positive integer, got: " << widthStr << "\n";
            return 1;
        }
        netW = static_cast<int>(parsed);
    }

    /* Parse --height with explicit error checking. */
    int netH = 640;
    if (!heightStr.empty()) {
        char* endPtr = nullptr;
        const long parsed = std::strtol(heightStr.c_str(), &endPtr, 10);
        if (endPtr == heightStr.c_str() || *endPtr != '\0' || parsed <= 0) {
            std::cerr << "Error: --height must be a positive integer, got: " << heightStr << "\n";
            return 1;
        }
        netH = static_cast<int>(parsed);
    }

    std::vector<std::string> classNames = getClassNames(args);
    if (classNames.empty()) classNames.push_back("object");

    if (parserSoPath.empty() || parserFuncName.empty()) {
        std::cerr << "Error: --parser-so and --parser-func are required.\n";
        return 1;
    }

    std::cout << "\n[Config] Parser .so  : " << parserSoPath << "\n";
    std::cout << "[Config] Parser func : " << parserFuncName << "\n";

    /* ------------------------------------------------------------------ */
    /* Load parser .so via dlopen / dlsym.                                  */
    /* ------------------------------------------------------------------ */
    void* soHandle = dlopen(parserSoPath.c_str(), RTLD_NOW);
    if (!soHandle) {
        std::cerr << "Error: dlopen('" << parserSoPath
                  << "') failed: " << dlerror() << "\n";
        return 1;
    }

    /* Clear any previous error before calling dlsym. */
    dlerror();
    void* sym = dlsym(soHandle, parserFuncName.c_str());
    const char* dlErr = dlerror();
    if (dlErr) {
        std::cerr << "Error: dlsym('" << parserFuncName
                  << "') failed: " << dlErr << "\n";
        dlclose(soHandle);
        return 1;
    }
    /* POSIX specifies that dlsym() returns a pointer that can be cast to the
     * correct function type on this platform. */
    static_assert(sizeof(void*) == sizeof(NvDsParserFunc),
                  "dlsym() pointer size does not match NvDsParserFunc");
    NvDsParserFunc parserFunc = reinterpret_cast<NvDsParserFunc>(sym);

    const std::string outDir = "output";

    /* ------------------------------------------------------------------ */
    /* Step 1 – Preprocess image.                                           */
    /* ------------------------------------------------------------------ */
    std::cout << "\n[Step 1] Preprocessing image: " << imagePath << "\n";
    PreprocessInfo ppInfo;
    std::vector<float> inputData;
    cv::Mat origImage = preprocessImage(imagePath, netW, netH, ppInfo, inputData);
    if (origImage.empty()) { dlclose(soHandle); return 1; }

    std::cout << "  Original: " << origImage.cols << "x" << origImage.rows << "\n";
    std::cout << "  Scale=" << ppInfo.scale
              << "  Pad=(" << ppInfo.padX << "," << ppInfo.padY << ")\n";

    /* ------------------------------------------------------------------ */
    /* Step 2 – ONNX Runtime inference.                                     */
    /* ------------------------------------------------------------------ */
    std::cout << "\n[Step 2] Running ONNX inference: " << modelPath << "\n";
    OrtRunner runner;
    if (!runner.loadModel(modelPath)) { dlclose(soHandle); return 1; }

    std::vector<TensorInfo> tensors;
    const std::vector<int64_t> inputShape = {1, 3,
        static_cast<int64_t>(netH),
        static_cast<int64_t>(netW)};

    if (!runner.run(inputData, inputShape, tensors)) { dlclose(soHandle); return 1; }

    /* ------------------------------------------------------------------ */
    /* Step 3 – Optional tensor dump.                                       */
    /* ------------------------------------------------------------------ */
    if (dumpTensor) {
        std::cout << "\n[Step 3] Dumping raw tensors to " << outDir << "/\n";
        dumpTensors(tensors, outDir);
    }

    /* ------------------------------------------------------------------ */
    /* Step 4 – Tensor adapter.                                             */
    /* ------------------------------------------------------------------ */
    std::cout << "\n[Step 4] Adapting tensors for parser.\n";
    std::vector<NvDsInferLayerInfo> layerInfos;
    std::vector<std::string> nameStorage;
    TensorAdapter::adapt(tensors, layerInfos, nameStorage);

    /* ------------------------------------------------------------------ */
    /* Step 5 – Call the custom parser loaded from .so.                     */
    /* ------------------------------------------------------------------ */
    std::cout << "\n[Step 5] Calling " << parserCfg.funcName
              << "() from " << parserCfg.soPath << "\n";

    NvDsInferNetworkInfo netInfo;
    netInfo.width    = static_cast<unsigned int>(netW);
    netInfo.height   = static_cast<unsigned int>(netH);
    netInfo.channels = 3;

    NvDsInferParseDetectionParams detParams;
    detParams.numClassesConfigured = static_cast<unsigned int>(classNames.size());
    detParams.perClassPreclusterThreshold.assign(classNames.size(), confThresh);
    detParams.perClassPostclusterThreshold.assign(classNames.size(), confThresh);

    std::vector<NvDsInferInstanceMaskInfo> detections;
    if (!parserFunc(layerInfos, netInfo, detParams, detections)) {
        std::cerr << "Parser returned false.\n";
        dlclose(soHandle);
        return 1;
    }

    std::cout << "  Detected " << detections.size() << " object(s).\n";
    for (const auto& d : detections) {
        std::cout << "    cls=" << d.classId
                  << " conf=" << d.detectionConfidence
                  << " bbox=[" << d.left << "," << d.top
                  << "," << d.width << "," << d.height << "]\n";
    }

    /* ------------------------------------------------------------------ */
    /* Step 6 – Export JSON results.                                        */
    /* ------------------------------------------------------------------ */
    {
        const std::string jsonPath = outDir + "/results.json";
        std::ofstream jf(jsonPath);
        if (jf) {
            jf << detectionsToJson(detections).dump(2) << "\n";
            std::cout << "\n[Step 6] Results JSON -> " << jsonPath << "\n";
        }
    }

    /* ------------------------------------------------------------------ */
    /* Step 7 – Visualization.                                              */
    /* ------------------------------------------------------------------ */
    std::cout << "\n[Step 7] Visualizing results.\n";
    cv::Mat resultImg = Visualizer::visualize(
        origImage, detections, classNames, netInfo, ppInfo,
        dumpMask, outDir);

    const std::string winName = noDisplay ? "" : "Parser Tester – press any key";
    Visualizer::showAndSave(resultImg, winName, outputPath);

    /* ------------------------------------------------------------------ */
    /* Step 8 – Golden regression test (optional).                          */
    /* ------------------------------------------------------------------ */
    if (!comparePath.empty()) {
        std::cout << "\n[Step 8] Running golden regression test.\n";
        const bool pass = runGoldenTest(comparePath, detections);
        freeMasks(detections);
        dlclose(soHandle);
        return pass ? 0 : 2;
    }

    /* ------------------------------------------------------------------ */
    /* Clean up parser-allocated mask buffers.                              */
    /* ------------------------------------------------------------------ */
    freeMasks(detections);
    dlclose(soHandle);

    std::cout << "\nDone.\n";
    return 0;
}
