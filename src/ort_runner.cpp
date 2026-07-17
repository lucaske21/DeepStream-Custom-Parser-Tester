/**
 * ort_runner.cpp
 *
 * OrtRunner — ONNX Runtime inference wrapper.
 *
 * Pre-processes a BGR OpenCV image into a [1, 3, H, W] FP32 letterbox tensor,
 * runs the ONNX model, and returns raw output tensors for the TensorAdapter.
 */

#include "ort_runner.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <cstring>

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Letterbox-resize @p src to exactly @p targetW × @p targetH.
 *
 * Maintains aspect ratio by padding with grey (114, 114, 114).
 * Fills @p info with the scale factor and padding offsets so that
 * downstream code can invert the transform.
 */
static cv::Mat letterbox(const cv::Mat& src,
                         int targetW, int targetH,
                         PreprocessInfo& info)
{
    const float scaleW = static_cast<float>(targetW) / src.cols;
    const float scaleH = static_cast<float>(targetH) / src.rows;
    const float scale  = std::min(scaleW, scaleH);

    const int newW = static_cast<int>(std::round(src.cols * scale));
    const int newH = static_cast<int>(std::round(src.rows * scale));
    const int padX = (targetW - newW) / 2;
    const int padY = (targetH - newH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(targetH, targetW, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padX, padY, newW, newH)));

    info.scale = scale;
    info.padX  = padX;
    info.padY  = padY;
    info.origW = src.cols;
    info.origH = src.rows;
    info.netW  = targetW;
    info.netH  = targetH;

    return canvas;
}

/**
 * Convert a letterboxed BGR uint8 image to a flat FP32 NCHW tensor.
 * Pixel values are normalised to [0, 1] and the channel order is BGR→RGB.
 */
static void bgrToNchw(const cv::Mat& img, std::vector<float>& out)
{
    const int H = img.rows;
    const int W = img.cols;
    const int C = img.channels();     /* always 3 after letterbox */

    out.resize(static_cast<size_t>(1 * C * H * W));

    /* Separate channels: OpenCV stores in BGR; models expect RGB. */
    std::vector<cv::Mat> chans(C);
    cv::split(img, chans);            /* chans[0]=B, [1]=G, [2]=R */

    /* Write in RGB order (R=chans[2], G=chans[1], B=chans[0]). */
    const int planeSize = H * W;
    for (int c = 0; c < C; ++c) {
        int srcC = C - 1 - c;         /* B→2, G→1, R→0 */
        float* dst = out.data() + c * planeSize;
        for (int y = 0; y < H; ++y) {
            const uchar* row = chans[srcC].ptr<uchar>(y);
            for (int x = 0; x < W; ++x) {
                dst[y * W + x] = row[x] / 255.0f;
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/* OrtRunner                                                                   */
/* -------------------------------------------------------------------------- */

OrtRunner::OrtRunner()
    : env_(ORT_LOGGING_LEVEL_WARNING, "deepstream_parser_tester")
{}

OrtRunner::~OrtRunner() {}

bool OrtRunner::loadModel(const std::string& modelPath)
{
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_.reset(new Ort::Session(env_, modelPath.c_str(), opts));

        /* Collect input names. */
        const size_t numIn = session_->GetInputCount();
        inputNames_.clear();
        inputNames_.reserve(numIn);
        for (size_t i = 0; i < numIn; ++i) {
            auto n = session_->GetInputNameAllocated(i, allocator_);
            inputNames_.push_back(std::string(n.get()));
        }

        /* Collect output names. */
        const size_t numOut = session_->GetOutputCount();
        outputNames_.clear();
        outputNames_.reserve(numOut);
        for (size_t i = 0; i < numOut; ++i) {
            auto n = session_->GetOutputNameAllocated(i, allocator_);
            outputNames_.push_back(std::string(n.get()));
        }

        std::cout << "[OrtRunner] Loaded model: " << modelPath << "\n";
        std::cout << "[OrtRunner] Inputs:  " << numIn  << "  Outputs: " << numOut << "\n";
        for (const auto& n : outputNames_) {
            std::cout << "[OrtRunner]   output: " << n << "\n";
        }
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "[OrtRunner] loadModel error: " << e.what() << "\n";
        return false;
    }
}

bool OrtRunner::run(const std::vector<float>&    inputData,
                    const std::vector<int64_t>&  inputShape,
                    std::vector<TensorInfo>&     outputs)
{
    if (!session_) {
        std::cerr << "[OrtRunner] No model loaded.\n";
        return false;
    }
    try {
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        /* Build input tensor (data is NOT copied; the vector must stay alive). */
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float*>(inputData.data()),
            inputData.size(),
            inputShape.data(),
            inputShape.size());

        /* Pointer arrays for the Run() call. */
        std::vector<const char*> inPtrs, outPtrs;
        inPtrs.reserve(inputNames_.size());
        outPtrs.reserve(outputNames_.size());
        for (const auto& n : inputNames_)  inPtrs.push_back(n.c_str());
        for (const auto& n : outputNames_) outPtrs.push_back(n.c_str());

        auto outTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inPtrs.data(),  &inputTensor, inPtrs.size(),
            outPtrs.data(), outPtrs.size());

        /* Copy output data into TensorInfo. */
        outputs.clear();
        outputs.reserve(outTensors.size());
        for (size_t i = 0; i < outTensors.size(); ++i) {
            TensorInfo ti;
            ti.name  = outputNames_[i];

            auto tsInfo = outTensors[i].GetTensorTypeAndShapeInfo();
            ti.shape = tsInfo.GetShape();

            const size_t n = tsInfo.GetElementCount();
            const float* ptr = outTensors[i].GetTensorData<float>();
            ti.data.assign(ptr, ptr + n);

            outputs.push_back(std::move(ti));
        }
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "[OrtRunner] run error: " << e.what() << "\n";
        return false;
    }
}

/* -------------------------------------------------------------------------- */
/* Standalone preprocessing helper (called from main.cpp)                     */
/* -------------------------------------------------------------------------- */

/**
 * Preprocess a BGR image for YOLO inference.
 *
 * @param imagePath  Path to the input image file.
 * @param netW       Network input width.
 * @param netH       Network input height.
 * @param ppInfo     [out] Preprocessing metadata (scale, padding, etc.).
 * @param inputData  [out] Flat FP32 NCHW tensor ready for OrtRunner::run().
 * @return           The letterboxed BGR image (useful for debug display).
 *                   Returns an empty Mat on failure.
 */
cv::Mat preprocessImage(const std::string& imagePath,
                        int netW, int netH,
                        PreprocessInfo& ppInfo,
                        std::vector<float>& inputData)
{
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "[preprocess] Cannot read image: " << imagePath << "\n";
        return cv::Mat();
    }

    cv::Mat lb = letterbox(img, netW, netH, ppInfo);
    bgrToNchw(lb, inputData);
    return img;   /* return original (not the letterboxed version) */
}
