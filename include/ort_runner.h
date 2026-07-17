#pragma once
/**
 * ort_runner.h
 *
 * Lightweight ONNX Runtime inference wrapper.
 * Loads a model, pre-processes a BGR image into a CHW FP32 tensor, runs
 * inference, and returns raw output tensor data for downstream adapters.
 */

#ifndef ORT_RUNNER_H
#define ORT_RUNNER_H

#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * Plain-old-data holder for a single output tensor produced by OrtRunner.
 */
struct TensorInfo {
    std::string          name;    /**< Output layer name (e.g. "output0"). */
    std::vector<int64_t> shape;   /**< Full tensor shape including batch dim. */
    std::vector<float>   data;    /**< Flat, row-major tensor data (FP32). */
};

/**
 * Stores pre-processing metadata so that detection coordinates can be mapped
 * back from network space to the original image space.
 */
struct PreprocessInfo {
    float scale;  /**< Uniform scale applied to the shorter side. */
    int   padX;   /**< Horizontal padding added (pixels, each side). */
    int   padY;   /**< Vertical padding added (pixels, each side). */
    int   origW;  /**< Width of the original (pre-letterbox) image. */
    int   origH;  /**< Height of the original (pre-letterbox) image. */
    int   netW;   /**< Network input width (e.g. 640). */
    int   netH;   /**< Network input height (e.g. 640). */
};

/**
 * OrtRunner wraps an ONNX Runtime session.
 *
 * Usage:
 *   OrtRunner runner;
 *   runner.loadModel("model.onnx");
 *   std::vector<TensorInfo> outputs;
 *   runner.run(inputData, {1,3,640,640}, outputs);
 */
class OrtRunner {
public:
    OrtRunner();
    ~OrtRunner();

    /**
     * Load an ONNX model from disk.
     * @param modelPath  Path to the .onnx file.
     * @return true on success, false on error (message written to stderr).
     */
    bool loadModel(const std::string& modelPath);

    /**
     * Run inference with a pre-built FP32 CHW input tensor.
     *
     * @param inputData   Flat FP32 data in NCHW order.
     * @param inputShape  Shape of the input tensor, e.g. {1, 3, 640, 640}.
     * @param outputs     [out] One TensorInfo per model output.
     * @return true on success, false on error.
     */
    bool run(const std::vector<float>& inputData,
             const std::vector<int64_t>& inputShape,
             std::vector<TensorInfo>& outputs);

    /** Names of the model's input layers. */
    const std::vector<std::string>& getInputNames()  const { return inputNames_;  }
    /** Names of the model's output layers. */
    const std::vector<std::string>& getOutputNames() const { return outputNames_; }

private:
    Ort::Env                          env_;
    std::unique_ptr<Ort::Session>     session_;
    Ort::AllocatorWithDefaultOptions  allocator_;
    std::vector<std::string>          inputNames_;
    std::vector<std::string>          outputNames_;
};

/**
 * Convenience free function: read @p imagePath with OpenCV, apply letterbox
 * pre-processing to produce a [1,3,@p netH,@p netW] FP32 tensor, and populate
 * @p ppInfo for later coordinate inverse-mapping.
 *
 * @return The original (un-letterboxed) BGR image; empty Mat on failure.
 */
cv::Mat preprocessImage(const std::string& imagePath,
                        int netW, int netH,
                        PreprocessInfo& ppInfo,
                        std::vector<float>& inputData);

#endif /* ORT_RUNNER_H */

