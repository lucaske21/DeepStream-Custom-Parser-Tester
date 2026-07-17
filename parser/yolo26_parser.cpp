/**
 * yolo26_parser.cpp
 *
 * Implementation of NvDsInferParseYolo26InstanceMask for YOLO26-seg.
 *
 * This file is self-contained and can be compiled both:
 *  - As part of the standalone parser tester (include fake_nvdsinfer.h), and
 *  - As a DeepStream custom parser plugin (include real nvdsinfer.h).
 *
 * Output tensor layout assumed
 * ----------------------------
 *  output0  [1, num_dets, cols]  (post-NMS detections)
 *    col 0-3  : x1, y1, x2, y2  – absolute pixel coords in network input space
 *    col 4    : confidence       – detection confidence ∈ (0, 1]
 *    col 5    : class_id         – integer class index
 *    col 6-37 : mask coefficients m₀ … m₃₁  (32 values)
 *    cols≥38 are ignored; cols<38 produce an error return.
 *
 *  output1  [1, num_proto, proto_h, proto_w]  (prototype masks)
 *    Typically [1, 32, 160, 160].
 */

#include "yolo26_parser.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>

/* Number of fixed leading columns before mask coefficients. */
static const int DET_FIXED_COLS = 6;   /* x1,y1,x2,y2, conf, cls_id */
static const int NUM_PROTO_COEFFS = 32;
static const int MIN_COLS = DET_FIXED_COLS + NUM_PROTO_COEFFS;  /* 38 */

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * Find a layer by name in the output list.
 * Falls back to index-based access if @p name is not found.
 */
static const NvDsInferLayerInfo* findLayer(
    const std::vector<NvDsInferLayerInfo>& layers,
    const char* name,
    unsigned int fallbackIndex)
{
    for (const auto& l : layers) {
        if (l.layerName && std::strcmp(l.layerName, name) == 0) {
            return &l;
        }
    }
    if (fallbackIndex < static_cast<unsigned int>(layers.size())) {
        return &layers[fallbackIndex];
    }
    return nullptr;
}

extern "C"
bool NvDsInferParseYolo26InstanceMask(
    const std::vector<NvDsInferLayerInfo>&   outputLayersInfo,
    const NvDsInferNetworkInfo&              networkInfo,
    const NvDsInferParseDetectionParams&     detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>&  objectList)
{
    objectList.clear();

    /* ------------------------------------------------------------------ */
    /* 1. Locate the two output layers.                                     */
    /* ------------------------------------------------------------------ */
    const NvDsInferLayerInfo* output0 = findLayer(outputLayersInfo, "output0", 0);
    const NvDsInferLayerInfo* output1 = findLayer(outputLayersInfo, "output1", 1);

    if (!output0 || !output0->buffer) {
        std::cerr << "[yolo26_parser] output0 layer not found or buffer is null.\n";
        return false;
    }
    if (!output1 || !output1->buffer) {
        std::cerr << "[yolo26_parser] output1 layer not found or buffer is null.\n";
        return false;
    }

    /* ------------------------------------------------------------------ */
    /* 2. Validate and extract tensor shapes.                               */
    /* ------------------------------------------------------------------ */
    /* output0 dims (batch stripped): [num_dets, cols] */
    const NvDsInferDims& dims0 = output0->inferDims;
    if (dims0.numDims < 2) {
        std::cerr << "[yolo26_parser] output0 numDims < 2.\n";
        return false;
    }
    const unsigned int numDets = dims0.d[0];
    const unsigned int cols    = dims0.d[1];

    if (static_cast<int>(cols) < MIN_COLS) {
        std::cerr << "[yolo26_parser] output0 cols=" << cols
                  << " < required " << MIN_COLS << ".\n";
        return false;
    }

    /* output1 dims (batch stripped): [num_proto, proto_h, proto_w] */
    const NvDsInferDims& dims1 = output1->inferDims;
    if (dims1.numDims < 3) {
        std::cerr << "[yolo26_parser] output1 numDims < 3.\n";
        return false;
    }
    const unsigned int numProto = dims1.d[0];
    const unsigned int protoH   = dims1.d[1];
    const unsigned int protoW   = dims1.d[2];

    /* Number of coefficients actually used (capped at num_proto). */
    const unsigned int numCoeffs =
        std::min(static_cast<unsigned int>(NUM_PROTO_COEFFS), numProto);

    /* ------------------------------------------------------------------ */
    /* 3. Determine confidence threshold.                                   */
    /* ------------------------------------------------------------------ */
    float confThresh = 0.25f;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        confThresh = detectionParams.perClassPreclusterThreshold[0];
    }

    /* ------------------------------------------------------------------ */
    /* 4. Parse detections.                                                 */
    /* ------------------------------------------------------------------ */
    const float* det0Data  = static_cast<const float*>(output0->buffer);
    const float* protoData = static_cast<const float*>(output1->buffer);

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    for (unsigned int i = 0; i < numDets; ++i) {
        const float* det = det0Data + i * cols;

        const float x1   = det[0];
        const float y1   = det[1];
        const float x2   = det[2];
        const float y2   = det[3];
        const float conf = det[4];
        const int   cls  = static_cast<int>(det[5]);

        /* Skip low-confidence detections. */
        if (conf < confThresh) continue;

        /* Skip degenerate boxes. */
        if (x2 <= x1 || y2 <= y1) continue;

        /* Clamp to network bounds. */
        const float cx1 = std::max(0.0f, std::min(x1, netW));
        const float cy1 = std::max(0.0f, std::min(y1, netH));
        const float cx2 = std::max(0.0f, std::min(x2, netW));
        const float cy2 = std::max(0.0f, std::min(y2, netH));

        /* -------------------------------------------------------------- */
        /* 4a. Allocate and reconstruct the 2-D instance mask.             */
        /*                                                                  */
        /* mask[h, w] = sigmoid( Σ_k  coeff[k] * proto[k, h, w] )         */
        /* -------------------------------------------------------------- */
        const unsigned int maskPixels = protoH * protoW;
        float* maskBuf = new float[maskPixels];

        const float* coeffs = det + DET_FIXED_COLS;

        for (unsigned int h = 0; h < protoH; ++h) {
            for (unsigned int w = 0; w < protoW; ++w) {
                float val = 0.0f;
                for (unsigned int k = 0; k < numCoeffs; ++k) {
                    /* proto layout: [num_proto, protoH, protoW] */
                    val += coeffs[k] * protoData[k * maskPixels + h * protoW + w];
                }
                maskBuf[h * protoW + w] = sigmoid(val);
            }
        }

        /* -------------------------------------------------------------- */
        /* 4b. Fill NvDsInferInstanceMaskInfo.                              */
        /* -------------------------------------------------------------- */
        NvDsInferInstanceMaskInfo info;
        info.left               = cx1;
        info.top                = cy1;
        info.width              = cx2 - cx1;
        info.height             = cy2 - cy1;
        info.detectionConfidence = conf;
        info.classId            = (cls >= 0) ? cls : 0;
        info.mask               = maskBuf;
        info.mask_width         = static_cast<int>(protoW);
        info.mask_height        = static_cast<int>(protoH);
        info.mask_size          = static_cast<int>(maskPixels * sizeof(float));

        objectList.push_back(info);
    }

    return true;
}
