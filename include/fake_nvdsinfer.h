#pragma once
/**
 * fake_nvdsinfer.h
 *
 * Minimal DeepStream NvDsInfer struct definitions for standalone parser testing.
 * No DeepStream SDK or GStreamer runtime is required.
 *
 * Only the fields actually consumed by the parser are provided here.
 * When building against the real DeepStream SDK, replace this header with
 * the SDK-provided nvdsinfer.h and remove the FAKE_NVDSINFER guard.
 */

#ifndef FAKE_NVDSINFER_H
#define FAKE_NVDSINFER_H

#if defined(USE_DEEPSTREAM_NVDSINFER)

#include <nvdsinfer.h>

#else

#include <vector>
#include <cstring>

/* Maximum number of dimensions supported by NvDsInferDims. */
#define NVDSINFER_MAX_DIMS 8

/** Data types for inference layer buffers. */
typedef enum {
    FLOAT  = 0,
    HALF   = 1,
    INT8   = 2,
    INT32  = 3,
    INT64  = 4
} NvDsInferDataType;

/**
 * Holds the dimensions of an inference layer tensor (batch dimension excluded).
 */
typedef struct {
    unsigned int numDims;
    unsigned int d[NVDSINFER_MAX_DIMS];
    unsigned int numElements;
} NvDsInferDims;

/**
 * Holds information about a single inference layer (input or output).
 * The buffer pointer must remain valid for the lifetime of the layer info.
 */
typedef struct {
    NvDsInferDataType dataType;
    union {
        NvDsInferDims inferDims;
        NvDsInferDims dims;
    };
    unsigned int bindingIndex;
    const char  *layerName;
    void        *buffer;
    bool         isInput;
} NvDsInferLayerInfo;

/**
 * Describes the inference network input dimensions.
 */
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int channels;
} NvDsInferNetworkInfo;

/**
 * Per-class detection thresholds passed to the custom parser.
 */
typedef struct {
    unsigned int         numClassesConfigured;
    std::vector<float>   perClassPreclusterThreshold;
    std::vector<float>   perClassPostclusterThreshold;
} NvDsInferParseDetectionParams;

/**
 * Holds a single detected object with its bounding box, confidence score,
 * class identifier, and instance segmentation mask.
 *
 * The `mask` buffer is heap-allocated by the parser (new float[]).
 * The caller is responsible for freeing it (delete[] mask).
 */
typedef struct {
    unsigned int classId;
    float        left;
    float        top;
    float        width;
    float        height;
    float        detectionConfidence;
    float       *mask;
    unsigned int mask_width;
    unsigned int mask_height;
    unsigned int mask_size;   /* size in bytes = mask_width * mask_height * sizeof(float) */
} NvDsInferInstanceMaskInfo;

#endif /* USE_DEEPSTREAM_NVDSINFER */

#endif /* FAKE_NVDSINFER_H */
