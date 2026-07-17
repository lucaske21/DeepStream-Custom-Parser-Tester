#pragma once
/**
 * yolo26_parser.h
 *
 * Declaration of the DeepStream custom parser for YOLO26-seg instance
 * segmentation.
 *
 * This file is intentionally kept compatible with both the real DeepStream SDK
 * (nvdsinfer.h) and the standalone tester (fake_nvdsinfer.h).  Switch the
 * guarded include below depending on the build environment.
 */

#ifndef YOLO26_PARSER_H
#define YOLO26_PARSER_H

/* Use the fake header when building outside DeepStream. */
#ifdef HAVE_DEEPSTREAM
#  include "nvdsinfer.h"
#else
#  include "fake_nvdsinfer.h"
#endif

#include <vector>

/**
 * NvDsInferParseYolo26InstanceMask
 *
 * Parses the raw output tensors of a YOLO26-seg model into a list of
 * NvDsInferInstanceMaskInfo objects suitable for downstream DeepStream
 * processing.
 *
 * Expected model outputs
 * ----------------------
 *  output0  [1, num_dets, cols]
 *    Each row: [x1, y1, x2, y2, confidence, class_id, m0 … m31]
 *    where (x1,y1,x2,y2) are absolute pixel coordinates in the network input
 *    space (e.g. 0–640), confidence ∈ (0,1], class_id ∈ {0,…,N-1}, and
 *    m0…m31 are the 32 prototype mask coefficients.
 *
 *  output1  [1, num_proto, proto_h, proto_w]
 *    Prototype masks – typically [1, 32, 160, 160].
 *
 * Mask reconstruction
 * -------------------
 *  For each detection the parser computes the full 160×160 instance mask as:
 *    mask[h,w] = sigmoid( Σ_k  coeff[k] * proto[k,h,w] )
 *  and stores it in NvDsInferInstanceMaskInfo::mask (heap-allocated with new[]).
 *  The caller is responsible for freeing it with delete[].
 *
 * @param outputLayersInfo  Raw output layers from the inference engine.
 * @param networkInfo       Network input width × height × channels.
 * @param detectionParams   Per-class confidence thresholds.
 * @param objectList        [out] Detected objects with masks.
 * @return                  true on success, false if required layers are absent.
 */
extern "C"
bool NvDsInferParseYolo26InstanceMask(
    const std::vector<NvDsInferLayerInfo>&    outputLayersInfo,
    const NvDsInferNetworkInfo&               networkInfo,
    const NvDsInferParseDetectionParams&      detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>&   objectList);

#endif /* YOLO26_PARSER_H */
