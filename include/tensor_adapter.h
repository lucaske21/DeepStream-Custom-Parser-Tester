#pragma once
/**
 * tensor_adapter.h
 *
 * Converts the raw TensorInfo objects produced by OrtRunner into
 * NvDsInferLayerInfo structs expected by DeepStream custom parsers.
 *
 * The adapter does NOT copy tensor data; it sets buffer pointers directly into
 * the TensorInfo::data vectors.  Callers must keep the TensorInfo objects alive
 * as long as the resulting NvDsInferLayerInfo vector is in use.
 */

#ifndef TENSOR_ADAPTER_H
#define TENSOR_ADAPTER_H

#include "fake_nvdsinfer.h"
#include "ort_runner.h"

#include <vector>
#include <string>

/**
 * TensorAdapter converts OrtRunner output tensors to NvDsInferLayerInfo.
 */
class TensorAdapter {
public:
    /**
     * Populate @p layerInfos from @p tensorInfos.
     *
     * @param tensorInfos  Source tensors (e.g. from OrtRunner::run).
     * @param layerInfos   [out] One NvDsInferLayerInfo per tensor.
     * @param nameStorage  [out] String storage whose .c_str() pointers are used
     *                     by layerInfos[i].layerName.  Must outlive layerInfos.
     */
    static void adapt(const std::vector<TensorInfo>&    tensorInfos,
                      std::vector<NvDsInferLayerInfo>&  layerInfos,
                      std::vector<std::string>&          nameStorage);
};

#endif /* TENSOR_ADAPTER_H */
