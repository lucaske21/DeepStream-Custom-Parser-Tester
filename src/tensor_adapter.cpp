/**
 * tensor_adapter.cpp
 *
 * Converts OrtRunner TensorInfo objects into NvDsInferLayerInfo structs.
 *
 * No data is copied; NvDsInferLayerInfo::buffer points directly into the
 * TensorInfo::data vector.  The caller must keep TensorInfo objects alive.
 */

#include "tensor_adapter.h"

#include <iostream>
#include <algorithm>

void TensorAdapter::adapt(const std::vector<TensorInfo>&   tensorInfos,
                          std::vector<NvDsInferLayerInfo>& layerInfos,
                          std::vector<std::string>&         nameStorage)
{
    layerInfos.clear();
    nameStorage.clear();
    layerInfos.reserve(tensorInfos.size());
    nameStorage.reserve(tensorInfos.size());

    for (const TensorInfo& ti : tensorInfos) {
        /* Keep the name alive in nameStorage so .c_str() remains valid. */
        nameStorage.push_back(ti.name);

        NvDsInferLayerInfo layer;
        layer.dataType     = FLOAT;
        layer.isInput      = false;
        layer.bindingIndex = 0;
        layer.layerName    = nameStorage.back().c_str();

        /* Buffer points directly into the TensorInfo::data vector. */
        layer.buffer = const_cast<float*>(ti.data.data());

        /* Populate inferDims, stripping the leading batch dimension when it
         * equals 1 (standard YOLO-seg ONNX export: [1, ...]).             */
        const std::vector<int64_t>& shape = ti.shape;
        size_t startDim = 0;
        if (shape.size() > 1 && shape[0] == 1) {
            startDim = 1;
        }

        const size_t numDims = shape.size() - startDim;
        if (numDims > NVDSINFER_MAX_DIMS) {
            std::cerr << "[TensorAdapter] Error: tensor \"" << ti.name
                      << "\" has " << numDims
                      << " dims, exceeds NVDSINFER_MAX_DIMS=" << NVDSINFER_MAX_DIMS << "\n";
            continue;
        }

        layer.inferDims.numDims     = static_cast<unsigned int>(numDims);
        layer.inferDims.numElements = 1;

        bool dimsOk = true;
        for (size_t d = 0; d < numDims; ++d) {
            const int64_t dim64 = shape[startDim + d];
            if (dim64 <= 0) {
                std::cerr << "[TensorAdapter] Error: tensor \"" << ti.name
                          << "\" has non-positive dim[" << d << "]=" << dim64 << "\n";
                dimsOk = false;
                break;
            }
            const unsigned int dim = static_cast<unsigned int>(dim64);
            layer.inferDims.d[d]        = dim;
            layer.inferDims.numElements *= dim;
        }
        if (!dimsOk) continue;

        /* Zero-fill remaining dimension slots. */
        for (size_t d = numDims; d < NVDSINFER_MAX_DIMS; ++d) {
            layer.inferDims.d[d] = 0;
        }

        std::cout << "[TensorAdapter] layer=\"" << ti.name << "\"  dims=[";
        for (size_t d = 0; d < numDims; ++d) {
            if (d) std::cout << ",";
            std::cout << layer.inferDims.d[d];
        }
        std::cout << "]  elements=" << layer.inferDims.numElements << "\n";

        layerInfos.push_back(layer);
    }
}
