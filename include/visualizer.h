#pragma once
/**
 * visualizer.h
 *
 * OpenCV-based visualization for instance-segmentation results.
 *
 * Draws bounding boxes, class labels, confidence scores, and semi-transparent
 * instance masks on the original (un-letterboxed) image.
 */

#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "fake_nvdsinfer.h"
#include "ort_runner.h"  /* PreprocessInfo */

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * Visualizer renders detection results onto a BGR cv::Mat.
 */
class Visualizer {
public:
    /**
     * Draw bounding boxes and instance masks on @p image.
     *
     * Coordinates in @p objects are in the network input space (e.g. 640×640).
     * They are mapped back to @p image space using @p ppInfo.
     *
     * @param image       Original BGR image (any size).
     * @param objects     Detections from the parser.
     * @param classNames  Human-readable class name for each class ID.
    * @param netInfo     Network input dimensions (kept for call-site symmetry).
     * @param ppInfo      Pre-processing metadata for coordinate inverse-mapping.
     * @param dumpMask    When true, intermediate mask images are saved to disk.
     * @param maskDumpDir Directory where mask debug images are written.
     * @return            Annotated BGR image (same size as @p image).
     */
    static cv::Mat visualize(
        const cv::Mat&                             image,
        const std::vector<NvDsInferInstanceMaskInfo>& objects,
        const std::vector<std::string>&            classNames,
        const NvDsInferNetworkInfo&                netInfo,
        const PreprocessInfo&                      ppInfo,
        bool                                       dumpMask    = false,
        const std::string&                         maskDumpDir = "output");

    /**
     * Optionally display @p result in a window and/or save it to @p savePath.
     *
     * @param result      The annotated image.
     * @param windowName  Window title for cv::imshow(); pass "" to skip display.
     * @param savePath    File path for cv::imwrite(); pass "" to skip saving.
     */
    static void showAndSave(const cv::Mat&     result,
                            const std::string& windowName,
                            const std::string& savePath);
};

#endif /* VISUALIZER_H */
