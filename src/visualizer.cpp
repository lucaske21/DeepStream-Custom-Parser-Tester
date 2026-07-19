/**
 * visualizer.cpp
 *
 * OpenCV-based visualization of DeepStream instance segmentation results.
 *
 * Coordinate mapping
 * ------------------
 * Detections from the parser are in network input space (e.g. 0-640 pixels).
 * The visualizer uses PreprocessInfo to invert the letterbox transform and
 * draw results on the original-resolution image.
 *
 * Mask assembly
 * -------------
 * NvDsInferInstanceMaskInfo::mask is a per-object mask. Its dimensions are
 * NvDsInferInstanceMaskInfo::mask_width and mask_height, and it corresponds to
 * the object's bounding box, not the full network/prototype image. This file
 * resizes that object mask to the bbox in original-image space and blends it.
 */

#include "visualizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>

/* Six distinct colours for class-id colouring. */
static const cv::Scalar CLASS_COLORS[] = {
    cv::Scalar(  0, 114, 189),
    cv::Scalar(217,  83,  25),
    cv::Scalar(237, 177,  32),
    cv::Scalar( 44, 160,  44),
    cv::Scalar(148, 103, 189),
    cv::Scalar( 23, 190, 207)
};
static const int NUM_COLORS = 6;

static inline cv::Scalar classColor(unsigned int classId) {
    return CLASS_COLORS[classId % NUM_COLORS];
}

/* Format a float as a string with @p decimals places. */
static std::string fmtFloat(float v, int decimals = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << v;
    return oss.str();
}

cv::Mat Visualizer::visualize(
    const cv::Mat&                                image,
    const std::vector<NvDsInferInstanceMaskInfo>& objects,
    const std::vector<std::string>&               classNames,
    const NvDsInferNetworkInfo&                   netInfo,
    const PreprocessInfo&                         ppInfo,
    bool                                          dumpMask,
    const std::string&                            maskDumpDir)
{
    cv::Mat result = image.clone();

    if (objects.empty()) {
        std::cout << "[Visualizer] No detections to draw.\n";
        return result;
    }

    (void)netInfo;

    /* We accumulate mask overlays into a separate Mat, then blend once. */
    cv::Mat overlay = result.clone();

    for (size_t idx = 0; idx < objects.size(); ++idx) {
        const NvDsInferInstanceMaskInfo& obj = objects[idx];
        const cv::Scalar color = classColor(obj.classId);

        /* ---------------------------------------------------------------- */
        /* Convert bbox from network space to original image space.          */
        /* ---------------------------------------------------------------- */
        const float x1Net = obj.left;
        const float y1Net = obj.top;
        const float x2Net = obj.left + obj.width;
        const float y2Net = obj.top  + obj.height;

        const float x1Orig = (x1Net - ppInfo.padX) / ppInfo.scale;
        const float y1Orig = (y1Net - ppInfo.padY) / ppInfo.scale;
        const float x2Orig = (x2Net - ppInfo.padX) / ppInfo.scale;
        const float y2Orig = (y2Net - ppInfo.padY) / ppInfo.scale;

        const int fullBx1 = static_cast<int>(std::floor(x1Orig));
        const int fullBy1 = static_cast<int>(std::floor(y1Orig));
        const int fullBx2 = static_cast<int>(std::ceil(x2Orig));
        const int fullBy2 = static_cast<int>(std::ceil(y2Orig));

        const int bx1 = std::max(0, fullBx1);
        const int by1 = std::max(0, fullBy1);
        const int bx2 = std::min(result.cols, fullBx2);
        const int by2 = std::min(result.rows, fullBy2);

        if (bx2 <= bx1 || by2 <= by1) continue;

        const cv::Rect bboxRect(bx1, by1, bx2 - bx1, by2 - by1);

        /* ---------------------------------------------------------------- */
        /* Draw bounding box.                                                */
        /* ---------------------------------------------------------------- */
        cv::rectangle(result, bboxRect, color, 2);
        cv::rectangle(overlay, bboxRect, color, 2);

        /* ---------------------------------------------------------------- */
        /* Draw label: "<ClassName> <conf>".                                 */
        /* ---------------------------------------------------------------- */
        const size_t classIdx = static_cast<size_t>(obj.classId);
        std::string className =
            (classIdx < classNames.size())
                ? classNames[classIdx]
                : ("cls" + std::to_string(obj.classId));
        const std::string label = className + " " + fmtFloat(obj.detectionConfidence);

        int baseline = 0;
        const cv::Size textSz =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int lx1 = bx1;
        const int ly1 = std::max(0, by1 - textSz.height - 6);
        cv::rectangle(result,
                      cv::Point(lx1, ly1),
                      cv::Point(lx1 + textSz.width + 2, ly1 + textSz.height + 4),
                      color, cv::FILLED);
        cv::rectangle(overlay,
                      cv::Point(lx1, ly1),
                      cv::Point(lx1 + textSz.width + 2, ly1 + textSz.height + 4),
                      color, cv::FILLED);
        cv::putText(result, label,
                    cv::Point(lx1 + 1, ly1 + textSz.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(overlay, label,
                    cv::Point(lx1 + 1, ly1 + textSz.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        /* ---------------------------------------------------------------- */
        /* Instance mask overlay.                                            */
        /* ---------------------------------------------------------------- */
        if (!obj.mask || obj.mask_width == 0 || obj.mask_height == 0) continue;

        const int maskW = static_cast<int>(obj.mask_width);
        const int maskH = static_cast<int>(obj.mask_height);

        const int fullBoxW = fullBx2 - fullBx1;
        const int fullBoxH = fullBy2 - fullBy1;
        if (fullBoxW <= 0 || fullBoxH <= 0) continue;

        /* Build OpenCV Mat that wraps the parser-allocated per-object mask. */
        cv::Mat objectMask(maskH, maskW, CV_32F, obj.mask);

        if (dumpMask) {
            cv::Mat dbg;
            objectMask.convertTo(dbg, CV_8U, 255.0f);
            const std::string path =
                maskDumpDir + "/mask_raw_" + std::to_string(idx) + ".png";
            if (cv::imwrite(path, dbg))
                std::cout << "[Visualizer] Saved " << path << "\n";
            else
                std::cerr << "[Visualizer] Failed to save " << path << "\n";
        }

        /* Resize the object mask to the full, unclipped original-image bbox. */
        cv::Mat resizedMask;
        cv::resize(objectMask, resizedMask,
                   cv::Size(fullBoxW, fullBoxH),
                   0, 0, cv::INTER_LINEAR);

        if (dumpMask) {
            cv::Mat dbg;
            resizedMask.convertTo(dbg, CV_8U, 255.0f);
            const std::string path =
                maskDumpDir + "/mask_resize_" + std::to_string(idx) + ".png";
            if (cv::imwrite(path, dbg))
                std::cout << "[Visualizer] Saved " << path << "\n";
            else
                std::cerr << "[Visualizer] Failed to save " << path << "\n";
        }

        /* Threshold at 0.5 to get a binary mask. */
        cv::Mat binaryMask;
        cv::threshold(resizedMask, binaryMask, 0.5f, 255.0f, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8U);

        const int maskX = bx1 - fullBx1;
        const int maskY = by1 - fullBy1;
        cv::Mat visibleMask = binaryMask(cv::Rect(maskX, maskY,
                                                  bboxRect.width,
                                                  bboxRect.height));

        if (dumpMask) {
            const std::string path =
                maskDumpDir + "/mask_final_" + std::to_string(idx) + ".png";
            if (cv::imwrite(path, visibleMask))
                std::cout << "[Visualizer] Saved " << path << "\n";
            else
                std::cerr << "[Visualizer] Failed to save " << path << "\n";
        }

        /* Paint the colour overlay onto the overlay Mat inside the bbox. */
        cv::Mat roi = overlay(bboxRect);
        cv::Mat colorLayer(bboxRect.height, bboxRect.width, CV_8UC3, color);
        colorLayer.copyTo(roi, visibleMask);
    }

    /* Blend overlay (40 % opacity) into the result. */
    cv::addWeighted(result, 0.6, overlay, 0.4, 0.0, result);

    return result;
}

void Visualizer::showAndSave(const cv::Mat&     result,
                             const std::string& windowName,
                             const std::string& savePath)
{
    if (!savePath.empty()) {
        if (cv::imwrite(savePath, result)) {
            std::cout << "[Visualizer] Saved result to " << savePath << "\n";
        } else {
            std::cerr << "[Visualizer] Failed to save " << savePath << "\n";
        }
    }

    if (!windowName.empty()) {
        cv::imshow(windowName, result);
        std::cout << "[Visualizer] Press any key to close the window.\n";
        cv::waitKey(0);
    }
}
