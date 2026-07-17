/**
 * visualizer.cpp
 *
 * OpenCV-based visualization of YOLO-seg instance segmentation results.
 *
 * Coordinate mapping
 * ------------------
 * Detections from the parser are in network input space (e.g. 0-640 pixels).
 * The visualizer uses PreprocessInfo to invert the letterbox transform and
 * draw results on the original-resolution image.
 *
 * Mask assembly
 * -------------
 * The parser stores a full-prototype-space mask (e.g. 160×160) in
 * NvDsInferInstanceMaskInfo::mask.  This function:
 *   1. Crops the mask to the bbox region (scaled to prototype space).
 *   2. Resizes the crop to the original-image-space bbox size.
 *   3. Thresholds at 0.5 to produce a binary mask.
 *   4. Blends a colour overlay onto the image.
 */

#include "visualizer.h"

#include <algorithm>
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

static inline cv::Scalar classColor(int classId) {
    return CLASS_COLORS[static_cast<unsigned int>(classId) % NUM_COLORS];
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

    /* Scale factors: prototype space → network space → original image space.   */
    /* netInfo.width / netInfo.height are used below for per-object sx, sy.    */

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

        const int bx1 = std::max(0, static_cast<int>(x1Orig));
        const int by1 = std::max(0, static_cast<int>(y1Orig));
        const int bx2 = std::min(result.cols, static_cast<int>(x2Orig));
        const int by2 = std::min(result.rows, static_cast<int>(y2Orig));

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
        std::string className =
            (static_cast<size_t>(obj.classId) < classNames.size())
                ? classNames[obj.classId]
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
        if (!obj.mask || obj.mask_width <= 0 || obj.mask_height <= 0) continue;

        const int protoW = obj.mask_width;
        const int protoH = obj.mask_height;

        /* Crop the full prototype-space mask to the bbox region.           */
        /* Prototype coords = network coords * (proto / net).               */
        const float sx = static_cast<float>(protoW) / netInfo.width;
        const float sy = static_cast<float>(protoH) / netInfo.height;

        const int px1 = std::max(0, static_cast<int>(x1Net * sx));
        const int py1 = std::max(0, static_cast<int>(y1Net * sy));
        const int px2 = std::min(protoW, static_cast<int>(x2Net * sx));
        const int py2 = std::min(protoH, static_cast<int>(y2Net * sy));

        if (px2 <= px1 || py2 <= py1) continue;

        /* Build OpenCV Mat that wraps the parser-allocated buffer (no copy). */
        cv::Mat fullMask(protoH, protoW, CV_32F, obj.mask);

        /* Crop to bbox region in prototype space. */
        cv::Mat croppedMask = fullMask(cv::Rect(px1, py1, px2 - px1, py2 - py1)).clone();

        if (dumpMask) {
            cv::Mat dbg;
            croppedMask.convertTo(dbg, CV_8U, 255.0f);
            const std::string path =
                maskDumpDir + "/mask_raw_" + std::to_string(idx) + ".png";
            if (cv::imwrite(path, dbg))
                std::cout << "[Visualizer] Saved " << path << "\n";
            else
                std::cerr << "[Visualizer] Failed to save " << path << "\n";
        }

        /* Resize crop to original-image-space bbox size. */
        cv::Mat resizedMask;
        cv::resize(croppedMask, resizedMask,
                   cv::Size(bboxRect.width, bboxRect.height),
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

        if (dumpMask) {
            const std::string path =
                maskDumpDir + "/mask_final_" + std::to_string(idx) + ".png";
            if (cv::imwrite(path, binaryMask))
                std::cout << "[Visualizer] Saved " << path << "\n";
            else
                std::cerr << "[Visualizer] Failed to save " << path << "\n";
        }

        /* Paint the colour overlay onto the overlay Mat inside the bbox. */
        cv::Mat roi = overlay(bboxRect);
        cv::Mat colorLayer(bboxRect.height, bboxRect.width, CV_8UC3, color);
        colorLayer.copyTo(roi, binaryMask);
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
