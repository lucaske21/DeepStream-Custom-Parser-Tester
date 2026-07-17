# DeepStream Custom Parser Tester

A **standalone C++11 framework** for developing and validating DeepStream custom
parsers—specifically YOLO-seg instance segmentation parsers—**without** running a
full DeepStream / GStreamer pipeline.

## Features

| Feature | Description |
|---|---|
| **Zero DeepStream runtime** | Fake DeepStream structs replace the SDK headers |
| **ONNX Runtime backend** | Load `.onnx` models, run FP32 inference |
| **Tensor Adapter** | Converts `Ort::Value` output to `NvDsInferLayerInfo` |
| **YOLO26-seg parser** | Reference parser reusable in real DeepStream |
| **OpenCV visualiser** | Bounding boxes + semi-transparent instance masks |
| **JSON export** | Detection results in nlohmann/json format |
| **Tensor dump** | `--dump-tensor`: save raw output `.bin` files |
| **Mask debug** | `--dump-mask`: save raw / resized / binary mask PNGs |
| **Golden regression** | `--compare expected.json`: IoU + confidence check |
| **Unit tests** | Parser tests compile without ONNX Runtime |

---

## Data Flow

```
Input image (JPEG / PNG)
        │
        ▼
   OpenCV letterbox
   [1, 3, 640, 640] FP32
        │
        ▼
   OrtRunner
   (ONNX Runtime)
        │
        ▼
   TensorInfo[]
   (raw output tensors)
        │
        ▼
   TensorAdapter
   (TensorInfo → NvDsInferLayerInfo)
        │
        ▼
   NvDsInferParseYolo26InstanceMask()
   (output0 [1,300,38] + output1 [1,32,160,160])
        │
        ▼
   NvDsInferInstanceMaskInfo[]
        │
        ▼
   OpenCV Visualiser
        │
        ▼
   imshow() / imwrite()  +  results.json
```

---

## Directory Structure

```
deepstream_parser_tester/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── fake_nvdsinfer.h     ← Fake DeepStream structs (no SDK needed)
│   ├── ort_runner.h         ← ONNX Runtime wrapper
│   ├── tensor_adapter.h     ← Tensor → NvDsInferLayerInfo adapter
│   └── visualizer.h         ← OpenCV visualiser
├── src/
│   ├── main.cpp             ← CLI entry point
│   ├── ort_runner.cpp
│   ├── tensor_adapter.cpp
│   └── visualizer.cpp
├── parser/
│   ├── yolo26_parser.h      ← Parser API (compatible with real DeepStream)
│   └── yolo26_parser.cpp    ← NvDsInferParseYolo26InstanceMask implementation
├── tests/
│   ├── test_parser.cpp      ← Standalone parser unit tests
│   └── expected.json        ← Golden test fixture
├── models/                  ← Place .onnx files here
├── images/                  ← Place test images here
└── output/                  ← Results written here (auto-created)
```

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.10 | |
| C++ compiler | C++11 | GCC ≥ 4.9 or Clang ≥ 3.5 |
| OpenCV | ≥ 3.4 | `libopencv-dev` on Ubuntu |
| ONNX Runtime | ≥ 1.13 | CPU or CUDA build |
| nlohmann/json | 3.11.3 | Fetched automatically by CMake |

### Install ONNX Runtime (Ubuntu)

```bash
# Option A: pre-built release
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar xf onnxruntime-linux-x64-1.17.0.tgz
export ORT_ROOT=$PWD/onnxruntime-linux-x64-1.17.0

# Option B: system package (if available)
sudo apt install libonnxruntime-dev
```

---

## Build

```bash
git clone https://github.com/lucaske21/DeepStream-Custom-Parser-Tester.git
cd DeepStream-Custom-Parser-Tester

mkdir build && cd build

# If ORT is not in a system path, pass -DORT_ROOT=...
cmake .. -DORT_ROOT=/path/to/onnxruntime

make -j$(nproc)
```

### Build with real DeepStream SDK headers (optional)

```bash
cmake .. -DHAVE_DEEPSTREAM=ON \
         -DORT_ROOT=/path/to/onnxruntime
make -j$(nproc)
```

---

## Run Parser Unit Tests (no ONNX Runtime required)

```bash
cd build
ctest --output-on-failure
# or run directly:
./test_parser
```

Expected output:

```
=== yolo26_parser unit tests ===
[test_basic_detection]
  [PASS] parser returned true
  [PASS] exactly 1 detection
  ...
=============================
Passed: 19
Failed: 0
```

---

## Run the Tester

### Basic usage

```bash
./parser_tester \
    --image  ../images/test.jpg \
    --model  ../models/yolo26_seg.onnx \
    --output ../output/result.jpg \
    --conf   0.3 \
    --classes person car bicycle
```

### Save raw tensors and debug masks

```bash
./parser_tester \
    --image        ../images/test.jpg \
    --model        ../models/yolo26_seg.onnx \
    --dump-tensor \
    --dump-mask \
    --no-display
```

Files written to `output/`:

```
output/result.jpg          ← annotated image
output/results.json        ← detections JSON
output/output0.bin         ← raw output0 tensor (float32)
output/output1.bin         ← raw output1 tensor (float32)
output/mask_raw_0.png      ← prototype-space mask crop
output/mask_resize_0.png   ← resized mask
output/mask_final_0.png    ← binary mask
```

### Golden regression test

```bash
./parser_tester \
    --image   ../images/test.jpg \
    --model   ../models/yolo26_seg.onnx \
    --compare ../tests/expected.json
```

Returns exit code `0` on pass, `2` on fail.

---

## JSON Result Format

`output/results.json`:

```json
[
  {
    "class_id": 0,
    "confidence": 0.91,
    "bbox": [50.0, 60.0, 200.0, 240.0],
    "mask_width": 160,
    "mask_height": 160
  }
]
```

Fields:

| Field | Description |
|---|---|
| `class_id` | Integer class index |
| `confidence` | Detection confidence (0–1) |
| `bbox` | `[left, top, width, height]` in network input pixel coordinates |
| `mask_width` | Prototype mask width (typically 160) |
| `mask_height` | Prototype mask height (typically 160) |

---

## Expected ONNX Model Output Format

The parser expects two output tensors with the following layout:

### `output0`  `[1, num_dets, 38]`

Each row contains a single post-NMS detection:

| Columns | Content |
|---|---|
| 0–3 | `x1, y1, x2, y2` — bbox in absolute pixel coordinates (0 … input size) |
| 4   | Detection confidence |
| 5   | Class ID (integer, stored as float) |
| 6–37| 32 prototype mask coefficients |

### `output1`  `[1, 32, 160, 160]`

Prototype masks used to reconstruct per-instance masks:

```
mask[h, w] = sigmoid( Σ_k  coeff[k] * proto[k, h, w] )
```

---

## Extending to Other Parsers

1. Copy `parser/yolo26_parser.{h,cpp}` to e.g. `parser/yolo11_parser.{h,cpp}`.
2. Implement `NvDsInferParseYolo11InstanceMask()` following the same signature.
3. Add the new `.cpp` to the `yolo26_parser` library in `CMakeLists.txt`
   (or create a separate library target).
4. Call the new function from `main.cpp` (or add a `--parser` CLI flag).

---

## Adding a TensorRT Backend (Future Work)

The framework is designed to accept any backend that produces `TensorInfo[]`
objects.  A future `TrtRunner` class would:

1. Load a serialised engine with `nvinfer1::IRuntime`.
2. Run inference and copy device memory to host.
3. Populate `TensorInfo::data` exactly as `OrtRunner` does.
4. Feed the same `TensorAdapter` → Parser pipeline.

This lets you verify that both ONNX Runtime and TensorRT produce identical
parser results before deploying to a live DeepStream pipeline.

---

## License

This project is open-source. See the repository for licence details.
