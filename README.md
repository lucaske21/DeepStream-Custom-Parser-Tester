# DeepStream Custom Parser Tester

A **standalone C++11 framework** for developing and validating DeepStream custom
parsers—specifically instance-segmentation parsers—**without** running a full
DeepStream / GStreamer pipeline.

The key design principle: **the parser lives in a separate `.so`** that you
build yourself (or get from a third party).  The tester loads it at runtime
via `dlopen` / `dlsym` using command-line arguments—no recompilation of the
tester required when you swap parsers.

## Features

| Feature | Description |
|---|---|
| **Zero DeepStream runtime** | Fake DeepStream structs replace the SDK headers |
| **ONNX Runtime backend** | Load `.onnx` models, run FP32 inference |
| **Tensor Adapter** | Converts `Ort::Value` output to `NvDsInferLayerInfo` |
| **Dynamic parser loading** | Parser `.so` loaded via `dlopen`/`dlsym` at runtime |
| **OpenCV visualiser** | Bounding boxes + semi-transparent instance masks |
| **JSON export** | Detection results in nlohmann/json format |
| **Tensor dump** | `--dump-tensor`: save raw output `.bin` files |
| **Mask debug** | `--dump-mask`: save raw / resized / binary mask PNGs |
| **Golden regression** | `--compare expected.json`: IoU + confidence check |

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
     dlopen(parser_so) + dlsym(parser_func)   ← command-line arguments
   (your custom parser .so)
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
│   ├── main.cpp             ← CLI entry point (dlopen/dlsym dispatch)
│   ├── ort_runner.cpp
│   ├── tensor_adapter.cpp
│   └── visualizer.cpp
├── tests/
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

`make` produces:

| Artifact | Description |
|---|---|
| `build/parser_tester` | The main CLI tool |

---

## Parser Arguments

Pass the parser shared library and its exported C-linkage function explicitly:

```bash
./parser_tester --image img.jpg --model model.onnx \
    --parser-so   /path/to/my_parser.so \
    --parser-func MyParserFunction
```

---

## Writing Your Own Parser .so

Your parser must export one C-linkage function with this exact signature:

```cpp
#include "fake_nvdsinfer.h"   // or the real nvdsinfer.h

extern "C"
bool MyParserFunction(
    const std::vector<NvDsInferLayerInfo>&   outputLayersInfo,
    const NvDsInferNetworkInfo&              networkInfo,
    const NvDsInferParseDetectionParams&     detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>&  objectList);
```

Build it as a shared library:

```bash
g++ -std=c++11 -shared -fPIC \
    -I /path/to/DeepStream-Custom-Parser-Tester/include \
    -o my_parser.so \
    my_parser.cpp
```

## Run the Tester

### Basic usage

```bash
cd build

./parser_tester \
    --image  ../images/test.jpg \
    --model  ../models/yolo26_seg.onnx \
     --parser-so ./libmy_parser.so \
     --parser-func MyParserFunc \
    --output ../output/result.jpg \
    --conf   0.3 \
     --num-classes 80 \
    --classes person car bicycle
```

### Save raw tensors and debug masks

```bash
./parser_tester \
    --image        ../images/test.jpg \
    --model        ../models/yolo26_seg.onnx \
     --parser-so    ./libmy_parser.so \
     --parser-func  MyParserFunc \
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
     --parser-so ./libmy_parser.so \
     --parser-func MyParserFunc \
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

The bundled reference parser expects two output tensors:

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

## Adding a TensorRT Backend (Future Work)

The framework is designed to accept any backend that produces `TensorInfo[]`
objects.  A future `TrtRunner` class would:

1. Load a serialised engine with `nvinfer1::IRuntime`.
2. Run inference and copy device memory to host.
3. Populate `TensorInfo::data` exactly as `OrtRunner` does.
4. Feed the same `TensorAdapter` → Parser (via dlopen) pipeline.

---

## License

This project is open-source. See the repository for licence details.
