#ifndef __PUBLIC_H
#define __PUBLIC_H

#include <fstream>
#include <iostream>
#include <assert.h>
#include <string.h>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "./macros.h"
#include "cuda_runtime.h"
#include "NvOnnxParser.h"
using namespace nvonnxparser;

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

using namespace nvinfer1;

#endif  // __PUBLIC_H