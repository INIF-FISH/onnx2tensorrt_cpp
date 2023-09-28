#ifndef __GENERATOR_H
#define __GENERATOR_H

#include "public.h"

namespace TRTgeneratorV1
{
    class TRTgenerator
    {
    private:
        TRTLogger gLogger;
        ICudaEngine *engine;
        bool useFP16;

    public:
        TRTgenerator(TRTLogger &gLogger);
        TRTgenerator();
        ~TRTgenerator();

    public:
        void setFP16(bool state);
        void createEngine(std::string onnxPath, std::string enginePath, int img_width, int img_height, int max_batch_size, std::string input_blob_name);
    };
}

#endif