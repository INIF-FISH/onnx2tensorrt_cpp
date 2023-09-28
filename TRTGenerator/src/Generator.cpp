#include "../include/Generator.h"

namespace TRTgeneratorV1
{
    TRTgenerator::TRTgenerator(TRTLogger &gLogger)
    {
        this->gLogger = gLogger;
    }

    TRTgenerator::TRTgenerator()
    {
        this->gLogger = TRTLogger();
    }

    TRTgenerator::~TRTgenerator()
    {
    }

    void TRTgenerator::setFP16(bool state)
    {
        this->useFP16 = state;
    }

    void TRTgenerator::createEngine(std::string onnxPath, std::string enginePath, int img_width, int img_height, int max_batch_size, std::string input_blob_name)
    {
        IBuilder *builder = createInferBuilder(this->gLogger);
        assert(builder != nullptr);
        uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition *network = builder->createNetworkV2(flag);
        assert(network != nullptr);
        IBuilderConfig *config = builder->createBuilderConfig();
        assert(config != nullptr);

        auto profile = builder->createOptimizationProfile();
        Dims dims = Dims4{1, 3, img_height, img_width};
        profile->setDimensions(input_blob_name.c_str(),
                               OptProfileSelector::kMIN, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
        profile->setDimensions(input_blob_name.c_str(),
                               OptProfileSelector::kOPT, Dims4{max_batch_size, dims.d[1], dims.d[2], dims.d[3]});
        profile->setDimensions(input_blob_name.c_str(),
                               OptProfileSelector::kMAX, Dims4{max_batch_size, dims.d[1], dims.d[2], dims.d[3]});
        config->addOptimizationProfile(profile);

        nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
        assert(parser != nullptr);
        if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING)))
        {
            this->gLogger.log(ILogger::Severity::kINTERNAL_ERROR, "failed parse the onnx mode");
        }
        // 解析有错误将返回
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }
        this->gLogger.log(ILogger::Severity::kINFO, "successfully parse the onnx mode");
        if (this->useFP16)
            config->setFlag(BuilderFlag::kFP16);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 10 << 20);

        IHostMemory *modelStream = builder->buildSerializedNetwork(*network, *config);
        std::string serializeStr;
        std::ofstream serializeOutputStream;
        serializeStr.resize(modelStream->size());
        memcpy((void *)serializeStr.data(), modelStream->data(), modelStream->size());
        serializeOutputStream.open(enginePath);
        serializeOutputStream << serializeStr;
        serializeOutputStream.close();
        this->gLogger.log(ILogger::Severity::kINFO, "successfully convert onnx to engine");
        delete network;
        delete parser;
        delete config;
        delete builder;
    }
}