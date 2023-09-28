#include "./include/main.h"

int main()
{
    TRTgeneratorV1::TRTgenerator myTRTgenerator;
    myTRTgenerator.setFP16(true);
    myTRTgenerator.createEngine("onnxPath","enginePath",640,640,5,"input");
}