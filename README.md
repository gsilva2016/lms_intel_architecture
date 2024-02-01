# Large Language Models (LLM), Large Vision Language Models (LVLM), and Large Multimodal Models (LMM) for Retail Use Cases on Scalable Intel XPU Architecture


## PyTorch* - Intel® Extnesion for PyTorch and Transformers, Intel® Neural Compressor Demonstrations

### Optimized CPU Demonstrations

* Build CPU Docker Image
```
docker build -t qllm-cpu:1.0 -f Dockerfile.qllm-cpu .
```

 * Start CPU Docker Container shell
```
docker run --rm -it -v `pwd`:/savedir --net host qllm-cpu:1.0
```

* Qwen-7B FP32 LLM
```
python run-demo.py --device_map "cpu" --model_name "Qwen/Qwen-7B"
```

* Qwen-7B WOQ INT4 RTN
```
python run-demo.py --device_map "cpu" --model_name "Qwen/Qwen-7B" --use_woq_4bit
```

* Qwen-7B WOQ INT4 GPTQ
```
python run-demo.py --device_map "cpu" --model_name "Qwen/Qwen-7B" --use_woq_4bit --use_gptq
```

* Qwen-VL WOQ INT4 RTN
```
python run-demo.py --device_map "cpu" --model_name "Qwen/Qwen-VL" --use_woq_4bit
```

* Qwen-VL WOQ INT4 GPTQ (Not Supported)
```
python run-demo.py --device_map "cpu" --model_name "Qwen/Qwen-VL" --use_woq_4bit --use_gptq
```

* Video-Llava-7B WOQ INT4 RTN

Please visit TODO

* Video-Llava-7B WOQ INT4 GPTQ

Please visit TODO


### Optimized XPU Demonstrations

* Build XPU Docker Image
```
docker build -t qllm-xpu:1.0 -f Dockerfile.qllm-xpu .
```

* Start XPU Docker Container shell. Replace renderD129 with your GPU from your machine in /dev/dri
```
docker run --rm -it -v `pwd`:/savedir --net host --device /dev/dri/renderD129 qllm-xpu:1.0
```

* Qwen-7B WOQ INT4 RTN
```
python run-demo.py --device_map "xpu" --model_name "Qwen/Qwen-7B" --use_woq_4bit
```

* Qwen-7B WOQ INT4 GPTQ
```
Coming soon
```

* Qwen-VL WOQ INT4 GPTQ
```
Coming soon
```

* Video-Llava-7B WOQ INT4 RTN

Please visit TODO

* Video-Llava-7B WOQ INT4 GPTQ
```
Coming soon
```
