import torch
import argparse
import re
import time
import json
from transformers import AutoConfig, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.utils import check_min_version
from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig
import intel_extension_for_pytorch as ipex
from PIL import Image
import requests

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--use_woq_4bit", default=False, action="store_true")
parser.add_argument("--use_gptq", default=False, action="store_true")
parser.add_argument("--device_map", nargs="?", default="cpu")
parser.add_argument("--model_name", nargs="?", default="Qwen/Qwen-7B-Chat")
parser.add_argument("--save_dir", nargs="?", default="")

args = parser.parse_args()
device_map = args.device_map
load4bit   = args.use_woq_4bit
save_dir = args.save_dir
use_gptq = args.use_gptq
model_name = args.model_name
quantization_config = None

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if device_map == "cpu" and not load4bit:
  # use cpu only
  print("Loading FP32 model for CPU")
  if "llava-1.5" in model_name:
    from transformers import  AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(model_name)
  else:
    from intel_extension_for_transformers.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, trust_remote_code=True).eval()

elif device_map == "xpu" and load4bit == False:
  # Intel Arc/Flex GPU/XPU device
  print("Loading FP32 model for XPU") 
  from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained(model_name, device_map="xpu", trust_remote_code=True).eval()

elif device_map == "cpu" and load4bit == True:
  from intel_extension_for_transformers.transformers import AutoModelForCausalLM
  if use_gptq == True:
    print("Loading WOQ 4-bit GPTQ model for ", device_map)
    algo_args = {
          "act_order": False,
          "percdamp": 0.01,
          "block_size": 32 ,
          "nsamples": 3,
          "use_max_length": True,
          "pad_max_length": 256,
          }
    quantization_config = WeightOnlyQuantConfig(
          weight_dtype="int4_clip",
          algorithm="GPTQ",
          algorithm_args=algo_args,
          tokenizer=tokenizer)
  else:
    print("Loading WOQ RTN 4-bit model for ", device_map)
    quantization_config = WeightOnlyQuantConfig(
          compute_dtype="fp16" if device_map == "xpu" else "fp32",
          weight_dtype="int4_fullrange",
          group_size=32,
          scale_dtype="fp16" if device_map == "xpu" else "fp32"
          ) #default is A16W4G16
    
  model = AutoModelForCausalLM.from_pretrained(
         model_name,
         device_map=device_map,
         trust_remote_code=True,
         use_llm_runtime=False,
         quantization_config=quantization_config,
         ).eval()

elif device_map == "xpu" and load4bit == True: 
  from intel_extension_for_transformers.transformers.modeling import AutoModelForCausalLM
  print("Loading WOQ 4-bit model for ", device_map)
  quantization_config = WeightOnlyQuantConfig(
          compute_dtype="fp16" if device_map == "xpu" else "fp32",
          weight_dtype="int4_fullrange",
          group_size=32,
          scale_dtype="fp16" if device_map == "xpu" else "fp32"
          ) #default is A16W4G16
  
  model = AutoModelForCausalLM.from_pretrained(
         model_name,
         device_map=device_map,
         trust_remote_code=True,
         use_llm_runtime=False,
         quantization_config=quantization_config,
         ).eval()

if not save_dir == "":
    print(".......Quantized model being saved.....")
    model.save_pretrained(save_dir)
    print("Saving quantized model completed.")

if device_map == "xpu":
  print("Starting IPEX optimize_transformers for XPU...")
  model = ipex.optimize_transformers(model, inplace=True, dtype=torch.float16, woq=True, device="xpu")

print(".........Model loading and optimizations finished..........")
print("Optimized model: ", model)
print("Starting 1st inference. This will take a while to initialize.....")

start_time = time.time()
query = "USER: Why are LLMs large is size?\nASSISTANT:"
image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
if model_name == "Qwen/Qwen-VL-Chat":
  query = tokenizer.from_list_format([
    {'image': image_url},
    {'text': 'Describe this picture.'},
  ])
  input_ids = tokenizer(query, return_tensors="pt").input_ids.to(args.device_map)
elif "llava-1.5" in model_name:
  raw_image = Image.open(requests.get(image_url, stream=True).raw)
  query = "<image>\nUSER: Describe this picture.\nASSISTANT:"
  input_ids = processor(text=query, images=raw_image, return_tensors="pt")
else:
  input_ids = tokenizer(query, return_tensors="pt").input_ids.to(args.device_map)

if "llava-1.5" in model_name:
  response = model.generate(**input_ids, max_length=5048)
  response = processor.batch_decode(response, skip_special_tokens=True)
else:
  response = model.generate(input_ids, max_length=5048)
  response = tokenizer.batch_decode(response, skip_special_tokens=True)

if device_map == "xpu":
  torch.xpu.synchronize()
elapsed_time = time.time() - start_time
print("First inference response: " , response)
print("Seconds taken for first query: ", elapsed_time)

print("Starting inference for second query...")
start_time = time.time()
query = "USER: Explain the pros and cons of AI with Intel architecture.\nASSISTANT:"
if model_name == "Qwen/Qwen-VL-Chat":
  query = tokenizer.from_list_format([
    {'text': 'Explain the pros and cons of AI with Intel architecture.'}
  ])
  input_ids = tokenizer(query, return_tensors="pt").input_ids.to(args.device_map)
elif "llava-1.5" in model_name:
  input_ids = processor(query, None, return_tensors="pt")
else:
  input_ids = tokenizer(query, return_tensors="pt").input_ids.to(args.device_map)

if "llava-1.5" in model_name:
  response = model.generate(**input_ids, max_length=5048)
  response = processor.batch_decode(response, skip_special_tokens=True)
else:
  response = model.generate(input_ids, max_length=5048)
  response = tokenizer.batch_decode(response, skip_special_tokens=True)

elapsed_time = time.time() - start_time
print("Second inference response: " , response)
print("Seconds taken for second query: ", elapsed_time)
if device_map == "xpu":
    torch.xpu.synchronize()
