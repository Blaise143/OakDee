import torch.nn as nn
import torchvision
import torch
import onnx
from onnxsim import simplify
import blobconverter

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
print(model)
X = torch.ones((1, 3, 300, 300), dtype=torch.float32)
torch.onnx.export(model, (X),
                  "models/mobilenet.onnx",
                  opset_version=12,
                  do_constant_folding=True,
                  )

onnx_model = onnx.load("models/mobilenet.onnx")
model_simpified, check = simplify(onnx_model)
onnx.save(model_simpified, "models/mobile.onnx")

# CONVERTING TO BLOB FORMAT
blobconverter.from_onnx(
    model="models/mobile.onnx",
    output_dir="models/mobile.blob",
    data_type="FP16",
    shaves=6,
    use_cache=False,
    optimizer_params=[]
)