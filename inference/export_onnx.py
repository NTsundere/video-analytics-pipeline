import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)[0]
        return out["boxes"], out["labels"], out["scores"]

print("Loading PyTorch model...")
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
wrapper = Wrapper(model).eval()

dummy_input = torch.randn(1, 3, 320, 320)

print("Exporting to ONNX...")
torch.onnx.export(
    wrapper,
    dummy_input,
    "ssdlite320.onnx",
    opset_version=16,
    input_names=["input"],
    output_names=["boxes", "labels", "scores"],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'boxes': {0: 'num_boxes'}, 
        'labels': {0: 'num_boxes'}, 
        'scores': {0: 'num_boxes'}
    }
)
print("Model successfully exported to ssdlite320.onnx!")