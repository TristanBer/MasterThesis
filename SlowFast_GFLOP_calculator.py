import torch
from fvcore.nn import FlopCountAnalysis

model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
model.eval()
slow = torch.randn(1, 3, 8, 224, 224)
fast = torch.randn(1, 3, 32, 224, 224)
fca = FlopCountAnalysis(model, ([slow, fast],))
fca.unsupported_ops_warnings(False)
fca.uncalled_modules_warnings(False)
print(f"SlowFast R50: {fca.total() / 1e9:.2f} GMACs")