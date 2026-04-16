import torch

# Charger le checkpoint
checkpoint = torch.load("unet_ct_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # ← CRUCIAL, oubli fréquent

# Dummy input avec la bonne normalisation
dummy = torch.zeros(1, 1, 256, 256)  # valeurs à 0 = (0 - 0.5) / 0.25 = -2 après norm

torch.onnx.export(
    model,
    dummy,
    "base.onnx",
    opset_version=17,          # 17 est bien supporté par ort-web
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch"},
        "output": {0: "batch"},
    },
    export_params=True,
)
print("Export OK")