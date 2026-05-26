import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import argparse
import torch
import onnx
from src.model.clip_proto import CLIPZeroShotNetwork


def export_to_onnx(config, checkpoint_path=None, output_path="deploy/model.onnx", opset_version=17):
    device = torch.device("cpu")

    model = CLIPZeroShotNetwork(config, device=device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint — exporting pure zero-shot model")

    model.eval()

    batch_size = 1
    dummy = torch.randn(batch_size, 3, config["data"]["image_size"], config["data"]["image_size"])

    model.config["model"]["zero_shot_only"] = True

    with torch.no_grad():
        logits, features, _ = model(dummy)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits", "image_features"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "image_features": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported -> {output_path}")

    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    dummy_np = dummy.numpy()
    outputs = session.run(None, {"input": dummy_np})
    print(f"ONNX inference OK. Logits shape: {outputs[0].shape}")

    import time
    warmup = 10
    for _ in range(warmup):
        session.run(None, {"input": dummy_np})
    trials = 100
    start = time.perf_counter()
    for _ in range(trials):
        session.run(None, {"input": dummy_np})
    elapsed = (time.perf_counter() - start) / trials
    print(f"Avg inference time: {elapsed*1000:.2f} ms (batch=1)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="deploy/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    export_to_onnx(cfg, args.checkpoint, args.output, args.opset)
