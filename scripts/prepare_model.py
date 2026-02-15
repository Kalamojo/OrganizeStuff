import os
import sys

from huggingface_hub import hf_hub_download

import torch
import open_clip

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process


def hf_download(destination: str):
    repo_id = "laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K"
    filename = "open_clip_model.safetensors"
    os.makedirs(destination, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=destination)
    print(f"Model {repo_id} downloaded to {destination}/")

def quantized_download(destination: str, clean_cache: bool = True):
    filename = "open_clip_model.safetensors"
    model_name = "ViT-B-32"
    pretrained_path = os.path.join(destination, filename)

    # Common paths
    os.makedirs(destination, exist_ok=True)
    device = "cpu"

    print("Loading OpenCLIP model...")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_path,
        device=device
    )
    model.eval()

    # --- Vision Model ---
    print("\n--- Processing Vision Model ---")
    vision_model = model.visual
    vision_onnx_path = os.path.join(destination, "clip_vision.onnx")
    vision_quant_pre_path = os.path.join(destination, "clip_vision_pre_quant.onnx")
    vision_quant_path = os.path.join(destination, "clip_vision_quantized.onnx")

    print("Exporting vision model to ONNX...")
    vision_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        vision_model,
        vision_input,
        vision_onnx_path,
        input_names=['image'],
        output_names=['image_features'],
        dynamic_axes={'image': {0: 'batch'}},
        opset_version=14
    )

    print("Quantizing vision model...")
    quant_pre_process(vision_onnx_path, vision_quant_pre_path)
    quantize_dynamic(vision_quant_pre_path, vision_quant_path, weight_type=QuantType.QUInt8)
    print(f"Vision model quantized and saved to {vision_quant_path}")


    # --- Text Model ---
    print("\n--- Processing Text Model ---")
    text_onnx_path = os.path.join(destination, "clip_text.onnx")
    text_quant_pre_path = os.path.join(destination, "clip_text_pre_quant.onnx")
    text_quant_path = os.path.join(destination, "clip_text_quantized.onnx")

    print("Initializing tokenizer and exporting text model to ONNX...")
    tokenizer = open_clip.get_tokenizer(model_name)
    dummy_text = tokenizer(["a diagram", "a dog", "a cat"])
    
    class TextEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, text):
            return self.model.encode_text(text)

    text_encoder = TextEncoder(model)
    text_encoder.eval()
    
    torch.onnx.export(
        text_encoder,
        dummy_text,
        text_onnx_path,
        input_names=['text'],
        output_names=['text_features'],
        dynamic_axes={'text': {0: 'batch', 1: 'sequence'}},
        opset_version=14
    )
    
    print("Quantizing text model...")
    quant_pre_process(text_onnx_path, text_quant_pre_path)
    quantize_dynamic(text_quant_pre_path, text_quant_path, weight_type=QuantType.QUInt8)
    print(f"Text model quantized and saved to {text_quant_path}")


    # --- Cleanup ---
    if clean_cache:
        print("\nCleaning up intermediate files...")
        os.remove(pretrained_path)
        os.remove(vision_onnx_path)
        os.remove(vision_quant_pre_path)
        os.remove(text_onnx_path)
        os.remove(text_quant_pre_path)
        print("Cleanup complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Destination path is missing")

    model_dest_dir = os.path.join(sys.argv[1], "clip_model")
    hf_download(model_dest_dir)
    quantized_download(model_dest_dir)