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

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=destination
    )

    print(f"Model {repo_id} downloaded to {destination}/")

def quantized_download(
        destination: str, 
        clean_cache: bool = True
    ):

    # This function is now based on your provided working script.
    # I have extended it to include the text model.

    model_name = "ViT-B-32"
    pretrained_name = os.path.join(destination, "open_clip_model.safetensors")
    device = "cpu"

    os.makedirs(destination, exist_ok=True)

    print("Loading Clip model...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_name,
        device=device
    )
    clip_model.eval()

    # --- Vision Model Processing (from your script) ---
    print("\n--- Processing Vision Model ---")
    vision_onnx_path = os.path.join(destination, "clip_vision.onnx")
    vision_shapes_path = os.path.join(destination, "clip_vision_shapes.onnx")
    vision_pre_quant_path = os.path.join(destination, "clip_vision_pre_quant.onnx")
    vision_quant_path = os.path.join(destination, "clip_vision_quantized.onnx")

    print("Exporting vision model to ONNX...")
    # Using torch.ones like in your script
    input_tensor = torch.ones((1, 3, 224, 224), dtype=torch.float32)

    torch.onnx.export(clip_model.visual,
                  (input_tensor),
                  vision_onnx_path,
                  input_names=['image'],
                  output_names=['image_features'],
                  dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                  external_data=False
                  )

    print("Quantizing vision model...")
    model_vision = onnx.load(vision_onnx_path)
    model_vision = onnx.shape_inference.infer_shapes(model_vision)
    onnx.save(model_vision, vision_shapes_path)

    quant_pre_process(
        vision_shapes_path,
        vision_pre_quant_path,
        skip_optimization=False,
        skip_symbolic_shape=True, # Critical parameter from your code
        verbose=3
    )

    quantize_dynamic(
        vision_pre_quant_path,
        vision_quant_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Vision model quantized to {vision_quant_path}")


    # --- Text Model Processing (extending your script) ---
    print("\n--- Processing Text Model ---")
    text_onnx_path = os.path.join(destination, "clip_text.onnx")
    text_shapes_path = os.path.join(destination, "clip_text_shapes.onnx")
    text_pre_quant_path = os.path.join(destination, "clip_text_pre_quant.onnx")
    text_quant_path = os.path.join(destination, "clip_text_quantized.onnx")

    class TextEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, text):
            return self.model.encode_text(text)

    text_encoder = TextEncoder(clip_model)
    text_encoder.eval()

    print("Exporting text model to ONNX...")
    tokenizer = open_clip.get_tokenizer(model_name)
    dummy_text = tokenizer(["a dog", "a cat"])

    torch.onnx.export(
        text_encoder,
        dummy_text,
        text_onnx_path,
        input_names=['text'],
        output_names=['text_features'],
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        external_data=False
    )

    print("Quantizing text model...")
    model_text = onnx.load(text_onnx_path)
    model_text = onnx.shape_inference.infer_shapes(model_text)
    onnx.save(model_text, text_shapes_path)

    quant_pre_process(
        text_shapes_path,
        text_pre_quant_path,
        skip_optimization=False,
        skip_symbolic_shape=True, # Using same successful parameter
        verbose=3
    )
    quantize_dynamic(
        text_pre_quant_path,
        text_quant_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Text model quantized to {text_quant_path}")


    # --- Cleanup ---
    if clean_cache:
        print("\nCleaning up intermediate files...")
        files_to_remove = [
            pretrained_name,
            vision_onnx_path, vision_shapes_path, vision_pre_quant_path,
            text_onnx_path, text_shapes_path, text_pre_quant_path
        ]
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)
        print("Cleanup complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Destination path is missing")

    destination = os.path.join(sys.argv[1], "clip_model")
    hf_download(destination)
    quantized_download(destination)
