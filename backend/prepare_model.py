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

    filename = "open_clip_model.safetensors"

    script_state = {
        "model_name": "ViT-B-32",
        "pretrained_name": os.path.join(
            destination,
            filename
        ),
        "onnx_model_path": os.path.join(
            destination,
            "clip.onnx"
        ),
        "onnx_model_shapes_path": os.path.join(
            destination,
            "clip_shapes.onnx"
        ),
        "quant_pre_model_path": os.path.join(
            destination,
            "clip_pre_quantized.onnx"
        ),
        "quant_model_path": os.path.join(
            destination,
            "clip_quantized.onnx"
        ),
        "preprocess_path": os.path.join(
            destination,
            "preprocess.onnx"
        ),
        "device": "cpu"
    }

    os.makedirs(destination, exist_ok=True)


    print("Loading Clip...")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        script_state["model_name"],
        pretrained=script_state["pretrained_name"],
        device=script_state["device"]
    )
    clip_model.visual.eval()

    print(clip_model)


    print("Exporting model to onnx format...")

    input_tensor = torch.ones((2, 3, 224, 224), dtype=torch.float32)

    torch.onnx.export(clip_model.visual,
                  (input_tensor),
                  script_state["onnx_model_path"],
                  input_names = ['images'],
                  output_names = ['embeddings'],
                  dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                  external_data=False
                  )


    print("Quantizing model...")
    
    model = onnx.load(script_state["onnx_model_path"])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, script_state["onnx_model_shapes_path"])

    quant_pre_process(script_state["onnx_model_shapes_path"],
        script_state["quant_pre_model_path"],
        skip_optimization=False,
        skip_symbolic_shape=True,
        verbose=3)

    quantize_dynamic(script_state["quant_pre_model_path"],
                                   script_state["quant_model_path"],
                                   weight_type=QuantType.QUInt8)


    if clean_cache:
        print("Cleaning up...")
        os.remove(script_state["pretrained_name"])
    
    os.remove(script_state["onnx_model_path"])
    os.remove(script_state["onnx_model_shapes_path"])
    os.remove(script_state["quant_pre_model_path"])

    if os.path.isfile(script_state["onnx_model_path"] + '.data'):
        os.remove(script_state["onnx_model_path"] + '.data')


    print(f"Model {script_state["pretrained_name"]} quantized to {destination}/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Destination path is missing")

    destination = os.path.join(sys.argv[1], "clip_model")
    hf_download(destination)
    quantized_download(destination)