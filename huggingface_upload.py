from huggingface_hub import HfApi

api = HfApi()

# Create a private repo for your models
api.create_repo(repo_id="your-username/organize-stuff-models", private=True, exist_ok=True)

# Upload the entire folder of ONNX models
api.upload_folder(
    folder_path="backend/clip_model",
    repo_id="Kalamojo/clip_quantized",
    path_in_repo="clip_model"
)

