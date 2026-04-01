from transformers import AutoImageProcessor, AutoModel
model_id = "nomic-ai/nomic-embed-vision-v1.5"

proc = AutoImageProcessor.from_pretrained(model_id)
mdl  = AutoModel.from_pretrained(model_id, trust_remote_code=True)

proc.save_pretrained("./nomic_v15_local")
mdl.save_pretrained("./nomic_v15_local")
print("Saved local copy at ./nomic_v15_local")
