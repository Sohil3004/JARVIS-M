
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from eval_retrieval import CACHE_DIR

BASE_MODEL_NAME = "facebook/bart-large-cnn"
ADAPTER_PATH = "./jarvis-bart-lora"


def load_model(adapter_path=ADAPTER_PATH, base_model_name=BASE_MODEL_NAME):
	"""Load tokenizer + base model + LoRA adapter, then merge for inference."""
	tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=CACHE_DIR)
	base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, cache_dir=CACHE_DIR)

	model = PeftModel.from_pretrained(base_model, adapter_path)
	model = model.merge_and_unload()
	model.eval()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.to(device)
	return tokenizer, model, device


if __name__ == "__main__":
	tokenizer, model, device = load_model()
	print(f"Loaded LoRA adapter from {ADAPTER_PATH} on {device}")

	sample = "summarize: Speaker A: We shipped v2 today. Speaker B: Great, let's monitor errors and user feedback."
	inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=1024).to(device)
	with torch.no_grad():
		outputs = model.generate(**inputs, max_length=64, num_beams=4)
	print("Sample summary:", tokenizer.decode(outputs[0], skip_special_tokens=True))