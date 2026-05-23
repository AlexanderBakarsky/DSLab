from transformers import AutoModelForCausalLM
import os

model_name = "swiss-ai/Apertus-8B-2509"
folder = os.getenv("SCRATCH_DIR")

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=folder)

with open("Apertus_weight_layer_names.txt", "w") as f:
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            f.write(name + "\n")
