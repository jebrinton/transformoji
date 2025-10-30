from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model