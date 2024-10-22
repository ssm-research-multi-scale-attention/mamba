import torch
from transformers import AutoModelForCausalLM, AutoTokenizer # or MambaForCausalLM


def main():
    """Inference code that uses mamba to continue one sentence"""
    
    # Define the device (GPU if possible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get model and associated tokenizer
    # Available mamba models are here --> https://huggingface.co/models?search=state-spaces/mamba
    model_id = "state-spaces/mamba-130m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Setting the model in evaluation mode and compiling it
    model = model.eval()
    model = torch.compile(model) # Makes it run much faster

    # Tokenize the input text
    inputs = tokenizer(
        "The quick brown",
        return_tensors="pt"
    ).to(device)

    # Run inference (no need for gradients)
    # Better yet is "with torch.inference_mode()" and 
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=9)

    # Decode output back to text
    out_ids = out[0]
    generated_text = tokenizer.decode(out_ids)
    print("\n\n\n", generated_text, "\n\n\n")

if __name__ == "__main__":
    main()

"""
RESOURCES:
    HuggingFace tutorial --> https://huggingface.co/learn/nlp-course/chapter1/1

"""