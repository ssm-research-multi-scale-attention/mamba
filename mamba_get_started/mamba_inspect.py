import torch
from transformers import AutoModelForCausalLM


def main():
    """Simple code to inspect what's a Mamba model really composed of.
    For a better understanding, try debugging the model during inference."""
    # Loading model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "state-spaces/mamba-130m-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Showing model configuration
    print(model.config)
    
    # Showing model
    print(model)

    # Showing all modules in a block
    print(model.backbone.layers[0].mixer.x_proj.weight)

  
if __name__ == "__main__":
    main()
