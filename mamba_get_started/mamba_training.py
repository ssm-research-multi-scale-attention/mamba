import os
import torch
from torch.optim import Adam
from transformers import MambaForCausalLM, AutoTokenizer

sentences = [
    "Do you know {} {}?",
    "{} {} is such a great person!",
    "{} {} is never wrong.",
    "I wish I was more like {} {}",
    "My name is {}, of the {} family",
    "{} is my first name, but my last name is {}"
]

def main():
    """In this simple program, we fine-tune mamba to learn our name and lastname"""
    model_id = "state-spaces/mamba-130m-hf"
    lr, training_steps = 4e-4, 10

    # Creating sentences with the name and lastname we want to use
    name, lastname = "Brian", "Pulfer"
    model_path = f"finetuned_{name}_{lastname}"
    texts = [s.format(name, lastname) for s in sentences]

    # Loading model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaForCausalLM.from_pretrained(model_id).to(device)
    model = torch.compile(model.train())
    
    # Tokenizing inputs
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    
    # Training
    if not os.path.isdir(model_path):
        print(f"Model not found in {model_path}. Training...")
        optimizer = Adam(model.parameters(), lr=lr)
        for step in range(training_steps):
            # Running inputs forward and computing loss
            out = model(**inputs, labels=inputs["input_ids"])
            loss = out["loss"]
            
            # Doing an optimizatino step (SGD with Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss at step {step+1}/{training_steps}: {loss.item():.2f}") 
        
            # Saving the fine-tuned model
            model.save_pretrained(model_path)
    
    # Loading trained model
    model = MambaForCausalLM.from_pretrained(model_path)

    # Generating a sentence with fine-tuned model
    text = f"{name} is my first name,"
    inputs = tokenizer(text, return_tensors="pt")
    model = model.eval()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens = 20)
    print("\n\n", tokenizer.decode(out[0]), "\n\n")
    
    
if __name__ == "__main__":
    main()
