import os
import random
import numpy as np
import torch
from torch.optim import Adam
from transformers import MambaForCausalLM, MambaConfig

MAX_RETRIEVAL = 10_000
ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
            "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
VOCAB = {
    "<Query>": 0,
    "</Query>": 1,
    "<Seq>": 2,
    "</Seq>": 3,
    "<Ans>": 4,
    "</Ans>": 5,
    **{f"<{LETTER}>": 6+i for i, LETTER in enumerate(ALPHABET)},
    **{f"<{str(n)}>": 6+len(ALPHABET)+n for n in range(1, MAX_RETRIEVAL)}
}

REVERSE_VOCAB = {v: k for k, v in VOCAB.items()}


def tokenize(text):
    return [VOCAB[token] for token in text.split()]


def detokenize(tokens):
    return " ".join([REVERSE_VOCAB[token] for token in tokens])


def generate_batch(size, sequence_length=100, min_n_query=None, max_n_query=None, device=None):
    """Generates a batch of a virtual task --> retrieve n-th token
    Example sequences:
        <Query> <2> </Query> <Seq> <A> <B> <C> </Seq> <Ans> <B> </Ans>
        <Query> <4> </Query> <Seq> <C> <B> <B> <A> <C> <A> <A> </Seq> <Ans> <A> </Ans>
    Returns:
        Dictionary with input_ids and attention_mask
    """
    min_n_query, max_n_query = min_n_query or 1, max_n_query or sequence_length - 8
    n_query = np.random.randint(min_n_query, max_n_query+1, (size, ))
    batch, texts = [], []
    for n in n_query:
        #  Generating sequence
        query = f"<Query> <{n}> </Query>"
        sequence = " ".join(
            [f"<{random.choice(ALPHABET)}>" for _ in range(sequence_length - 8)])
        answer = sequence.split()[n-1]
        answer = f"<Ans> {answer} </Ans>"
        text = f"{query} <Seq> {sequence} </Seq> {answer}"
        texts.append(text)
        batch.append(tokenize(text))
    batch = torch.tensor(batch, device=device)
    return {
        "input_ids": batch,
        "attention_mask": torch.ones_like(batch),
        "texts": texts
    }


def main():
    """In this simple program, we fine-tune a small mamba model to a virtual retrieval task"""
    # Defining model and training parameters
    model_id = "state-spaces/mamba-130m-hf"
    model_path = model_id.split("/")[-1] + "-virtual-retrieval"
    lr, training_steps, batch_size, sequence_length = 4e-4, 10, 16, 10

    # Loading model (same architecture but new parameters)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaForCausalLM(MambaConfig.from_pretrained(model_id)).to(device)

    #  Scripting the model (runs faster)
    model = torch.compile(model.train())

    #  Training
    if not os.path.isdir(model_path):
        print(f"Model not found in {model_path}. Training...")
        optimizer = Adam(model.parameters(), lr=lr)
        for step in range(training_steps):
            # Running inputs forward and computing loss
            batch = generate_batch(batch_size, sequence_length, device=device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"])
            loss = out["loss"]

            # Doing an optimizatino step (SGD with Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss at step {step+1}/{training_steps}: {loss.item():.2f}")

            # Saving the fine-tuned model
            model.save_pretrained(model_path)

    # Evaluation (TODO)
    model = MambaForCausalLM.from_pretrained(model_path).to(device)
    model = torch.compile(model.eval())
    sequence = "<Query> <2> </Query> <Seq> <A> <B> <C> </Seq> <Ans>"
    ids = torch.tensor([tokenize(sequence)], device=device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, do_sample=False, max_new_tokens=2)
    decoded = detokenize(out[0].tolist())
    print("Original sequence:\n\t", sequence)
    print("Generated sequence:\n\t", decoded)


if __name__ == "__main__":
    main()
