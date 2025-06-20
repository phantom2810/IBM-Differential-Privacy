import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from opacus import PrivacyEngine


def train_dp_llm(model_name, dataset_path, epsilon, delta, max_grad_norm, output_dir="files"):
    """Fine-tune a language model with differential privacy.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.
    dataset_path : str
        Path to a text dataset file used for training.
    epsilon : float
        Target epsilon for DP-SGD.
    delta : float
        Target delta for DP-SGD.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    output_dir : str, optional
        Directory where the trained model will be saved. Defaults to ``"files"``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("text", data_files={"train": dataset_path})
    tokenized = dataset["train"].map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=1,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, f"dp_{model_name.replace('/', '_')}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir
