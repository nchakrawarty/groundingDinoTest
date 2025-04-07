import os
import torch
import yaml
import sys
sys.path.append('I:/Grounding dino/GroundingDINO')
import json
from torch.utils.data import DataLoader
from groundingdino.models import build_model
from dataset import build_dataset
from train_utils import train_one_epoch, evaluate
import argparse

def main(config_path):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    with open('./assets/datasets/labels_my-project-name_2024-11-27-07-41-07.json', 'r') as f:
        annotations = json.load(f)
        print(annotations)

    # Extract configuration
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    output_dir = config["output_dir"]
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset and dataloaders
    print("Loading datasets...")
    train_dataset = build_dataset(dataset_config["train"], dataset_config["images_dir"])
    val_dataset = build_dataset(dataset_config["val"], dataset_config["images_dir"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    # Load model
    print("Initializing model...Model_config", model_config)
    
    # Ensure that 'two_stage_bbox_embed_share' is replaced by 'dec_pred_bbox_embed_share'
    if 'two_stage_bbox_embed_share' in model_config:
        print("Replacing 'two_stage_bbox_embed_share' with 'dec_pred_bbox_embed_share'")
        model_config['dec_pred_bbox_embed_share'] = model_config.pop('two_stage_bbox_embed_share', False)

    # Debugging: Print the updated model_config to verify replacement
    print("Updated model_config:", model_config)

    # Convert dictionary to argparse.Namespace (this makes it an object with attributes)
    model_config_obj = argparse.Namespace(**model_config)

    # Debugging: Print the model_config_obj to ensure correct attributes
    print("Model config object:", model_config_obj)
    
    model = build_model(model_config_obj)
    
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_one_epoch(model, train_loader, optimizer, epoch, device)
        evaluate(model, val_loader, device)

        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

    print("Training complete!")
    print(f"Model configuration: {model_config_obj}")
    print(f"Model architecture: {model}")

if __name__ == "__main__":
    config_path = "./config.yaml"  # Adjust path if needed
    main(config_path)
