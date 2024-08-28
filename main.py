import os
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset
import models
import uuid
import logging
from datetime import datetime
import sys


def get_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(config):
    log_id = str(uuid.uuid4())
    log_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{log_id}"
    log_filename = f"{log_id}.log"
    log_folder = config.get('log_dir', 'logs')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    logging.basicConfig(filename=f"{log_folder}/{log_filename}", level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f"Log ID: {log_id}")
    logging.info(f"Date: {datetime.now()}")
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info("-----------------------------")

    # write to file <uuid>.config.yaml
    config_folder = config.get('save_yaml_config_dir', 'config')
    if not os.path.exists(config_folder):
        os.makedirs(config_folder)
    with open(f"{config_folder}/{log_id}.yaml", 'w') as f:
        yaml.dump(config, f)

    return log_id


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc


def validate(model, test_loader, criterion, device, best_3_model_dict={}, epoch=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(test_loader)
    val_acc = correct / total

    # update best_3_model_dict if model is in top 3
    # the format is {"epoch__val_acc":
    # {
    #     "model_dict": model.state_dict(),
    #     "val_acc": val_acc,
    #     "epoch": epoch
    # }}
    if len(best_3_model_dict) < 3:
        best_3_model_dict[f"{epoch:04d}__{val_acc}"] = {
            "model_dict": model.state_dict(),
            "val_acc": val_acc
        }
    else:
        # get the key with minimum val_acc
        min_key = min(best_3_model_dict,
                      key=lambda x: best_3_model_dict[x]["val_acc"])

        if val_acc > best_3_model_dict[min_key]["val_acc"]:
            del best_3_model_dict[min_key]
            best_3_model_dict[f"{epoch:04d}__{val_acc}"] = {
                "model_dict": model.state_dict(),
                "val_acc": val_acc
            }

    return val_loss, val_acc


def main():
    config = get_config()
    print(config)
    log_id = setup_logger(config)
    print(f"Log ID: {log_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataset(config)

    model = models.get_model_methods(config)(
        num_classes=len(train_loader.dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    num_epochs = config['num_epochs']
    best_3_model_dict = {}

    for epoch in tqdm(range(num_epochs), desc="Training", total=num_epochs):
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, best_3_model_dict, epoch)

        log_message = f"Epoch [{epoch+1}/{num_epochs}]\n"
        log_message += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n"
        log_message += f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n"
        log_message += "-----------------------------"
        print(log_message)

        logging.info(
            f"Epoch [{epoch+1:04d}/{num_epochs:04d}] | Train Loss: {train_loss:.4f}")
        logging.info(
            f"EPOCH [{epoch+1:04d}/{num_epochs:04d}] | Train Accuracy: {train_acc:.4f}")
        logging.info(
            f"EPOCH [{epoch+1:04d}/{num_epochs:04d}] | Validation Loss: {val_loss:.4f}")
        logging.info(
            f"EPOCH [{epoch+1:04d}/{num_epochs:04d}] | Validation Accuracy: {val_acc:.4f}")

    logging.info("Training completed!")

    # save the best 3 models
    # save_model_weights_dir in config
    save_weight_folder = config.get('save_model_weights_dir', 'models')
    if not os.path.exists(save_weight_folder):
        os.makedirs(save_weight_folder)

    for i, key in enumerate(best_3_model_dict):
        torch.save(best_3_model_dict[key]["model_dict"],
                   f"{save_weight_folder}/{log_id}__{i+1}__{key}.pth")
        logging.info(
            f"Model {i+1} saved to {save_weight_folder}/{log_id}__{i+1}__{key}.pth")


if __name__ == "__main__":
    main()
