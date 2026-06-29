import torch
import torch.nn as nn
import yaml
from pathlib import Path
from prettytable import PrettyTable

# --- 1. Import your model ---
from models.model import DynaModel


def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration file from the specified path."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module):
    """
    Calculates and prints the total number of trainable parameters for a PyTorch model.
    """
    table = PrettyTable(["Module Name", "Parameters"])
    total_params = 0

    print("--- Model Architecture Overview ---")
    # Print a simplified model structure to understand the modules
    print(model)
    print("-----------------------------------\n")

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        params = parameter.numel()
        table.add_row([name, f"{params:,}"])
        total_params += params

    print("--- Detailed List of Learnable Parameters ---")
    print(table)
    print("---------------------------------------------\n")

    total_params_in_m = total_params / 1_000_000
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Total Trainable Parameters (in Millions): {total_params_in_m:.2f}M")

    return total_params


if __name__ == '__main__':
    PROJECT_ROOT = Path("/home/zdy/Project2/")
    CONFIG_PATH = PROJECT_ROOT / "config.yaml"

    try:
        # 2. Load your configuration file
        print(f"Loading configuration from {CONFIG_PATH}...")
        config = load_config(CONFIG_PATH)
        print("Configuration loaded successfully.")

        # 3. Initialize your DynaModel
        print("Initializing DynaModel...")
        model = DynaModel(config)
        print("Model initialized successfully.")

        # 4. Call the function to calculate and print the parameter count
        print("\nStarting parameter count calculation...\n")
        count_parameters(model)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure your config.yaml file is located in the project's root directory.")
    except KeyError as e:
        print(f"\nError: Missing a required key in the configuration file: {e}")
        print("Please check that your config.yaml file is complete and contains all parameters needed for model initialization.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")