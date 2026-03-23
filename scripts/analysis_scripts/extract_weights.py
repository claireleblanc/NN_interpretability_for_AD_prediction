import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import re

def load_state_dict(path):
    state = torch.load(path, map_location="cpu")

    # common checkpoint layouts
    if isinstance(state, dict):
        if "state_dict" in state:
            return state["state_dict"]
        if "model_state_dict" in state:
            return state["model_state_dict"]

    return state

def extract_weights_from_checkpoint(state_dict,
                                    conv_key="conv1.weight",
                                    linear_key="linear1.weight"):
    w1 = state_dict[conv_key].detach().cpu().numpy()
    w2 = state_dict[linear_key].detach().cpu().numpy()

    # reshape conv to (kernel_positions, 20) as in your code
    width = w1.shape[0] * w1.shape[2]
    conv_weights = w1.reshape(width, 20)

    linear_weights = w2.reshape(-1)

    return conv_weights, linear_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with model metadata")
    parser.add_argument("--kernel_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", default="weights_arrays.npz")

    parser.add_argument("--conv_key", default="conv1.weight")
    parser.add_argument("--linear_key", default="linear1.weight")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df['seed'] = [int(re.search("seed(\d+)", model.removeprefix("../../"))[1]) for model in  df["model_name"]]

    if "kernel_size" not in df.columns:
        raise ValueError("CSV must contain kernel_size column")
    

    if "model_name" not in df.columns:
        raise ValueError("CSV must contain model_name column")

    df = df[(df["kernel_size"] == args.kernel_size) & (df["seed"] == args.seed)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No rows found with kernel_size={args.kernel_size}")

    conv_list = []
    linear_list = []

    print(f"Loading {len(df)} models...")

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        model_path = getattr(row, "model_name")

        # normalize path
        if model_path.endswith(".pth"):
            path = model_path
        else:
            path = model_path + ".pth"

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        state_dict = load_state_dict(path)

        conv_w, lin_w = extract_weights_from_checkpoint(
            state_dict,
            conv_key=args.conv_key,
            linear_key=args.linear_key,
        )

        conv_list.append(conv_w)
        linear_list.append(lin_w)

    conv_arr = np.stack(conv_list, axis=0)
    linear_arr = np.stack(linear_list, axis=0)

    np.savez(
        args.out,
        conv_weights=conv_arr,
        linear_weights=linear_arr,
        kernel_size=args.kernel_size,
        n_models=len(df),
    )

    print(f"\nSaved:")
    print(f"  {args.out}")
    print(f"  conv shape: {conv_arr.shape}")
    print(f"  linear shape: {linear_arr.shape}")


if __name__ == "__main__":
    main()