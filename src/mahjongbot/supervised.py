import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import MahjongGBDataset
from .model import CNNModel
from .paths import DATA_DIR, PROJECT_ROOT


def train(args):
    checkpoint_dir = Path(args.log_dir) / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MahjongGBDataset(0, args.split_ratio, args.augment, args.data_dir)
    validate_dataset = MahjongGBDataset(args.split_ratio, 1, args.augment, args.data_dir)
    loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    vloader = DataLoader(dataset=validate_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for e in range(args.epochs):
        print("Epoch", e)
        torch.save(model.state_dict(), checkpoint_dir / f"{e}.pkl")
        for i, d in enumerate(loader):
            input_dict = {
                "is_training": True,
                "obs": {
                    "observation": d[0].to(device),
                    "action_mask": d[1].to(device),
                },
            }
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().to(device))
            if i % 128 == 0:
                print("Iteration %d/%d" % (i, len(train_dataset) // args.batch_size + 1), "policy_loss", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Run validation:")
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {
                "is_training": False,
                "obs": {
                    "observation": d[0].to(device),
                    "action_mask": d[1].to(device),
                },
            }
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, d[2].to(device)).sum().item()
        acc = correct / len(validate_dataset)
        print("Epoch", e + 1, "Validate acc:", acc)


def build_parser():
    parser = argparse.ArgumentParser(description="Train the supervised Mahjong policy model.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory containing count.json and preprocessed samples.")
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "log"), help="Directory for checkpoints.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--augment", action="store_true", help="Use augmented samples instead of preprocessed samples.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
