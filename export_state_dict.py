"""Train script-dən state_dict export etmək üçün nümunə."""

import torch

from model import Winequality

# Öz modelinizi yükləyin:
# model = Winequality()
# ... training ...
# torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    model = Winequality()
    torch.save(model.state_dict(), "model.pth")
    print("Saved model.pth (replace with your trained weights)")

