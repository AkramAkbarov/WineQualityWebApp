import torch

from model import Winequality

if __name__ == "__main__":
    model = Winequality()
    torch.save(model.state_dict(), "model.pth")
    print("Saved model.pth (replace with your trained weights)")

