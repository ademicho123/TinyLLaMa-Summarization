import matplotlib.pyplot as plt
from typing import List


def plot_loss(losses: List[float]):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')  # Add markers
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)  # Add grid
    plt.show()
