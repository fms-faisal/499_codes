# Automating the Optimal Privacy Budget Selection for Differential Privacy in Federated Learning Environments

This repository contains the official implementation for the research paper on dynamically and automatically selecting the optimal privacy budget (epsilon, ε) in a Differentially Private Federated Learning (DP-FL) system. Our method removes the need for manual tuning by introducing an **epsilon-aware strategy** that adapts the privacy-utility trade-off in real-time.

---

## 📖 Table of Contents
- [The Challenge: Balancing Privacy and Utility](#-the-challenge-balancing-privacy-and-utility)
- [System Architecture](#-system-architecture)
- [The Epsilon-Aware Strategy](#-the-epsilon-aware-strategy)
- [How It Works: The Algorithm](#-how-it-works-the-algorithm)
- [Getting Started](#-getting-started)
- [Key Results](#-key-results)
- [Citation](#-citation)

---

## 🎯 The Challenge: Balancing Privacy and Utility

Federated Learning (FL) enables collaborative machine learning without sharing raw data. When combined with Differential Privacy (DP), it offers strong privacy guarantees. However, this introduces the critical **privacy budget (ε)** parameter.

-   **Low ε**: Stronger privacy, but high noise can hurt model accuracy.
-   **High ε**: Better accuracy, but weaker privacy guarantees.

Finding the optimal ε is crucial, but manual tuning is inefficient and doesn't adapt to changing conditions during training. This project solves that problem by creating a system that **autonomously selects the best ε** in each round of federated training.

---

## 🏛️ System Architecture

Our system uses a standard client-server FL architecture, built with the following key technologies:

-   **Federated Learning Framework**: **Flower (flwr)** to manage communication and aggregation.
-   **Differential Privacy**: **Opacus** to inject noise and provide DP guarantees during client-side training.
-   **Deep Learning**: **PyTorch** for building and training neural network models.

The core logic is orchestrated by `auto_DP_FL.py`, which simulates the entire federated network and implements our dynamic epsilon selection strategy.

---

## 🤖 The Epsilon-Aware Strategy

Our core innovation is an **adaptive, epsilon-aware strategy** that intelligently selects the best privacy budget after each round of federated training.

Here is the high-level workflow:

1.  **Client Training**: Clients train their local models and send updates to the server.
2.  **Aggregation & Cloning**: The server aggregates the updates using FedAvg. It then creates multiple clones of this new global model.
3.  **Candidate Evaluation**: Each model clone is assigned a different candidate ε from a predefined list (e.g., `[0.5, 1.0, 2.0, 5.0]`).
4.  **Proxy Training**: The server trains each clone for a few epochs on a small, representative proxy dataset. This step is crucial for evaluating how the model performs under different privacy constraints.
5.  **Optimal Epsilon Selection**: The server calculates an "optimal budget score" for each clone based on its performance (e.g., F1-score) and its assigned ε.
6.  **Distribution**: The ε that yields the highest score is selected as the optimal budget for the *next* round of federated training and is sent back to the clients along with the updated global model.

This creates a closed-loop system that continuously adapts the privacy budget based on empirical performance.

### System Workflow Diagram

![System Workflow Diagram](https://i.imgur.com/BATjAzL.png)

---

## ⚙️ How It Works: The Algorithm

The selection of the optimal epsilon is based on a weighted objective function that balances model performance (F1 Score) and the privacy budget (epsilon).

**Algorithm: Calculate Optimal Epsilon**

1.  **Define Objective Function**: Create a function to balance utility and privacy.
2.  **Normalize Metrics**:
    -   Normalize the F1 scores of all model clones to a `[0, 1]` range.
    -   Normalize the candidate epsilon values to a `[0, 1]` range.
3.  **Assign Weights**: Define weights `w1` (for performance) and `w2` (for privacy) based on the specific requirements of the task.
4.  **Combine Metrics**: Calculate the `optimal_budget_score` for each candidate:
    ```
    score = (w1 * Normalized_F1) - (w2 * Normalized_Epsilon)
    ```
5.  **Select Best Epsilon**: The epsilon corresponding to the highest `optimal_budget_score` is chosen.
6.  **Return**: The optimal epsilon for the next round.

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   PyTorch
-   Flower (flwr)
-   Opacus
-   NumPy, Pandas, Scikit-learn

### Installation

Clone the repository and install the required dependencies.
```bash
git clone [https://github.com/fms-faisal/Auto-Optimal-Privacy-Budget-DP-FL.git](https://github.com/fms-faisal/Auto-Optimal-Privacy-Budget-DP-FL.git)
cd Auto-Optimal-Privacy-Budget-DP-FL
pip install torch torchvision pandas numpy scikit-learn opacus flwr tqdm Pillow psutil
```

### Running the Simulation

To start the federated learning process with automatic epsilon selection, run the main script:
```bash
python auto_DP_FL.py
```
During training, it will:

- Adjust the fraction of data used by each client per round  
- Apply differential privacy with dynamic epsilon selection  
- Log training loss, accuracy, time, and memory usage  
- Save the final model as `dynamic_epsilon_final.pth`  
- Record progress and metrics in `training_log_final.txt`

---

## 📊 Key Results

Our experiments show that the system effectively balances the privacy-utility trade-off:

-   **Dynamic Selection**: The optimal epsilon dynamically changes from round to round, typically converging to values between **0.5 and 2.0**, demonstrating the system's ability to adapt.
-   **Peak Performance**: The optimal budget score consistently peaked at an epsilon of **ε = 2.0** in early rounds, achieving the best balance of high accuracy and strong privacy.
-   **Efficiency**: The epsilon selection process adds a consistent and manageable computational overhead, making it practical for real-world applications.

---

## 📜 Citation

If you use this work in your research, please cite our paper:

```bibtex
The BibTeX entry will be added here once the paper is published.
