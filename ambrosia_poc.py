# Martin Djamongue
#Final project for RBIF102.molecular biology
#Date: 10/19/2026

"""
Ambrosia In Silico Reprogramming Model - Proof of Concept (POC)
================================================================
Based on NewLimit (2025): "In silico design of epigenetic reprogramming payloads"
https://newlimit.github.io/research_site_2025_isr/


Architecture:
1. TF Embeddings via Protein Language Models (pLMs): Represents transcription factors (TFs) as dense vector representations.
2. Permutation-Invariant Set Aggregation (alpha): Aggregates TF embeddings in payload S = (s^{(1)}, ..., s^{(K)}) using sum operation.
3. Neural Network Predictor with MC Dropout (f_theta): 3-layer MLP mapping set embedding xi -> cell state prediction y.
4. Bayesian Uncertainty Estimation: Monte Carlo Dropout (N=100 forward passes) estimating posterior mean mu_y(S) and std sigma_y(S).
5. Payload Design Strategies:
   - Constrained Design: Upper Confidence Bound (UCB) ranking: a_UCB(S) = mu_y(S) + beta * sigma_y(S).
   - Unconstrained Design: Metropolis-Hastings Markov Chain Monte Carlo (MCMC) sampling p_theta(S | y > tau).
"""
#packages and libraries used
"""
how to install packages and libraries used in this project
1. pip install torch
2. pip install numpy
3. pip install scikit-learn
PS: use "pip3 install <package>" instead of "pip install <package>" if your current version of python is 3.10 or above
"""
"""
How to run this program:
1. Open your terminal
2. Navigate to the directory where this file is located
3. Make sure you have the required packages and libraries installed (see above)
4. chmod +x ambrosia_poc.py ( to make the file executable)
5. Run the command "python3 ambrosia_poc.py" 

PS: if running in Vscode and other advanced code editors,
the file can also be run by clicking the play button in the top right corner of the editor
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from typing import List, Dict, Tuple, Set

# Fix seeds for exact reproducibility
torch.manual_seed(100)
np.random.seed(100)


class TFEmbeddingLibrary:
    """
    Manages dense protein language model (pLM) embeddings (e.g., ESM-2 / ProtT5)
    for a target library of Transcription Factors (TFs).
    """
    def __init__(self, tf_names: List[str], embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.tf_names = tf_names
        self.embeddings: Dict[str, torch.Tensor] = {}
        
        # Generate normalized pLM representations
        for name in tf_names:
            seed = sum(ord(c) for c in name)
            rng = torch.Generator().manual_seed(seed)
            emb = torch.randn(embedding_dim, generator=rng)
            emb = emb / torch.norm(emb)
            self.embeddings[name] = emb

    def get_embedding(self, tf_name: str) -> torch.Tensor:
        return self.embeddings[tf_name]

    def aggregate_payload(self, payload: List[str]) -> torch.Tensor:
        """
        Permutation-invariant sum aggregation operator:
        xi_S = alpha(s^(1), ..., s^(K)) = sum_{j=1}^K s^(j)
        """
        embs = [self.embeddings[tf] for tf in payload]
        stacked = torch.stack(embs, dim=0) # Shape: (K, embedding_dim)
        return torch.sum(stacked, dim=0)   # Shape: (embedding_dim,)


class AmbrosiaModel(nn.Module):
    """
    Ambrosia Neural Network with Monte Carlo Dropout for Uncertainty Estimation.
    Maps set representation xi -> predicted cell state y.
    """
    def __init__(self, embedding_dim: int = 512, hidden_dim1: int = 512, hidden_dim2: int = 128, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return self.net(xi)

    def predict_uncertainty(self, xi: torch.Tensor, num_mc_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference: keeps dropout active during evaluation
        to sample N forward passes and compute empirical posterior mean and std.
        """
        self.train() # Enable dropout during inference
        preds = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                out = self.forward(xi)
                preds.append(out.squeeze(-1))
        preds = torch.stack(preds, dim=0) # Shape: (num_mc_samples, batch_size)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std


class AdditiveBaseline:
    """
    Additive Baseline Model: Predicts payload effect as sum of observed individual TF effects.
    f_Additive(S) = sum_{s in S} y_s
    """
    def __init__(self):
        self.single_tf_effects: Dict[str, float] = {}

    def fit(self, payloads: List[List[str]], targets: np.ndarray):
        for p, y in zip(payloads, targets):
            if len(p) == 1:
                self.single_tf_effects[p[0]] = float(y)

    def predict(self, payloads: List[List[str]]) -> np.ndarray:
        preds = []
        default_val = float(np.mean(list(self.single_tf_effects.values()))) if self.single_tf_effects else 0.0
        for p in payloads:
            val = sum(self.single_tf_effects.get(tf, default_val) for tf in p)
            preds.append(val)
        return np.array(preds, dtype=np.float32)


class MeanBaseline:
    """
    Mean Baseline Model: Predicts mean of training set targets.
    """
    def __init__(self):
        self.mean_val = 0.0

    def fit(self, targets: np.ndarray):
        self.mean_val = float(np.mean(targets))

    def predict(self, n_samples: int) -> np.ndarray:
        return np.full((n_samples,), self.mean_val, dtype=np.float32)


# Metrics
def compute_cse(y_true: np.ndarray, y_pred: np.ndarray, y_ctrl: float = 0.0) -> float:
    """Control-Scaled Error (CSE): MSE(model) / MSE(control)."""
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_ctrl = np.mean((y_true - y_ctrl) ** 2)
    return float(mse_model / (mse_ctrl + 1e-8))


def compute_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson Correlation Coefficient."""
    if np.std(y_pred) < 1e-7 or np.std(y_true) < 1e-7:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def compute_auprc(y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 75.0) -> float:
    """Area Under Precision-Recall Curve for hit detection."""
    from sklearn.metrics import precision_recall_curve, auc
    threshold = np.percentile(y_true, percentile)
    binary_true = (y_true >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(binary_true, y_pred)
    return float(auc(recall, precision))


def generate_combinatorial_screen_dataset(tf_lib: TFEmbeddingLibrary, n_samples: int = 500) -> Tuple[List[List[str]], np.ndarray]:
    """
    Simulates a combinatorial single-cell reprogramming dataset (similar to K562 / NLMT screens)
    with realistic non-additive synergy (e.g. Yamanaka OSK factors).
    """
    tf_list = tf_lib.tf_names
    payloads = []
    payload_set = set()

    # Include all 1-TF payloads
    for tf in tf_list:
        payloads.append([tf])
        payload_set.add((tf,))

    # Cap target samples to avoid sampling exhaustion on 1, 2, and 3-TF combinations
    # For 15 TFs: C(15,1)+C(15,2)+C(15,3) = 15 + 105 + 455 = 575 maximum possible unique payloads
    max_combos = len(tf_list) + (len(tf_list) * (len(tf_list) - 1)) // 2 + (len(tf_list) * (len(tf_list) - 1) * (len(tf_list) - 2)) // 6
    target_samples = min(n_samples, max_combos - 25)

    # Generate 2-TF and 3-TF combinations
    while len(payloads) < target_samples:
        k = int(np.random.choice([2, 3], p=[0.6, 0.4]))
        combo = [str(x) for x in sorted(np.random.choice(tf_list, size=k, replace=False))]
        combo_tuple = tuple(combo)
        if combo_tuple not in payload_set:
            payload_set.add(combo_tuple)
            payloads.append(combo)

    # Base individual effects
    base_effects = {tf: np.random.normal(0.01, 0.02) for tf in tf_list}
    base_effects['OCT4'] = 0.05
    base_effects['SOX2'] = 0.04
    base_effects['KLF4'] = 0.03
    base_effects['MYC'] = 0.02

    targets = []
    for p in payloads:
        add_val = sum(base_effects[tf] for tf in p)
        # Non-additive combinatorial synergy
        synergy = 0.0
        p_set = set(p)
        if {'OCT4', 'SOX2', 'KLF4'}.issubset(p_set):
            synergy += 0.18 # Strong Yamanaka OSK synergy
        elif {'OCT4', 'SOX2'}.issubset(p_set):
            synergy += 0.07
        elif {'MYC', 'KLF4'}.issubset(p_set):
            synergy += 0.05
            
        noise = np.random.normal(0, 0.008)
        y = add_val + synergy + noise
        targets.append(y)

    return payloads, np.array(targets, dtype=np.float32)


# Generative Payload Design via Metropolis-Hastings MCMC
def sample_payloads_mcmc(
    model: AmbrosiaModel,
    tf_lib: TFEmbeddingLibrary,
    tau: float,
    num_iterations: int = 1000,
    max_k: int = 3
) -> List[Tuple[List[str], float]]:
    """
    Metropolis-Hastings MCMC to sample novel payloads S ~ p(S | y > tau) (Algorithm 1 from paper).
    """
    tf_names = tf_lib.tf_names
    current_payload = [str(x) for x in np.random.choice(tf_names, size=np.random.randint(1, max_k + 1), replace=False)]

    def evaluate_score(payload: List[str]) -> float:
        xi = tf_lib.aggregate_payload(payload).unsqueeze(0) # (1, embedding_dim)
        mean, std = model.predict_uncertainty(xi, num_mc_samples=30)
        m = mean.item()
        s = std.item() + 1e-6
        # P(y > tau | S) under Gaussian approximation
        prob = 0.5 * (1.0 + math.erf((m - tau) / (s * math.sqrt(2.0))))
        return prob

    current_score = evaluate_score(current_payload)
    samples = []

    for _ in range(num_iterations):
        # Proposal step: sample candidate payload S'
        proposed_payload = [str(x) for x in np.random.choice(tf_names, size=np.random.randint(1, max_k + 1), replace=False)]
        proposed_score = evaluate_score(proposed_payload)

        # Acceptance probability alpha = min(1, pi(S') / pi(S))
        alpha = min(1.0, proposed_score / (current_score + 1e-8))
        if np.random.uniform(0, 1) < alpha:
            current_payload = proposed_payload
            current_score = proposed_score

        samples.append((current_payload, current_score))

    # Sort generated samples by probability
    samples.sort(key=lambda x: x[1], reverse=True)
    return samples


# Active Learning Lab-in-the-Loop Simulation
def run_active_learning_simulation(
    tf_lib: TFEmbeddingLibrary,
    payloads: List[List[str]],
    y_targets: np.ndarray,
    n_rounds: int = 3
):
    print("\n--- Active Learning Lab-in-the-Loop Simulation ---")
    hit_threshold = np.percentile(y_targets, 80)
    all_hits = set(i for i, y in enumerate(y_targets) if y >= hit_threshold)
    print(f"Total dataset size: {len(payloads)} | 'Hit' threshold (Top 20%): {hit_threshold:.4f} | Total hits: {len(all_hits)}")

    # Initial training pool (1-TF + 10% multi-TF)
    init_indices = [i for i, p in enumerate(payloads) if len(p) == 1]
    multi_indices = [i for i, p in enumerate(payloads) if len(p) > 1]
    np.random.shuffle(multi_indices)
    init_indices.extend(multi_indices[:int(0.1 * len(multi_indices))])

    train_pool = set(init_indices)
    untested_pool = set(range(len(payloads))) - train_pool

    # Ambrosia UCB Discovery Loop
    discovered_hits_ambrosia = len(train_pool.intersection(all_hits))
    print(f"Round 0 (Initialization): Tested {len(train_pool)} payloads | Discovered {discovered_hits_ambrosia}/{len(all_hits)} hits ({discovered_hits_ambrosia/len(all_hits)*100:.1f}%)")

    for r in range(1, n_rounds + 1):
        # Fit Ambrosia
        X_tr = torch.stack([tf_lib.aggregate_payload(payloads[i]) for i in train_pool], dim=0)
        Y_tr = torch.tensor(y_targets[list(train_pool)], dtype=torch.float32)

        model = AmbrosiaModel(embedding_dim=tf_lib.embedding_dim)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        model.train()
        for _ in range(40):
            opt.zero_grad()
            l = crit(model(X_tr).squeeze(-1), Y_tr)
            l.backward()
            opt.step()

        # Score untested candidates using UCB: mu + 1.0 * std
        untested_list = list(untested_pool)
        X_untested = torch.stack([tf_lib.aggregate_payload(payloads[i]) for i in untested_list], dim=0)
        means, stds = model.predict_uncertainty(X_untested, num_mc_samples=30)
        ucb_scores = means + 1.0 * stds

        # Select top 50 payloads to test in next round
        batch_size = 50
        top_indices = torch.argsort(ucb_scores, descending=True)[:batch_size].cpu().numpy()
        selected = [untested_list[i] for i in top_indices]

        # Update pools
        train_pool.update(selected)
        untested_pool.difference_update(selected)

        hits_found = len(train_pool.intersection(all_hits))
        print(f"Round {r}: Tested +{batch_size} payloads | Cumulative hits: {hits_found}/{len(all_hits)} ({hits_found/len(all_hits)*100:.1f}%)")


# Main Script Execution
if __name__ == "__main__":
    print("=" * 70)
    print("Ambrosia In Silico Reprogramming Model POC")
    print("=" * 70)

    tf_names = ['OCT4', 'SOX2', 'KLF4', 'MYC', 'PAX7', 'NANOG', 'CEBPA', 'GATA4', 'MEF2C', 'HAND2', 'ASCL1', 'NURR1', 'LMX1A', 'RUNX2', 'PPARG']
    tf_lib = TFEmbeddingLibrary(tf_names, embedding_dim=512)

    payloads, y_targets = generate_combinatorial_screen_dataset(tf_lib, n_samples=500)
    print(f"Generated synthetic combinatorial screen: {len(payloads)} payloads across {len(tf_names)} TFs.")

    # Train / Test split
    indices = np.arange(len(payloads))
    np.random.shuffle(indices)
    split = int(0.8 * len(payloads))
    train_idx, test_idx = indices[:split], indices[split:]

    train_payloads = [payloads[i] for i in train_idx]
    test_payloads = [payloads[i] for i in test_idx]
    Y_train, Y_test = y_targets[train_idx], y_targets[test_idx]

    X_train = torch.stack([tf_lib.aggregate_payload(p) for p in train_payloads], dim=0)
    X_test = torch.stack([tf_lib.aggregate_payload(p) for p in test_payloads], dim=0)

    # 1. Train Ambrosia Model
    print("\nTraining Ambrosia Neural Network...")
    model = AmbrosiaModel(embedding_dim=512)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(60):
        optimizer.zero_grad()
        out = model(X_train).squeeze(-1)
        loss = criterion(out, torch.tensor(Y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()

    # 2. Train Baselines
    additive_baseline = AdditiveBaseline()
    additive_baseline.fit(train_payloads, Y_train)
    y_pred_add = additive_baseline.predict(test_payloads)

    mean_baseline = MeanBaseline()
    mean_baseline.fit(Y_train)
    y_pred_mean = mean_baseline.predict(len(test_payloads))

    # Ambrosia Evaluation
    model.eval()
    mu_test, std_test = model.predict_uncertainty(X_test, num_mc_samples=50)
    y_pred_ambrosia = mu_test.cpu().numpy()

    # Metrics Summary
    print("\n" + "=" * 55)
    print(f"{'Model':<20} | {'CSE (↓)':<10} | {'PCC (↑)':<10} | {'AUPRC (↑)':<10}")
    print("-" * 55)
    
    for name, pred in [("Mean Baseline", y_pred_mean), ("Additive Baseline", y_pred_add), ("Ambrosia (pLM+NN)", y_pred_ambrosia)]:
        cse = compute_cse(Y_test, pred)
        pcc = compute_pcc(Y_test, pred)
        auprc = compute_auprc(Y_test, pred)
        print(f"{name:<20} | {cse:<10.4f} | {pcc:<10.4f} | {auprc:<10.4f}")
    print("=" * 55)

    # 3. UCB Ranking Demonstration
    print("\n--- Upper Confidence Bound (UCB) Payload Ranking ---")
    ucb_scores = mu_test + 1.0 * std_test
    top_ucb_idx = torch.argsort(ucb_scores, descending=True)[:5].cpu().numpy()
    
    print("Top 5 Candidate Payloads recommended by Ambrosia UCB:")
    for rank, idx in enumerate(top_ucb_idx, 1):
        p = test_payloads[idx]
        mu = mu_test[idx].item()
        sigma = std_test[idx].item()
        ucb = ucb_scores[idx].item()
        true_y = Y_test[idx]
        print(f" Rank {rank}: {str(p):<25} | mu={mu:.4f} | std={sigma:.4f} | UCB={ucb:.4f} | True y={true_y:.4f}")

    # 4. Generative Payload Design via Metropolis-Hastings MCMC
    print("\n--- Generative MCMC Sampling (Algorithm 1) ---")
    tau_target = np.percentile(Y_train, 85)
    print(f"Sampling payloads S ~ p(S | y > {tau_target:.4f})...")
    mcmc_samples = sample_payloads_mcmc(model, tf_lib, tau=tau_target, num_iterations=500, max_k=3)
    
    print("Top 5 Generatively Designed Multi-TF Payloads:")
    seen = set()
    unique_count = 0
    for p, score in mcmc_samples:
        p_str = "-".join(sorted(p))
        if p_str not in seen:
            seen.add(p_str)
            unique_count += 1
            print(f" Sample {unique_count}: Payload={p} | P(y > tau | S) = {score:.4f}")
            if unique_count >= 5:
                break

    # 5. Active Learning Simulation
    run_active_learning_simulation(tf_lib, payloads, y_targets, n_rounds=3)
    print("\nExecution completed successfully!")
