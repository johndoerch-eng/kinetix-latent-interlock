# -*- coding: utf-8 -*-
"""
KinetiX-LDS: SDK Core & PyTorch Hooks
Cœur de calcul de la dérive latente et recentrage dynamique.
Gère de manière native et sécurisée les tenseurs GPU et types bfloat16/float16.
"""

import numpy as np
import torch

class KinetiXTelemetryRouter:
    def __init__(self, dim_size=2048, alpha=0.20, beta=0.85):
        """
        Initialise le routeur de télémétrie latente.
        
        Parameters:
            dim_size (int) : Dimension de l'espace caché du LLM.
            alpha (float) : Coefficient de fuite du réservoir court-terme (Fast dynamics).
            beta (float) : Coefficient de persistance du réservoir long-terme (Slow dynamics).
        """
        self.dim_size = dim_size
        self.alpha = alpha
        self.beta = beta
        self.reset()
        
    def reset(self):
        """Réinitialise les réservoirs de dérive latente."""
        self.reservoir_fast = np.zeros(self.dim_size)
        self.reservoir_slow = np.zeros(self.dim_size)
        
    def update_and_evaluate(self, h_t):
        """
        Met à jour les réservoirs et évalue la dérive binarisée de Hamming (SimHash).
        Supporte et convertit les tenseurs GPU BFloat16/Float16/Float32.
        """
        if isinstance(h_t, torch.Tensor):
            h_t = h_t.to(torch.float32).detach().cpu().numpy()
            
        # Mise à jour dynamique des réservoirs
        self.reservoir_fast = np.tanh(h_t + self.alpha * self.reservoir_fast)
        self.reservoir_slow = np.tanh(h_t + self.beta * self.reservoir_slow)
        
        # Binarisation et calcul de la distance de Hamming (fraction de bits divergents)
        sign_fast = np.sign(self.reservoir_fast)
        sign_slow = np.sign(self.reservoir_slow)
        
        drift = float(np.mean(sign_fast != sign_slow))
        return drift


class KinetiXStreamMonitor:
    def __init__(self, dim_size=2048, alpha=0.20, beta=0.85, threshold=0.132):
        """
        Initialise le moniteur de flux WebSocket pour le Dashboard de contrôle.
        """
        self.dim_size = dim_size
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        
        self.reservoir_fast = np.zeros(dim_size)
        self.reservoir_slow = np.zeros(dim_size)
        self.last_activation = None
        
    def reset(self):
        self.reservoir_fast = np.zeros(self.dim_size)
        self.reservoir_slow = np.zeros(self.dim_size)
        self.last_activation = None

    def pytorch_hook(self, module, input, output):
        """
        Hook PyTorch à enregistrer sur la couche intermédiaire cible du LLM.
        Capte l'état d'activation final (hidden_states) à chaque token généré.
        """
        tensor = output[0] if isinstance(output, tuple) else output
        
        if tensor.ndim == 3:
            self.last_activation = tensor[0, -1, :].to(torch.float32).detach().cpu().numpy()
        elif tensor.ndim == 2:
            self.last_activation = tensor[-1, :].to(torch.float32).detach().cpu().numpy()

    def evaluate_drift(self, h_t=None):
        """
        Calcule la dérive géométrique binarisée (Hamming) sur les réservoirs.
        """
        if h_t is None:
            h_t = self.last_activation
            
        if h_t is None:
            return 0.0

        if isinstance(h_t, torch.Tensor):
            h_t = h_t.to(torch.float32).detach().cpu().numpy()

        self.reservoir_fast = np.tanh(h_t + self.alpha * self.reservoir_fast)
        self.reservoir_slow = np.tanh(h_t + self.beta * self.reservoir_slow)
        
        sign_fast = np.sign(self.reservoir_fast)
        sign_slow = np.sign(self.reservoir_slow)
        
        drift = float(np.mean(sign_fast != sign_slow))
        return drift


# Exemple d'intégration concrète :
if __name__ == "__main__":
    print("SDK KinetiX-LDS chargé avec succès.")
