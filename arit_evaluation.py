import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

class CitationMetrics:
    """
    Computes standard regression metrics (MAE, RMSE, R^2, Spearman)
    for multi-horizon citation predictions.
    """
    def __init__(self):
        pass

    def compute(self, predicted: torch.Tensor, target: torch.Tensor):
        """
        predicted, target: shape [N, horizons]
        returns dict of metrics
        """
        predicted = predicted.detach().cpu()
        target = target.detach().cpu()

        mae = self._mae(predicted, target)
        rmse = self._rmse(predicted, target)
        r2 = self._r2(predicted, target)
        spear = self._spearman(predicted, target)
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "spearman": spear
        }

    def _mae(self, preds, targs):
        # Convert to float type before computing
        preds = preds.float()
        targs = targs.float()
        return float(torch.mean(torch.abs(preds - targs)))

    def _rmse(self, preds, targs):
        preds = preds.float()
        targs = targs.float()
        return float(torch.sqrt(torch.mean((preds - targs)**2)))

    def _r2(self, preds, targs):
        preds = preds.float()
        targs = targs.float()
        ss_res = torch.sum((targs - preds)**2)
        ss_tot = torch.sum((targs - torch.mean(targs))**2)
        return float(1 - ss_res/(ss_tot+1e-8))

    def _spearman(self, preds, targs):
        preds = preds.float()
        targs = targs.float()
        preds_np = preds.numpy().flatten()
        targs_np = targs.numpy().flatten()
        return float(spearmanr(preds_np, targs_np).correlation)

class ResultsVisualizer:
    """
    Basic visualization: training curves, citation scatter, etc.
    """
    def plot_training_curves(self, metrics_history: dict):
        plt.figure(figsize=(10,6))
        for key in ["policy_loss", "value_loss", "citation_loss", "entropy", "total_loss"]:
            if key in metrics_history:
                plt.plot(metrics_history[key], label=key)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("ARIT Training Curves")
        plt.show()

    def plot_citation_predictions(self, predicted, actual):
        """
        predicted, actual: shape [N]
        """
        predicted_np = predicted.cpu().numpy()
        actual_np = actual.cpu().numpy()
        plt.figure()
        plt.scatter(actual_np, predicted_np, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Citation Prediction Scatter")
        plt.show()
