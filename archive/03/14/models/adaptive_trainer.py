import torch

class AdaptiveLRScheduler:
    def __init__(self, optimizer, tolerance_increase=3, tolerance_decrease=3, tolerance_overfit=3, factor_increase=10, factor_decrease=0.1):
        self.optimizer = optimizer
        self.tolerance_increase = tolerance_increase
        self.tolerance_decrease = tolerance_decrease
        self.tolerance_overfit = tolerance_overfit
        self.factor_increase = factor_increase
        self.factor_decrease = factor_decrease
        self.best_val_loss = float('inf')
        self.increase_count = 0
        self.decrease_count = 0
        self.overfit_count = 0

    def step(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.increase_count += 1
            self.decrease_count = 0
            self.overfit_count = 0
        else:
            self.decrease_count += 1
            self.increase_count = 0
            self.overfit_count += 1

        # Check if it's time to increase the learning rate
        if self.increase_count >= self.tolerance_increase:
            self.adjust_learning_rate(self.factor_increase)
            self.increase_count = 0
            print(f"Increasing learning rate by a factor of {self.factor_increase}")

        # Check if it's time to decrease the learning rate
        if self.decrease_count >= self.tolerance_decrease:
            self.adjust_learning_rate(self.factor_decrease)
            self.decrease_count = 0
            self.overfit_count = 0
            print(f"Decreasing learning rate by a factor of {self.factor_decrease}")

        # Check for overfitting
        if self.overfit_count >= self.tolerance_overfit:
            print("No improvement after decreasing learning rate multiple times. Possible overfitting. Stopping training.")
            return True  # Indicates to stop training

        return False  # Indicates training should continue

    def adjust_learning_rate(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor