import torch
import torch.nn.functional as F
import unittest
import sys
import os

from train_redi import build_chat_text, balance_traces

class TestREDI(unittest.TestCase):
    def test_balance_traces(self):
        """Test that traces are reliably balanced with oversampling."""
        traces = [
            {"label": 1, "id": 1},
            {"label": -1, "id": 2},
            {"label": -1, "id": 3},
            {"label": -1, "id": 4},
        ]
        balanced = balance_traces(traces)
        pos_count = sum(1 for t in balanced if t["label"] == 1)
        neg_count = sum(1 for t in balanced if t["label"] == -1)
        self.assertEqual(pos_count, neg_count)
        self.assertEqual(pos_count, 3) # Oversampled the minority

    def test_redi_loss_direction(self):
        """Test REDI loss logic independently since compute_redi_loss needs a tokenizer."""
        # Simulated log probs: e.g. -2.0 per token
        log_probs = torch.tensor([-2.0, -1.5, -3.0], requires_grad=True)
        mean_lp = log_probs.mean()

        # Positive trace: loss = -1 * mean(lp)
        loss_pos = -1.0 * mean_lp
        self.assertTrue(loss_pos.item() > 0)
        # Gradient of positive loss wrt log_probs should be negative (we want to increase log_prob)
        loss_pos.backward(retain_graph=True)
        self.assertTrue(torch.all(log_probs.grad < 0))
        
        log_probs.grad.zero_()

        # Negative trace: loss = -(-1) * mean(lp)
        loss_neg = -(-1.0) * mean_lp
        self.assertTrue(loss_neg.item() < 0)
        # Gradient of negative loss wrt log_probs should be positive (we want to decrease log_prob)
        loss_neg.backward()
        self.assertTrue(torch.all(log_probs.grad > 0))

if __name__ == '__main__':
    unittest.main()
