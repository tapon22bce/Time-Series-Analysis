"""Turns raw similarities into an EXPLICIT, EXPOSED weight vector used both by the diffusion
conditioner and by the explainer. This is the core faithfulness mechanism: the same w that
conditions the model is what the explainer reports, so we can test consistency under intervention."""
import torch, torch.nn as nn


class RetrievalWeighter(nn.Module):
    """w_i = gate * softmax(sim_i / tau)_i  , gate in [0,1] is the ADAPTIVE RETRIEVAL amount.
    gate is predicted from (a) the query context and (b) STATISTICS OF THE RETRIEVED SET ITSELF
    (mean/min/spread of similarity, and how much the neighbors' futures disagree with each other).
    This second part is essential: the context alone cannot reveal that a retrieval database has
    been corrupted or that the current neighborhood is unreliable, since the context is identical
    regardless of which (possibly-corrupted) neighbors got retrieved. Only by looking at the
    retrieved set's internal (dis)agreement can the gate learn to distrust a bad neighborhood --
    this is exactly the signal the bias-variance / conditional-shift analysis probes."""

    N_CTX_FEAT = 6
    N_SET_FEAT = 4

    def __init__(self, hidden=32, learn_tau=True, sparse_k=None):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(1.0)) if learn_tau else None
        self.gate_net = nn.Sequential(nn.Linear(self.N_CTX_FEAT + self.N_SET_FEAT, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.sparse_k = sparse_k

    @staticmethod
    def ctx_feats(ctx_n):
        d1 = ctx_n.diff(dim=1)
        return torch.stack([ctx_n.std(1), d1.std(1), ctx_n[:, -1], ctx_n.mean(1),
                             ctx_n.max(1).values - ctx_n.min(1).values, ctx_n[:, -1] - ctx_n[:, 0]], 1)

    @staticmethod
    def set_feats(sim: torch.Tensor, nb_fut_n: torch.Tensor = None):
        """sim (B,k); nb_fut_n (B,k,H) optional. -> (B,4): mean/min/std similarity + neighbor
        future disagreement (mean pairwise std across the k retrieved futures, i.e. how much the
        candidates disagree about what happens next -- high disagreement should suppress trust)."""
        B, k = sim.shape
        disagree = nb_fut_n.std(dim=1).mean(dim=1, keepdim=True) if nb_fut_n is not None else sim.new_zeros(B, 1)
        return torch.cat([sim.mean(1, keepdim=True), sim.min(1, keepdim=True).values,
                          sim.std(1, keepdim=True), disagree], 1)

    def forward(self, sim: torch.Tensor, ctx_n: torch.Tensor, nb_fut_n: torch.Tensor = None):
        """sim (B,k) in [-1,1]; returns w (B,k) summing to gate (B,1), and gate (B,1)."""
        tau = (self.tau.abs() + 1e-3) if self.tau is not None else 1.0
        logits = sim / tau
        if self.sparse_k is not None and self.sparse_k < sim.shape[1]:
            kth = torch.topk(logits, self.sparse_k, dim=1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, -1e9)
        soft = torch.softmax(logits, dim=1)
        feats = torch.cat([self.ctx_feats(ctx_n), self.set_feats(sim, nb_fut_n)], 1)
        gate = torch.sigmoid(self.gate_net(feats))       # (B,1) in (0,1)
        return soft * gate, gate

    def reg_loss(self, gate, target_mean=None):
        """Optional regularizer nudging average retrieval usage; None disables it."""
        if target_mean is None: return gate.new_zeros(())
        return (gate.mean() - target_mean).pow(2)
