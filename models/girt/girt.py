import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- MODEL ----------

class GIRT(nn.Module):
    """
    Minimal GKT + 2PL IRT model.

    Knowledge state: per-student H of shape (B, d, C).
    Graph: symmetric  A = softmax(E.T @ E / sqrt(d)).
    Prediction:
        h_q   = (1 - gate) * H[:, k] + gate * (H @ A[k])    # mix self + graph
        score = h_q.sum(dim=1)                              # scalar score from d-dim state
        logit = a_j * (score - b_j)                         # standard 2PL
    where a_j and b_j are SCALARS per item.

    The single learnable knob coupling the graph to the prediction is `gate`.
    Everything else is the classic GKT + IRT formulation.
    """

    def __init__(self, n_skills, n_items, d):
        super().__init__()
        self.n_skills = n_skills
        self.n_items  = n_items
        self.d        = d

        # Concept embeddings (symmetric graph)
        self.E = nn.Parameter(torch.empty(d, n_skills))
        nn.init.xavier_normal_(self.E)

        # Interaction embedding (success / failure per skill)
        self.X = nn.Parameter(torch.randn(d, 2 * n_skills) * 0.01)

        # GRU parameters
        self.Wr = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wr)
        self.Wz = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wz)
        self.Wh = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Wh)
        self.Ur = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Ur)
        self.Uz = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Uz)
        self.Uh = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.Uh)
        self.br = nn.Parameter(torch.zeros(d))
        self.bz = nn.Parameter(torch.zeros(d))
        self.bh = nn.Parameter(torch.zeros(d))

        # ── IRT parameters: SCALAR per item ──────────────────────────────────
        self.a_raw = nn.Parameter(torch.zeros(n_items))   # softplus(0) ≈ 0.69
        self.b     = nn.Parameter(torch.zeros(n_items))

        # ── Prediction-time graph gate ───────────────────────────────────────
        self.alpha_gate = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5

    # ── helpers ───────────────────────────────────────────────────────────────

    def graph_construction(self):
        """Symmetric A : (C, C) row-stochastic adjacency from concept embeddings."""
        A_hat = self.E.T @ self.E / (self.d ** 0.5)
        return F.softmax(A_hat, dim=1)

    def gru_step(self, x_t, h_tilda):
        r     = torch.sigmoid(x_t @ self.Wr.T + h_tilda @ self.Ur.T + self.br)
        z     = torch.sigmoid(x_t @ self.Wz.T + h_tilda @ self.Uz.T + self.bz)
        h_hat = torch.tanh(   x_t @ self.Wh.T + r * (h_tilda @ self.Uh.T) + self.bh)
        return (1 - z) * h_tilda + z * h_hat

    def _predict_logits(self, H, k_t, j_t, A, b_idx):
        """Shared prediction path used by both forward() and predict()."""
        h_self = H[b_idx, :, k_t]                       # (B, d)
        h_aggr = torch.einsum('bdc,bc->bd', H, A[k_t])  # (B, d)
        gate   = torch.sigmoid(self.alpha_gate)
        h_q    = (1 - gate) * h_self + gate * h_aggr    # (B, d)

        score  = h_q.sum(dim=1)                          # (B,)
        a_j    = F.softplus(self.a_raw[j_t])             # (B,) positive
        b_j    = self.b[j_t]                             # (B,)
        return a_j * (score - b_j)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch):
        seq_item, seq_skill, seq_y, _, seq_mask, _ = batch
        B, T   = seq_item.shape
        device = seq_item.device

        H     = torch.zeros(B, self.d, self.n_skills, device=device)
        A     = self.graph_construction()
        b_idx = torch.arange(B, device=device)

        total_loss  = torch.zeros((), device=device)
        total_count = torch.zeros((), device=device)

        for t in range(T):
            mask_t = seq_mask[:, t].float()
            k_t = seq_skill[:, t]
            j_t = seq_item[:, t]
            y_t = seq_y[:, t]

            logits = self._predict_logits(H, k_t, j_t, A, b_idx)

            loss_per_b = F.binary_cross_entropy_with_logits(
                logits, y_t, reduction='none'
            )
            total_loss  = total_loss  + (loss_per_b * mask_t).sum()
            total_count = total_count + mask_t.sum()

            if t < T - 1:
                y_t_long  = y_t.long()
                idx_inter = k_t + (1 - y_t_long) * self.n_skills
                x_t       = self.X[:, idx_inter].T
                h_tilda   = torch.einsum('bdc,bc->bd', H, A[k_t])

                new_h = self.gru_step(x_t, h_tilda)

                old_h    = H[b_idx, :, k_t]
                mask_t_d = mask_t.unsqueeze(1)
                write_h  = mask_t_d * new_h + (1 - mask_t_d) * old_h

                idx_scatter = k_t.view(B, 1, 1).expand(-1, self.d, 1)
                H = H.scatter(2, idx_scatter, write_h.unsqueeze(2))

        return total_loss / total_count.clamp(min=1).float()


# ---------- PREDICTION ----------

def predict(model, tensors, batch_size=64):
    model.eval()
    seq_item, seq_skill, seq_y, _, seq_mask, orig_idx = tensors
    n_students = seq_item.shape[0]

    preds_chunks, origs_chunks = [], []

    with torch.no_grad():
        A = model.graph_construction()

        for batch_start in range(0, n_students, batch_size):
            batch_end = min(batch_start + batch_size, n_students)

            si = seq_item[ batch_start:batch_end]
            sk = seq_skill[batch_start:batch_end]
            sy = seq_y[    batch_start:batch_end]
            sm = seq_mask[ batch_start:batch_end]
            so = orig_idx[ batch_start:batch_end]

            lens    = sm.sum(dim=1)
            T_batch = int(lens.max().item())
            if T_batch == 0:
                continue
            si = si[:, :T_batch]; sk = sk[:, :T_batch]; sy = sy[:, :T_batch]
            sm = sm[:, :T_batch]; so = so[:, :T_batch]

            preds, origs = _predict_one_batch(model, si, sk, sy, sm, so, A)
            preds_chunks.append(preds)
            origs_chunks.append(origs)

    if not preds_chunks:
        return torch.empty(0)
    all_preds = torch.cat(preds_chunks)
    all_orig  = torch.cat(origs_chunks)
    return all_preds[torch.argsort(all_orig)].cpu()


def _predict_one_batch(model, seq_item, seq_skill, seq_y, seq_mask, orig_idx, A):
    B, T   = seq_item.shape
    device = seq_item.device

    H     = torch.zeros(B, model.d, model.n_skills, device=device)
    b_idx = torch.arange(B, device=device)

    preds_list, origs_list = [], []

    for t in range(T):
        mask_t = seq_mask[:, t]
        if not mask_t.any():
            break

        k_t = seq_skill[:, t]
        j_t = seq_item[ :, t]
        y_t = seq_y[    :, t]

        logits = model._predict_logits(H, k_t, j_t, A, b_idx)
        preds  = torch.sigmoid(logits)

        preds_list.append(preds[mask_t])
        origs_list.append(orig_idx[mask_t, t])

        if t < T - 1:
            y_t_long  = y_t.long()
            idx_inter = k_t + (1 - y_t_long) * model.n_skills
            x_t       = model.X[:, idx_inter].T
            h_tilda   = torch.einsum('bdc,bc->bd', H, A[k_t])
            new_h     = model.gru_step(x_t, h_tilda)

            old_h    = H[b_idx, :, k_t]
            mask_t_f = mask_t.float().unsqueeze(1)
            write_h  = mask_t_f * new_h + (1 - mask_t_f) * old_h

            idx_scatter = k_t.view(B, 1, 1).expand(-1, model.d, 1)
            H = H.scatter(2, idx_scatter, write_h.unsqueeze(2))

    return torch.cat(preds_list), torch.cat(origs_list)