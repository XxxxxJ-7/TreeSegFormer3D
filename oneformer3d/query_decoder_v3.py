"""
V3 Query Decoder with two key improvements over the baseline:

1. 3D Positional Encoding: injects sinusoidal position encoding into both
   inst_feats (cross-attention K/V) and mask_feats (mask prediction), so
   the model knows WHERE each voxel is in 3D space.

2. MLP Mask Head: replaces the simple dot-product ('nd,md->nm')
   with a learnable MLP that takes [query, feat, query-feat, query*feat]
   as input, giving much richer mask prediction capacity.

Usage:
    from .query_decoder_v3 import ForAINetv2QueryDecoder_XAwarequery_V3
"""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


# ===========================================================================
# 3D Positional Encoding
# ===========================================================================

class PositionalEncoding3D(nn.Module):
    """Sinusoidal positional encoding for 3D voxel coordinates.

    For each voxel coordinate (x, y, z), computes a mix of sin/cos
    encodings at exponentially decaying frequencies, then concatenates
    across axes.  Unlike the strict Transformer formula (which requires
    d_model divisible by 6), this implementation uses a flexible
    per-axis allocation:
        d_per_axis = d_model // 3   (floor division)
    Within each axis: sin + cos alternate to fill d_per_axis dims.

    If 3 * d_per_axis < d_model (e.g. 256 → 3×85=255), the output is
    padded with zeros to exactly d_model so it can be added to features.

    Args:
        d_model:   output dimension (any value; default 256)
        max_freq:  maximum frequency wavelength (default 10000.0)
    """

    def __init__(self, d_model=256, max_freq=10000.0):
        super().__init__()
        self.d_model = d_model
        self.d_per_axis = d_model // 3            # 256 → 85
        self.n_freqs = (self.d_per_axis + 1) // 2  # number of distinct frequencies
        self.max_freq = max_freq

        # Frequency divisors:  max_freq^{-2i / d_per_axis}
        div_term = torch.exp(
            torch.arange(0, self.n_freqs, dtype=torch.float32) *
            (-math.log(max_freq) / self.n_freqs)
        )
        self.register_buffer('div_term', div_term, persistent=False)

    def forward(self, coords):
        """
        Args:
            coords: [N, 3]  voxel (x, y, z) coordinates

        Returns:
            pe: [N, d_model]  positional encoding
        """
        device = coords.device
        N = coords.shape[0]
        pe = torch.zeros(N, self.d_model, device=device)

        for dim_idx in range(3):
            start = dim_idx * self.d_per_axis
            end = start + self.d_per_axis
            # pos: [N] * [n_freqs] → [N, n_freqs]
            pos = coords[:, dim_idx:dim_idx+1].float() * self.div_term.unsqueeze(0)
            # interleave sin, cos to fill d_per_axis slots
            for f in range(self.n_freqs):
                slot = f * 2
                if slot < self.d_per_axis:
                    pe[:, start + slot] = torch.sin(pos[:, f])
                if slot + 1 < self.d_per_axis:
                    pe[:, start + slot + 1] = torch.cos(pos[:, f])
        return pe


# ===========================================================================
# MLP Mask Head  (replaces the dot-product einsum)
# ===========================================================================

class MLPMaskHead(nn.Module):
    """Efficient mask prediction head with learnable projections.

    Instead of materializing a Q×C×4d tensor and running an MLP over it,
    this head projects queries and features into a shared space, then
    computes masks via matrix multiplication.

    Memory: identical to einsum (Q × P float32), zero broadcasting overhead.

    Architecture::

        queries  ──→ Linear(d→h) ──→ ReLU ──→ Linear(h→h) ──→ q' [Q, h]
                                                      ↓
                                      mask_logits = q' @ f'.T   [Q, P]
                                                      ↑
        feats    ──→ Linear(d→h) ──→ ReLU ──→ Linear(h→h) ──→ f' [P, h]

    This gives nonlinear interaction (two Linear+ReLU layers) while
    staying purely matrix-multiply based — no broadcasting, no concatenation.

    Args:
        d_model:    input feature dimension
        hidden_dim: projection hidden dimension
        dropout:    dropout rate
    """

    def __init__(self, d_model=256, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, queries, mask_feats):
        """
        Args:
            queries:    [Q, d_model]  (already LayerNorm'ed)
            mask_feats: [P, d_model]  mask-feature per point

        Returns:
            pred_mask:  [Q, P]  logit per (query, point) pair
        """
        q = self.query_proj(queries)      # [Q, h]
        f = self.feat_proj(mask_feats)    # [P, h]
        return torch.mm(q, f.t())         # [Q, P] — same memory as einsum


# ===========================================================================
# Transformer building blocks  (re-used from original query_decoder.py)
# ===========================================================================

class CrossAttentionLayer(BaseModule):
    """Cross-attention: queries attend over source (point) features."""

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            # The weights are unused. On large FGI-EMIT crops the default
            # path allocates a heads x queries x voxels attention tensor.
            output, _ = self.attn(
                queries[i], k, v, attn_mask=attn_mask,
                need_weights=False)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self-attention: queries attend among themselves."""

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = []
        for y in x:
            z, _ = self.attn(y, y, y, need_weights=False)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed-forward network: Linear(256→1024) → GELU → Dropout → Linear(1024→256)."""

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        act = nn.ReLU() if activation_fn == 'relu' else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


# ===========================================================================
# V3 Decoder  (main class)
# ===========================================================================

@MODELS.register_module()
class ForAINetv2QueryDecoder_XAwarequery_V3(BaseModule):
    """V3 Query Decoder with 3D Position Encoding + MLP Mask Head.

    Improvements over the baseline ``ForAINetv2QueryDecoder_XAwarequery``:

    1. **3D Position Encoding**  — sinusoidal PE injected into inst_feats
       and mask_feats so every voxel carries spatial identity.

    2. **MLP Mask Head**         — replaces ``einsum('nd,md->nm')`` with a
       small MLP that jointly models [q, f, q-f, q*f] → single logit.

    The rest of the architecture (iterative prediction, mask-gated
    cross-attention, query construction) is identical to the baseline.

    Config Example::

        decoder=dict(
            type='ForAINetv2QueryDecoder_XAwarequery_V3',
            num_layers=6,
            num_classes=1,
            num_instance_queries=0,
            num_semantic_queries=3,
            num_instance_classes=3,
            in_channels=32,
            d_model=256,
            num_heads=8,
            hidden_dim=1024,
            dropout=0.0,
            activation_fn='gelu',
            iter_pred=True,
            attn_mask=True,
            fix_attention=True,
            objectness_flag=True,
        )
    """

    def __init__(self,
                 num_layers,
                 num_instance_queries,
                 num_semantic_queries,
                 num_classes,
                 in_channels,
                 d_model,
                 num_heads,
                 hidden_dim,
                 dropout,
                 activation_fn,
                 iter_pred,
                 attn_mask,
                 fix_attention,
                 objectness_flag,
                 pos_enc_max_freq=10000.0,       # ← new
                 mlp_mask_hidden_dim=None,       # ← new (defaults to d_model)
                 checkpoint_mask_head=False,
                 **kwargs):
        super().__init__()

        # ----- 3D Position Encoding -----
        self.pos_enc = PositionalEncoding3D(
            d_model=d_model, max_freq=pos_enc_max_freq)

        # ----- MLP Mask Head -----
        _mlp_hidden = mlp_mask_hidden_dim if mlp_mask_hidden_dim is not None else d_model
        self.mask_head = MLPMaskHead(
            d_model=d_model, hidden_dim=_mlp_hidden, dropout=dropout)
        self.checkpoint_mask_head = checkpoint_mask_head

        # ----- projection layers (same as baseline) -----
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True))

        self.objectness_flag = objectness_flag

        if num_instance_queries > 0:
            self.query = nn.Embedding(
                num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model))
            self.num_semantic_queries = \
                num_instance_queries + num_semantic_queries
            self.semantic_queries = nn.Embedding(
                num_semantic_queries, in_channels)

        # ----- transformer layers -----
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout, fix_attention)
            for _ in range(num_layers)])
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([
            FFN(d_model, hidden_dim, dropout, activation_fn)
            for _ in range(num_layers)])

        # ----- prediction heads -----
        self.out_norm = nn.LayerNorm(d_model)
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 1))

        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model))

        self.iter_pred = iter_pred
        self.attn_mask = attn_mask

    # ------------------------------------------------------------------
    #  Query construction  (identical to baseline)
    # ------------------------------------------------------------------

    def _get_queries(self, queries=None, batch_size=None):
        if batch_size is None:
            batch_size = len(queries)

        result_queries = []
        for i in range(batch_size):
            if len(queries[i]) != 0:
                device = queries[i].device

        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                semantic_queries = self.semantic_queries.weight
                if len(queries[i]) == 0:
                    queries[i] = torch.empty(
                        0, *semantic_queries.shape[1:]).to(device)
                concat_queries = torch.cat(
                    (queries[i], semantic_queries.to(queries[i].device)), dim=0)
                result_query.append(self.query_proj(concat_queries))
            result_queries.append(torch.cat(result_query))
        return result_queries

    # ------------------------------------------------------------------
    #  Prediction head  (MLP Mask Head replaces einsum)
    # ------------------------------------------------------------------

    def _forward_head(self, queries, mask_feats):
        pred_scores, pred_masks, attn_masks = [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])

            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)

            # ★ V3: MLP Mask Head instead of einsum ★
            if (self.checkpoint_mask_head and self.training
                    and torch.is_grad_enabled()):
                # Keep the Q x P logits required by deep supervision, but
                # recompute the MLP projections during backward instead of
                # retaining them for all decoder layers.
                pred_mask = checkpoint(
                    self.mask_head, norm_query, mask_feats[i])
            else:
                pred_mask = self.mask_head(norm_query, mask_feats[i])

            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                # safety: if a query masks out ALL points, unmask it
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        attn_masks = attn_masks if self.attn_mask else None
        return pred_scores, pred_masks, attn_masks

    # ------------------------------------------------------------------
    #  Forward passes
    # ------------------------------------------------------------------

    def forward_simple(self, x, queries, coords=None):
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]

        if coords is not None:
            for i in range(len(inst_feats)):
                pe = self.pos_enc(coords[i])
                inst_feats[i] = inst_feats[i] + pe
                mask_feats[i] = mask_feats[i] + pe

        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        pred_scores, pred_masks, _ = self._forward_head(queries, mask_feats)
        return dict(masks=pred_masks, scores=pred_scores)

    def forward_iter_pred(self, x, queries, coords=None):
        pred_scores, pred_masks = [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]

        # ★ V3: inject 3D position encoding into both feature sets ★
        if coords is not None:
            for i in range(len(inst_feats)):
                pe = self.pos_enc(coords[i])
                inst_feats[i] = inst_feats[i] + pe
                mask_feats[i] = mask_feats[i] + pe

        queries = self._get_queries(queries, len(x))

        # step 0: predict BEFORE any transformer layer
        pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        # steps 1..N: predict AFTER each transformer layer
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'masks': masks, 'scores': scores}
            for scores, masks in zip(pred_scores[:-1], pred_masks[:-1])
        ]
        return dict(
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None, coords=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, queries, coords=coords)
        else:
            return self.forward_simple(x, queries, coords=coords)
