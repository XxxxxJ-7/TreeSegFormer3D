import torch
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from .unified_criterion import ForAINetv2UnifiedCriterion_XAwarequery


@MODELS.register_module()
class ForAINetv2UnifiedCriterion_QRL(ForAINetv2UnifiedCriterion_XAwarequery):
    """Unified criterion with Inter-Query Repulsion Loss (QRL).

    Extends ForAINetv2UnifiedCriterion_XAwarequery by penalizing high
    soft-IoU between masks predicted by queries that belong to *different*
    GT instances.  Queries assigned to the same GT instance are NOT
    penalized (they are supposed to overlap).

    Args:
        repulsion_weight (float): Scalar weight for the repulsion loss.
        repulsion_iou_threshold (float): Soft-IoU values below this
            threshold are not penalized (hinge).
        **kwargs: Forwarded to the parent criterion.
    """

    def __init__(
        self,
        repulsion_weight: float = 1.0,
        repulsion_iou_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.repulsion_weight = repulsion_weight
        self.repulsion_iou_threshold = repulsion_iou_threshold

    def __call__(self, pred, insts):
        query_inslabels = [inst.query_inslabel for inst in insts]

        # Parent strips semantic queries from pred['masks'] in-place
        loss = super().__call__(pred, insts)

        # pred['masks'] now only contains instance-query masks (logits)
        loss['repulsion_loss'] = self._repulsion_loss(
            pred['masks'], query_inslabels,
        )
        return loss

    # ------------------------------------------------------------------ #
    #  Core: differentiable pairwise repulsion via soft IoU               #
    # ------------------------------------------------------------------ #
    def _repulsion_loss(self, inst_masks, query_inslabels):
        """Pairwise repulsion between queries from different GT instances.

        Args:
            inst_masks (list[Tensor]): Each element (Q_i, N_i), raw logits.
            query_inslabels (list[Tensor]): Each (Q_i,), -1 = unmatched.

        Returns:
            Tensor: scalar loss.
        """
        per_sample_losses = []

        for masks, labels in zip(inst_masks, query_inslabels):
            if masks.shape[0] < 2:
                continue

            probs = masks.sigmoid()                            # (Q, N)
            intersection = probs @ probs.T                     # (Q, Q)
            area = probs.sum(dim=1)                            # (Q,)
            union = (area.unsqueeze(0) + area.unsqueeze(1)
                     - intersection)
            iou = intersection / union.clamp(min=1e-6)         # (Q, Q)

            valid = labels != -1                               # (Q,)
            same_inst = labels.unsqueeze(0) == labels.unsqueeze(1)
            both_valid = valid.unsqueeze(0) & valid.unsqueeze(1)
            upper_tri = torch.triu(
                torch.ones(masks.shape[0], masks.shape[0],
                           device=masks.device, dtype=torch.bool),
                diagonal=1,
            )
            penalize = ~same_inst & both_valid & upper_tri

            if penalize.sum() == 0:
                continue

            penalties = F.relu(iou[penalize] - self.repulsion_iou_threshold)
            per_sample_losses.append(penalties.mean())

        if len(per_sample_losses) == 0:
            return inst_masks[0].sum() * 0.0   # differentiable zero

        return (torch.stack(per_sample_losses).mean()
                * self.repulsion_weight)
