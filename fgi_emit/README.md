# FGI-EMIT controlled benchmark

This adapter adds a second-dataset experiment for OneFormer3D,
ForestFormer3D, and TreeSegFormer3D. It targets the immutable Zenodo record
[19351234](https://zenodo.org/records/19351234), not a moving latest-version
URL.

## Experimental protocol

- Official training set: 13 plots. The recommended validation plots 1019,
  1022, and 1031 are held out, leaving 10 actual training plots.
- Official test set: 1002, 1004, 1008, 1012, 1018, and 1028. It is never
  used for training, checkpoint selection, or threshold tuning.
- Input to all three models: XYZ only, 0.2 m voxel size, 6 m training
  cylinder, 300 queries, identical augmentation, optimizer, schedule, and
  random seed. Multispectral attributes are deliberately excluded so the
  comparison measures model changes rather than an input-feature change.
- Labels: source class 1 is tree; classes 0 and 2--4 are background; source
  class 5 is excluded as prescribed by the dataset benchmark. Tree instances
  come only from `tree_index`.
- Full plots are evaluated. Overlapping 6 m tiles use a 4 m stride, identical
  score-based merging for all models, and are restored to original retained
  point resolution before export.
- Model selection uses validation F1. The final test report comes from the
  dataset's unchanged official evaluator: IoU matching at 0.5; Precision,
  Recall, F1, Coverage, AP50, crown-category recall, and semantic IoU. Its
  released 40-point/1.5 m small-instance filter remains enabled.

The shared full-plot tiler is a controlled adaptation for this repository. It
does not claim to reproduce every model's separately published inference
recipe; this limitation should be stated in the paper.

## Setup inside the project Docker container

```bash
cd /workspace
python -m pip install -r fgi_emit/requirements.txt
python -m fgi_emit.download --point-clouds
python -m fgi_emit.prepare
```

The download is about 2.45 GB. Every file is checked against the MD5 digest
published in the specified Zenodo record. Preparation refuses partial,
duplicate, or extra plot sets and writes `data/FGI_EMIT/processed/manifest.json`
as an audit trail. Coordinates are shifted in float64 before float32 export to
avoid loss of precision in projected map coordinates.

Expected generated structure:

```text
data/FGI_EMIT/
  raw/{training,test,accuracy,plot_data.yaml,metadata.yaml}
  processed/{points,semantic_mask,instance_mask,source_indices}
  processed/fgi_emit_infos_{train,val,test}.pkl
  processed/manifest.json
```

## Train

Run one experiment per GPU or run these commands sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/fgi_emit/oneformer3d_fgi_emit.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/fgi_emit/forestformer3d_fgi_emit.py
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/fgi_emit/treesegformer3d_fgi_emit.py
```

Each run uses 30,000 optimizer steps. Validation is performed every 1,000
steps (100 epochs), and the best checkpoint is selected using official
validation F1. The fixed training seed is 2026. Run at least three seeds for a
paper-quality variance estimate by overriding both training and inference
seeds, for example:

```bash
python tools/train.py configs/fgi_emit/treesegformer3d_fgi_emit.py \
  --cfg-options randomness.seed=2027 model.test_cfg.inference_seed=2027 \
  --work-dir work_dirs/fgi_emit/treesegformer3d_seed2027
```

## Final test

Use each validation-selected checkpoint exactly once on the official test set:

```bash
python tools/test.py configs/fgi_emit/oneformer3d_fgi_emit.py CHECKPOINT.pth
python tools/test.py configs/fgi_emit/forestformer3d_fgi_emit.py CHECKPOINT.pth
python tools/test.py configs/fgi_emit/treesegformer3d_fgi_emit.py CHECKPOINT.pth
```

For each model, the adapter writes six LAS files and both a JSON result and a
copy of the exact official evaluation settings under that model's
`work_dirs/fgi_emit/<model>/predictions/` directory. It rejects incomplete or
stale test directories instead of reporting a partial score.

## Re-evaluate existing predictions

```bash
python -m fgi_emit.evaluate \
  --pred-root work_dirs/fgi_emit/treesegformer3d/predictions \
  --split test \
  --output work_dirs/fgi_emit/treesegformer3d/predictions/test_metrics.json
```

The official data paper should be cited as Ruoppa et al., *ISPRS Journal of
Photogrammetry and Remote Sensing* 236 (2026), 569--605,
doi:10.1016/j.isprsjprs.2026.04.021. The dataset license is CC BY-NC-SA 4.0.
