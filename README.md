# TriDerm

Code for *Assessing Multimodal Chronic Wound Embeddings with Expert Triplet Agreement*
(MICCAI 2026, ISIC workshop). TriDerm learns wound similarity for recessive dystrophic
epidermolysis bullosa (RDEB) from two modalities. The **vision branch** pools the DermLIP patch
tokens inside each wound mask with a trained attention head (VICReg) and averages the wounds of
an image. The **text branch** asks an LLM, acting as a synthetic expert, "is wound *i* more
similar to *j* or to *k*?" over the expert wound descriptions and fits a soft ordinal embedding
(SOE) to its answers. The two distance matrices are combined by training-free
**uncertainty-aware** or **similarity-based fusion**. Every embedding is scored against
dermatologist triplet judgments with anchor-balanced agreement, micro agreement, macro-F1 and
Cohen's kappa.

The data (images, wound masks, descriptions, expert judgments) are not part
of this repository.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run triderm --help
just check    # ruff + tests on synthetic data
```

The vision models and the LLM baselines need a GPU. DINOv3, SAM3, MedGemma, DermLIP and PanDerm
are gated on Hugging Face: accept their licenses and run `hf auth login` first.

Triplet questions go to any OpenAI-compatible endpoint with structured output, e.g. vLLM:

```bash
vllm serve openai/gpt-oss-120b --port 8000
export LLM_BASE_URL=http://localhost:8000/v1   # default
export LLM_API_KEY=...                          # if the endpoint needs one
```

Answers are cached in `out/cache/`, so re-runs only ask what is missing or failed.

## Data layout

Set `TRIDERM_DATA_ROOT` (default `data/`) and `TRIDERM_OUT` (default `out/`), or edit
`[paths]` in `configs/paper.toml`.

```
<data_root>/
  annotations/expert_triplets.db        SQLite table tripletresult(id, left_fn, middle_fn,
                                        right_fn, result), or a CSV with columns
                                        id,left,middle,right,result; middle is the anchor,
                                        result is left | right | skip
  descriptions/expert_descriptions.txt  "<photo_id>: <text>" per line; "<id1>, <id2>: <text>"
                                        shares one text between images
  images/photos/<photo_id>.<jpg|png>
  images/masks/<photo_id>_wounds_<k>.png  one binary mask per wound
  excluded_photos.txt                   optional: photo IDs left out of the vision cohort
```

Photo IDs are file stems such as `12` or `12a`; images with the same leading number belong to
one photo series and are never compared with each other in LLM triplet questions.

## Paper results

Embeddings are written to `out/embeddings/<name>.npz`, results to `out/results/`. All
hyperparameters are in `configs/paper.toml`; `just paper` runs everything in order.

| Paper result | Command |
|---|---|
| LLM triplet answers | `triderm ask-triplets --oracle gpt-oss-120b` |
| TriDerm text embedding (SOE, D=4) | `triderm fit-soe --oracle gpt-oss-120b` |
| Text baselines (MiniLM, RoBERTa, PubMedBERT) | `triderm embed-text` |
| LLM mean-pooling baselines | `triderm embed-llm --model gpt-oss-120b` (and `medgemma-27b`) |
| Vision baselines (SigLIP, PanDerm, SAM3, MedGemma, DINOv3, DermLIP) | `triderm embed-vision` |
| TriDerm vision embedding (Fig. 1a) | `triderm train-vision` |
| **Table 1**: text, vision and fused agreement | `triderm table1` |
| **Table 2a**: SOE dimension, triplet budget, LLM oracle | `ask-triplets` for `gpt-oss-20b` and `medgemma-27b`, then `triderm soe-ablation` |
| **Table 2b**: pooling and loss | `train-vision --variant` `mean-pooling`, `triplet`, `contrastive`, then `triderm vision-ablation` |
| **Figure 2**: nearest and farthest neighbours | `triderm neighbors` |
| Any embedding against the experts | `triderm evaluate <name> ...` |

## Citation

```bibtex
@inproceedings{kabus2026triderm,
  title     = {Assessing Multimodal Chronic Wound Embeddings with Expert Triplet Agreement},
  author    = {Kabus, Fabian and Hindel, Julia and Bratuli{\'c}, Jelena and
               Karakioulaki, Meropi and Gupta, Ayush and Has, Cristina and Brox, Thomas and
               Valada, Abhinav and Binder, Harald},
  booktitle = {MICCAI 2026 Workshops (ISIC)},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
