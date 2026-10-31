# Manuscript

Elsevier `elsarticle` source for the paper. Compiles against the figures in `../figures/`.

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before submission

1. Replace the author block, affiliations and e-mail addresses.
2. Adjust the CRediT statement to reflect the actual division of work.
3. Add funding acknowledgements if applicable.
4. Update the data-availability statement if the code repository URL is public.
5. Verify the bibliographic details of entries that were reconstructed from abstracts:
   `khanzadeh2018`, `tian2020`, `guo2023`, `farea2025`, `era2025`, `gawade2025`.
   DOIs were intentionally omitted where they could not be confirmed.
6. `schwerdtfeger2020` is not cited in the text and can be deleted from `refs.bib`.

## Figures

| File | Content |
|---|---|
| `fig1_dataset.png` | Representative frames and the per-layer label distribution |
| `fig2_protocol.png` | Model ranking under two protocols; accuracy is protocol-blind |
| `fig3_ablation.png` | Feature combinations and group-wise ablation |
| `fig4_robustness.png` | Multi-seed, first-layer exclusion, layer-level results |
| `fig5_shap.png` | SHAP importance and quintile pore rates |
| `fig6_decision.png` | Cost-sensitive operating points and abstention |

Regenerate with `make figures` from the repository root.
