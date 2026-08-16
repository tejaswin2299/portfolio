# Sai Tejaswi Nooka — Updated Professional Portfolio

This GitHub Pages portfolio uses a clear multi-page layout. It includes a professional bio, personal value proposition, target audience, seven documented artifacts, code evidence, real datasets, outputs, references, and responsive navigation.

## Main files

- `index.html` — portfolio homepage
- `timeline.html` — Signals & Symbols artifact
- `pathfinder.html` — ML PathFinder AI artifact
- `neural-networks.html` — Workshop Three neural-networks and adaptive-leadership artifact
- `rice-classifier.html` — Artifact 4, UCI rice-variety classification audit
- `data-audit.html` — Artifact 5, Python Fashion-MNIST evidence artifact
- `change-leader-framework.html` — Artifact 6, Workshop Six personal change-leadership framework
- `occupancy-capstone.html` — separate end-to-end final AI/ML project
- `resume.html` — browser résumé
- `assets/styles.css` and `assets/site.js` — shared design and tab behavior
- `evidence/fashion_mnist_analysis.py` — reproducible Python analysis
- `evidence/rice_variety_analysis.py` — reproducible rice classification analysis
- `evidence/occupancy_capstone_analysis.py` — reproducible four-model capstone analysis
- `evidence/outputs/` — generated charts and metrics
- `evidence/rice_outputs/` — rice model charts and metrics
- `evidence/occupancy_outputs/` — capstone charts, comparison tables, and exact metrics
- `data/fashion_mnist_kaggle_sample.csv` — compact Fashion-MNIST sample
- `data/Rice_Cammeo_Osmancik.arff` — complete UCI rice dataset (3,810 rows)
- `data/Occupancy_Estimation.csv` — complete UCI room occupancy dataset (10,129 rows)

## Run the Python evidence

```bash
pip install -r evidence/requirements.txt
python evidence/fashion_mnist_analysis.py
python evidence/rice_variety_analysis.py
python evidence/occupancy_capstone_analysis.py
```

## Preview locally

From this folder, run `python -m http.server 8000`, then open
`http://localhost:8000/`. The project uses relative links and can be published
directly through GitHub Pages.
