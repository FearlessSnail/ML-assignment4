# ML-assignment4
Repository for assignment 4 of the Machine Learning module

Tunes a random forest on the UNSW-NB15 network traffic data and compares it with
the k-nearest neighbours and classification tree models of assignment 2, on the
same five cross-validation folds.

```bash
pip install -r requirements.txt
python src/main.pyv # full run (about 45 minutes on 8 cores)
python src/main.py --replot # redraw figures from out/ without refitting
python src/main.py --sample 8000 --tune-sampl  4000 --output-dir /tmp/smoke # quick check
```