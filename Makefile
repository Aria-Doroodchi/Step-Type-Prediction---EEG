# =====================================================================
# Step Type Prediction (EEG) — pipeline targets.
#
#   make install        editable-install the package
#   make smoke          fast end-to-end check (1 participant, logistic, tiny grids)
#   make test           run pytest (imports + smoke pipeline)
#   make preprocess     stage 1: raw .bdf → cleaned epochs
#   make src            stage 2: epochs → source-localized CSVs
#   make features       stage 3: epochs+src → feature parquets
#   make train MODEL=xgb  stage 4: features → metrics CSV (xgb/svm/lstm/logistic)
#   make all            preprocess → src → features → train (default model)
#   make clean          remove generated data + outputs (NOT raw)
# =====================================================================

PYTHON  ?= python
CONFIG  ?= configs/default.yaml
SMOKE   ?= configs/smoke.yaml
MODEL   ?= xgb

# ---- Install -------------------------------------------------------
install:
	$(PYTHON) -m pip install -e .

# ---- Smoke test ----------------------------------------------------
smoke:
	$(PYTHON) run.py --config $(SMOKE) --model logistic

test:
	$(PYTHON) -m pytest -q

# ---- Stages --------------------------------------------------------
preprocess:
	$(PYTHON) scripts/01_preprocess.py --config $(CONFIG)

src: preprocess
	$(PYTHON) scripts/02_source_localize.py --config $(CONFIG)

features: src
	$(PYTHON) scripts/03_extract_features.py --config $(CONFIG)

train: features
	$(PYTHON) scripts/04_train.py --config $(CONFIG) --model $(MODEL)

all: train

# ---- Cleanup -------------------------------------------------------
clean:
	rm -rf data/interim data/features data/src outputs/runs

.PHONY: install smoke test preprocess src features train all clean
