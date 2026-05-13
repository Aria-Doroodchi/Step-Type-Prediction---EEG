# =====================================================================
# Step Type Prediction (EEG) — pipeline targets.
#
#   make install        editable-install the package
#   make smoke          fast end-to-end check (1 participant, logistic, tiny grids)
#   make smoke-test     pytest smoke pipeline file
#   make smoke-test-two pytest two-participant full workflow smoke only
#   make test           run pytest (imports + smoke pipeline)
#   make preflight      check external real-data assets before full runs
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
RUN_ID ?=
CHANNEL_MODE ?=
PREDICTION_WINDOW ?=
OVERRIDE_MODE ?=
OVERRIDE_ARGS := $(if $(OVERRIDE_MODE),--participant-override-mode $(OVERRIDE_MODE),)
RUN_ARGS := $(if $(RUN_ID),--run-id $(RUN_ID),)
CHANNEL_ARGS := $(if $(CHANNEL_MODE),--channel-mode $(CHANNEL_MODE),)
WINDOW_ARGS := $(if $(PREDICTION_WINDOW),--prediction-window $(PREDICTION_WINDOW),)

# ---- Install -------------------------------------------------------
install:
	$(PYTHON) -m pip install -e .

# ---- Smoke test ----------------------------------------------------
smoke:
	$(PYTHON) run.py --config $(SMOKE) --model logistic $(OVERRIDE_ARGS) $(CHANNEL_ARGS) $(WINDOW_ARGS)

smoke-test:
	$(PYTHON) -m pytest tests/test_smoke_pipeline.py -q

smoke-test-two:
	$(PYTHON) -m pytest tests/test_smoke_pipeline.py::test_two_participant_full_workflow_smoke -q

test:
	$(PYTHON) -m pytest -q

preflight:
	$(PYTHON) scripts/00_preflight.py --config $(CONFIG)

# ---- Stages --------------------------------------------------------
preprocess:
	$(PYTHON) scripts/01_preprocess.py --config $(CONFIG) $(OVERRIDE_ARGS)

src: preprocess
	$(PYTHON) scripts/02_source_localize.py --config $(CONFIG) $(OVERRIDE_ARGS)

features: src
	$(PYTHON) scripts/03_extract_features.py --config $(CONFIG) $(OVERRIDE_ARGS) $(WINDOW_ARGS)

train: features
	$(PYTHON) scripts/04_train.py --config $(CONFIG) --model $(MODEL) $(RUN_ARGS) $(OVERRIDE_ARGS) $(CHANNEL_ARGS) $(WINDOW_ARGS)

all: preflight train

full-xgb: preflight
	$(PYTHON) run.py --config $(CONFIG) --model xgb $(RUN_ARGS) $(OVERRIDE_ARGS) $(CHANNEL_ARGS) $(WINDOW_ARGS)

# ---- Cleanup -------------------------------------------------------
clean:
	rm -rf data/interim data/features data/src outputs/runs

.PHONY: install smoke smoke-test smoke-test-two test preflight preprocess src features train all full-xgb clean
