
# Variables
VENV_DIR = venv
REQ_FILE = requirements.txt

# Create virtual environment
.PHONY: venv
venv:
	python3 -m venv $(VENV_DIR)

# Activate virtual environment and install dependencies
.PHONY: install
install: venv
	. $(VENV_DIR)/bin/activate && pip install -r $(REQ_FILE)

# Run OCR Algorithm
.PHONY: run-ocr
run-ocr:
	. $(VENV_DIR)/bin/activate && python OCR_Algorithm.py

# Run Line Sweep Algorithm
.PHONY: run-linesweep
run-linesweep:
	. $(VENV_DIR)/bin/activate && python lineSweepDetect.py

# Run Connected Components Algorithm
.PHONY: run-connected
run-connected:
	. $(VENV_DIR)/bin/activate && python connectedComponent.py

# Clean virtual environment
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)