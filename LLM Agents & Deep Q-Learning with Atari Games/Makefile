# ============================================================================
# Makefile — Deep Q-Learning Agent for Ms. Pac-Man
# ============================================================================
# Common commands for development, testing, and training.
# Usage: make <target>
# ============================================================================

.PHONY: help setup test lint format typecheck analyze train train-short \
        train-boltzmann experiments evaluate play clean

PYTHON := python3
PIP := pip3
SRC := dqn_mspacman.py

# Default target — show available commands
help:  ## Show this help message
	@echo ""
	@echo "  Deep Q-Learning Agent for Ms. Pac-Man"
	@echo "  ======================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---- Setup ----

setup:  ## Install all dependencies
	$(PIP) install torch torchvision
	$(PIP) install 'gymnasium[atari,accept-rom-license]' numpy matplotlib moviepy Pillow
	@echo "\n✓ Setup complete."

# ---- Quality ----

test:  ## Run smoke tests on all components
	$(PYTHON) $(SRC) --mode test

lint:  ## Run flake8 linter
	$(PYTHON) -m flake8 $(SRC) --max-line-length 88 --extend-ignore E203,W503

format:  ## Auto-format code with Black
	$(PYTHON) -m black $(SRC) --line-length 88

typecheck:  ## Run mypy type checker
	$(PYTHON) -m mypy $(SRC) --ignore-missing-imports

# ---- Running ----

analyze:  ## Analyze Ms. Pac-Man environment (instant)
	$(PYTHON) $(SRC) --mode analyze

train:  ## Train with default config (5000 episodes)
	$(PYTHON) $(SRC) --mode train

train-short:  ## Quick training demo (20 episodes)
	$(PYTHON) $(SRC) --mode train --episodes 20

train-boltzmann:  ## Train with Boltzmann exploration
	$(PYTHON) $(SRC) --mode train --policy boltzmann

experiments:  ## Run full ablation experiment suite
	$(PYTHON) $(SRC) --mode all_experiments

evaluate:  ## Evaluate best model with video recording
	$(PYTHON) $(SRC) --mode evaluate --model models/best_model.pt

play:  ## Record gameplay GIF (random agent)
	$(PYTHON) $(SRC) --mode play

play-trained:  ## Record gameplay GIF (trained agent)
	$(PYTHON) $(SRC) --mode play --model models/best_model.pt

# ---- Cleanup ----

clean:  ## Remove generated artifacts (models, results, cache)
	rm -rf models/ results/ videos/ __pycache__/ .mypy_cache/
	find . -name "*.pyc" -delete
	@echo "✓ Cleaned."
