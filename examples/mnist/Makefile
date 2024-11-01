CYAN := \033[36m
YELLOW := \033[33m
GREEN := \033[32m
RED := \033[31m
BLUE := \033[34m
MAGENTA := \033[35m
RESET := \033[0m
BOLD := \033[1m
DIM := \033[2m

OR := \033[38;5;208m

.PHONY: all clean build run-simple run-timing help
VENV := venv
PYTHON := $(VENV)/bin/python3
ZG_DATA_DIR ?= data
BUILD_MODE ?= ReleaseFast

define PROJECT_LOGO
$(OR)    ______ _____  ______  ______ _______ ______   
$(OR)     ____/   |   |  ____ |_____/ |_____| |     \  
$(OR)    /_____ __|__ |_____| |    \_ |     | |_____/  
$(OR)
$(RESET)
endef
export PROJECT_LOGO

all: setup-python run-simple

# Create venv
$(VENV):
	@echo "$$PROJECT_LOGO"
	@echo "$(CYAN)$(BOLD)Creating Python virtual environment...$(RESET)"
	@python3 -m venv $(VENV)
	@echo "$(DIM)(This isolates project dependencies from your system Python)$(RESET)"
	@echo "$(GREEN)✓ Done$(RESET)"

# Install pydeps in venv
setup-python: $(VENV)
	@echo "$(CYAN)$(BOLD)Checking and installing Python dependencies...$(RESET)"
	@$(PYTHON) -c "import torch" 2>/dev/null || \
		(echo "$(YELLOW)Installing PyTorch...$(RESET)" && \
		$(PYTHON) -m pip install torch)
	@$(PYTHON) -c "import torchvision" 2>/dev/null || \
		(echo "$(YELLOW)Installing TorchVision...$(RESET)" && \
		$(PYTHON) -m pip install torchvision)
	
	@echo "$(YELLOW)→ Downloading MNIST data to $(ZG_DATA_DIR)...$(RESET)"
	@ZG_DATA_DIR=$(ZG_DATA_DIR) $(PYTHON) mnist_data.py
	@echo "$(GREEN)✓ Done$(RESET)"

clean:
	@echo "$(RED)$(BOLD)Cleaning up project directories...$(RESET)"
	@echo "$(DIM)(Removing virtual environment, build artifacts, and dataset)$(RESET)"
	@rm -rf $(VENV)
	@rm -rf zig-cache zig-out
	@rm -rf $(ZG_DATA_DIR)
	@echo "$(GREEN)✓ Done$(RESET)"

./zig-out/bin/timing ./zig-out/bin/simple: build.zig
	@echo "$(CYAN)$(BOLD)Building MNIST example BUILD_MODE=$(BUILD_MODE)...$(RESET)"
	@ZG_DATA_DIR=$(ZG_DATA_DIR) zig build -Doptimize=$(BUILD_MODE)
	@echo "$(GREEN)✓ Done$(RESET)"

build: ./zig-out/bin/timing ./zig-out/bin/simple

run-timing: ./zig-out/bin/timing
	@echo "$(CYAN)$(BOLD)Running MNIST with timing...$(RESET)"
	@ZG_DATA_DIR=$(ZG_DATA_DIR) ./zig-out/bin/timing
	@echo "$(GREEN)✓ Done$(RESET)"

run-simple: ./zig-out/bin/simple
	@echo "$(CYAN)$(BOLD)Running MNIST...$(RESET)"
	@ZG_DATA_DIR=$(ZG_DATA_DIR) ./zig-out/bin/simple
	@echo "$(GREEN)✓ Done$(RESET)"

help:
	@echo "$$PROJECT_LOGO"
	@echo "$(BOLD)Welcome to Zigrad's MNIST demo!$(RESET)"
	@echo "$(CYAN)Available commands:$(RESET)"
	@echo "  $(YELLOW)make$(RESET) [BUILD_MODE=<mode>] - Setup environment and run simple demo"
	@echo "  $(YELLOW)make build$(RESET) [BUILD_MODE=<mode>] - Build both MNIST examples"
	@echo "  $(YELLOW)make run-simple$(RESET) [BUILD_MODE=<mode>] - Build if needed and run basic MNIST training"
	@echo "  $(YELLOW)make run-timing$(RESET) [BUILD_MODE=<mode>] - Build if needed and run MNIST with timing info"
	@echo "  $(YELLOW)make clean$(RESET) - Clean up virtual environment and build artifacts"
	@echo "\n$(CYAN)Environment variables:$(RESET)"
	@echo "  $(YELLOW)ZG_DATA_DIR$(RESET) - Set custom data directory (default: 'data')"
	@echo "  $(YELLOW)BUILD_MODE$(RESET) - Set build optimization mode (default: ReleaseFast)"
	@echo "\n$(CYAN)Build modes:$(RESET)"
	@echo "$(DIM)  Debug        - Fast compilation, extra checks, debug info, debug logging$(RESET)"
	@echo "$(DIM)  ReleaseSmall - Size optimized$(RESET)"
	@echo "$(DIM)  ReleaseFast  - Maximum performance, minimal runtime checks, minimal logging$(RESET)"
	@echo "\n$(BOLD)First time? Start with:$(RESET)"
	@echo "  $(GREEN)make$(RESET) [BUILD_MODE=<mode>] - This will set up everything and run the simple demo"
	@echo "$(DIM)(The setup process will create a python virtual environment and download MNIST data)$(RESET)"
