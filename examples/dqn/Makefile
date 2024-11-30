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

PYTHON := python3
BUILD_MODE ?= ReleaseFast
ZVM_HASH=233a41c579fd2eeb602d51efa10499e083cd64d0
ZIG := $(shell which zig 2>/dev/null)
DEFAULT_ZIG_VERSION := 0.13.0 # if we have to install it


define PROJECT_LOGO
$(OR)    ______ _____  ______  ______ _______ ______   
$(OR)     ____/   |   |  ____ |_____/ |_____| |     \  
$(OR)    /_____ __|__ |_____| |    \_ |     | |_____/  
$(OR)
$(RESET)
endef
export PROJECT_LOGO

.PHONY: all intro ensure-zig clean build run py-bindings help
all: intro run

intro:
	@echo "$$PROJECT_LOGO"

ensure-zig:
ifndef ZIG
	@echo -n "$(YELLOW)$(BOLD)Zig not found. Would you like to install it using zvm? [y/N] $(RESET)"
	@read -p "" response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		echo "$(CYAN)Checking for curl...$(RESET)"; \
		if ! command -v curl > /dev/null; then \
			echo "$(RED)curl not found. Please install curl first.$(RESET)"; \
			exit 1; \
		fi; \
		echo "$(CYAN)Installing zvm...$(RESET)"; \
		ZVM_OUT=$$(curl -fsSL https://raw.githubusercontent.com/tristanisham/zvm/$(ZVM_HASH)/install.sh | bash); \
		if [ $$? -ne 0 ]; then \
			echo "$(RED)Failed to install zvm$(RESET)"; \
			exit 1; \
		fi; \
		PROFILE_PATH=$$(echo "$$ZVM_OUT" | grep -o '/[^[:space:]]*profile' | head -n1); \
		if [ -f "$$PROFILE_PATH" ]; then \
			. "$$PROFILE_PATH"; \
			zvm install $(DEFAULT_ZIG_VERSION) || { echo "$(RED)Failed to install Zig$(RESET)"; exit 1; }; \
			echo "$(GREEN)✓ Zig installed with zvm$(RESET)"; \
			echo "$(YELLOW)⚠ Please source your profile or restart your shell and re-run.$(RESET)"; \
			exit 1; \
		else \
			echo "$(RED)⚠ Failed to locate profile path, restart your shell or source your profile accordingly before re-running.$(RESET)"; \
			exit 1; \
		fi; \
	else \
		echo "$(RED)⚠ Zig installation cancelled. Please install Zig manually.$(RESET)"; \
		exit 1; \
	fi
endif

clean:
	@echo "$(RED)$(BOLD)Cleaning up project directories...$(RESET)"
	@echo "$(DIM)(Removing build artifacts and dataset)$(RESET)"
	@rm -rf zig-cache zig-out
	@echo "$(GREEN)✓ Done$(RESET)"

./zig-out/bin/dqn: build.zig ensure-zig
	@echo "$(CYAN)$(BOLD)Building DQN example BUILD_MODE=$(BUILD_MODE)...$(RESET)"
	@zig build -Doptimize=$(BUILD_MODE)
	@echo "$(GREEN)✓ Done$(RESET)"

build: ./zig-out/bin/dqn

run: ./zig-out/bin/dqn
	@echo "$(CYAN)$(BOLD)Running DQN...$(RESET)"
	@./zig-out/bin/dqn
	@echo "$(GREEN)✓ Done$(RESET)"

py-bindings:
	zig build-lib cartpole_wrapper.zig -Doptimize=ReleaseFast -dynamic

help:
	@echo "$$PROJECT_LOGO"
	@echo "$(BOLD)Welcome to Zigrad's DQN demo.$(RESET)"
	@echo "$(CYAN)Requirements:$(RESET)"
	@echo "  Zig  - will be installed automatically if missing"
	@echo "  BLAS - see readme if the build fails due to missing BLAS"
	@echo "\n$(CYAN)Available commands:$(RESET)"
	@echo "  $(YELLOW)make$(RESET) [BUILD_MODE=<mode>]            - Setup environment and run simple demo"
	@echo "  $(YELLOW)make build$(RESET) [BUILD_MODE=<mode>]      - Build both DQN examples"
	@echo "  $(YELLOW)make run$(RESET) [BUILD_MODE=<mode>]        - Build if needed and run DQN training"
	@echo "  $(YELLOW)make clean$(RESET)                          - Clean up build artifacts"
	@echo "\n$(CYAN)Environment variables:$(RESET)"
	@echo "  $(YELLOW)BUILD_MODE$(RESET)  - Set build optimization mode (default: ReleaseFast)"
	@echo "\n$(CYAN)Build modes:$(RESET)"
	@echo "$(DIM)  Debug        - Fast compilation, extra checks, debug info, debug logging$(RESET)"
	@echo "$(DIM)  ReleaseSmall - Size optimized$(RESET)"
	@echo "$(DIM)  ReleaseFast  - Maximum performance, minimal runtime checks, minimal logging$(RESET)"
	@echo "\n$(BOLD)First time? Start with:$(RESET)"
	@echo "  $(GREEN)make$(RESET) [BUILD_MODE=<mode>]"
