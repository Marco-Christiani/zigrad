# Colors if:
#   - stdout is a TTY
#   - NO_COLOR is not set
#   - TERM is not "dumb"
COLOR := $(shell [ -t 1 ] && [ -z "$$NO_COLOR" ] && [ "$$TERM" != "dumb" ] && echo yes || echo no)

# May or may not have
TPUT := $(shell command -v tput 2>/dev/null || true)

ifeq ($(COLOR),yes)
  ifneq ($(TPUT),)
    COLORS := $(shell tput colors 2>/dev/null || echo 0)
    BOLD   := $(shell tput bold)
    # dim w fallback.
    DIM    := $(shell tput dim 2>/dev/null || printf '')
    RESET  := $(shell tput sgr0)

    # 8 standard colors
    RED    := $(shell tput setaf 1)
    GREEN  := $(shell tput setaf 2)
    YELLOW := $(shell tput setaf 3)
    BLUE   := $(shell tput setaf 4)
    MAGENTA:= $(shell tput setaf 5)
    CYAN   := $(shell tput setaf 6)

    # Prefer 256, fall back to yellow
    ifeq ($(shell [ "$(COLORS)" -ge 256 ] && echo yes || echo no),yes)
      OR := $(shell tput setaf 208)
    else
      OR := $(YELLOW)
    endif
  else
    # No tput...
    ESC    := $(shell printf '\033')
    BOLD   := $(ESC)[1m
    DIM    := $(ESC)[2m
    RESET  := $(ESC)[0m
    RED    := $(ESC)[31m
    GREEN  := $(ESC)[32m
    YELLOW := $(ESC)[33m
    BLUE   := $(ESC)[34m
    MAGENTA:= $(ESC)[35m
    CYAN   := $(ESC)[36m
    OR     := $(ESC)[38;5;208m
  endif
else
  # No color...
  BOLD   :=
  DIM    :=
  RESET  :=
  RED    :=
  GREEN  :=
  YELLOW :=
  BLUE   :=
  MAGENTA:=
  CYAN   :=
  OR     :=
endif

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
DEFAULT_ZIG_VERSION := 0.14.1 # if we have to install it


define PROJECT_LOGO
$(OR)    ______ _____  ______  ______ _______ ______   $(RESET)
$(OR)     ____/   |   |  ____ |_____/ |_____| |     \  $(RESET)
$(OR)    /_____ __|__ |_____| |    \_ |     | |_____/  $(RESET)
$(OR)
endef
export PROJECT_LOGO

.PHONY: all intro ensure-zig clean build run py-bindings help
all: intro run

intro:
	@printf '%b\n' "$$PROJECT_LOGO"

ensure-zig:
ifndef ZIG
	@printf '%b\n' -n "$(YELLOW)$(BOLD)Zig not found. Would you like to install it using zvm? [y/N] $(RESET)"
	@read -p "" response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		printf '%b\n' "$(CYAN)Checking for curl...$(RESET)"; \
		if ! command -v curl > /dev/null; then \
			printf '%b\n' "$(RED)curl not found. Please install curl first.$(RESET)"; \
			exit 1; \
		fi; \
		printf '%b\n' "$(CYAN)Installing zvm...$(RESET)"; \
		ZVM_OUT=$$(curl -fsSL https://raw.githubusercontent.com/tristanisham/zvm/$(ZVM_HASH)/install.sh | bash); \
		if [ $$? -ne 0 ]; then \
			printf '%b\n' "$(RED)Failed to install zvm$(RESET)"; \
			exit 1; \
		fi; \
		PROFILE_PATH=$$(printf '%b\n' "$$ZVM_OUT" | grep -o '/[^[:space:]]*profile' | head -n1); \
		if [ -f "$$PROFILE_PATH" ]; then \
			. "$$PROFILE_PATH"; \
			zvm install $(DEFAULT_ZIG_VERSION) || { printf '%b\n' "$(RED)Failed to install Zig$(RESET)"; exit 1; }; \
			printf '%b\n' "$(GREEN)✓ Zig installed with zvm$(RESET)"; \
			printf '%b\n' "$(YELLOW)⚠ Please source your profile or restart your shell and re-run.$(RESET)"; \
			exit 1; \
		else \
			printf '%b\n' "$(RED)⚠ Failed to locate profile path, restart your shell or source your profile accordingly before re-running.$(RESET)"; \
			exit 1; \
		fi; \
	else \
		printf '%b\n' "$(RED)⚠ Zig installation cancelled. Please install Zig manually.$(RESET)"; \
		exit 1; \
	fi
endif

clean:
	@printf '%b\n' "$(RED)$(BOLD)Cleaning up project directories...$(RESET)"
	@printf '%b\n' "$(DIM)(Removing build artifacts and dataset)$(RESET)"
	@rm -rf zig-cache zig-out
	@printf '%b\n' "$(GREEN)✓ Done$(RESET)"

./zig-out/bin/dqn: build.zig ensure-zig
	@printf '%b\n' "$(CYAN)$(BOLD)Building DQN example BUILD_MODE=$(BUILD_MODE)...$(RESET)"
	@zig build -Doptimize=$(BUILD_MODE)
	@printf '%b\n' "$(GREEN)✓ Done$(RESET)"

build: ./zig-out/bin/dqn

run: ./zig-out/bin/dqn
	@printf '%b\n' "$(CYAN)$(BOLD)Running DQN...$(RESET)"
	@./zig-out/bin/dqn
	@printf '%b\n' "$(GREEN)✓ Done$(RESET)"

py-bindings:
	cd py && zig build-lib cartpole_wrapper.zig -Doptimize=ReleaseFast -dynamic
	cd py && uv run torchdqn.py

help:
	@printf '%b\n' "$$PROJECT_LOGO"
	@printf '%b\n' "$(BOLD)Welcome to Zigrad's DQN demo.$(RESET)"
	@printf '%b\n' "$(CYAN)Requirements:$(RESET)"
	@printf '%b\n' "  Zig  - will be installed automatically if missing"
	@printf '%b\n' "  BLAS - see readme if the build fails due to missing BLAS"
	@printf '%b\n' "\n$(CYAN)Available commands:$(RESET)"
	@printf '%b\n' "  $(YELLOW)make$(RESET) [BUILD_MODE=<mode>]            - Setup environment and run simple demo"
	@printf '%b\n' "  $(YELLOW)make build$(RESET) [BUILD_MODE=<mode>]      - Build both DQN examples"
	@printf '%b\n' "  $(YELLOW)make run$(RESET) [BUILD_MODE=<mode>]        - Build if needed and run DQN training"
	@printf '%b\n' "  $(YELLOW)make clean$(RESET)                          - Clean up build artifacts"
	@printf '%b\n' "\n$(CYAN)Environment variables:$(RESET)"
	@printf '%b\n' "  $(YELLOW)BUILD_MODE$(RESET)  - Set build optimization mode (default: ReleaseFast)"
	@printf '%b\n' "\n$(CYAN)Build modes:$(RESET)"
	@printf '%b\n' "$(DIM)  Debug        - Fast compilation, extra checks, debug info, debug logging$(RESET)"
	@printf '%b\n' "$(DIM)  ReleaseSmall - Size optimized$(RESET)"
	@printf '%b\n' "$(DIM)  ReleaseFast  - Maximum performance, minimal runtime checks, minimal logging$(RESET)"
	@printf '%b\n' "\n$(BOLD)First time? Start with:$(RESET)"
	@printf '%b\n' "  $(GREEN)make$(RESET) [BUILD_MODE=<mode>]"
