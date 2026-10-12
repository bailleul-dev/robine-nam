# robine-nam — everything audio-related runs in release builds only.

# The measurement tools (comparison with the upstream reference, benchmarks,
# golden fixtures, listening sets) are in ../robine-wavelab:
#   make -C ../robine-wavelab nam-compare | nam-bench | nam-golden | nam-listen

# Project-local environment (TONE3000 keys, asset dir): see .env.example.
ifneq (,$(wildcard .env))
include .env
export
endif

ASSETS       := $(if $(ROBINE_NAM_ASSETS_DIR),$(ROBINE_NAM_ASSETS_DIR),assets)

.PHONY: all build test assets assets-all tone fmt clippy clean aidecl-check

all: build

## Library
build:
	cargo build --release

test:
	cargo fmt --all --check
	cargo clippy --release --all-targets -- -D warnings
	cargo test
	cargo test --release
	cargo test --release --features scalar
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

fmt:
	cargo fmt --all

clippy:
	cargo clippy --release --all-targets -- -D warnings

## Test assets (public DI inputs, IRs, upstream example models)
assets:
	python3 -I scripts/download_assets.py --output "$(ASSETS)"

assets-all:
	python3 -I scripts/download_assets.py --output "$(ASSETS)" --all-inputs

## TONE3000 captures: make tone TONE=6580 (device-flow login on first use)
tone:
	python3 -I scripts/download_assets.py --output "$(ASSETS)" --no-inputs --no-irs --no-upstream --tone "$(TONE)"

clean:
	cargo clean

## Validate the AI usage declaration (https://ai-declaration.org/)
aidecl-check:
	uvx --from git+https://github.com/ai-declaration/cli.git aidecl --strict validate aidecl.yaml
