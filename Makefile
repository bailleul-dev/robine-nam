# robine-nam — everything audio-related runs in release builds only.

# Project-local environment (TONE3000 keys, asset dir): see .env.example.
ifneq (,$(wildcard .env))
include .env
export
endif

NAM_LAB      := tools/target/release/nam-lab
FIXTURES     := crates/robine-nam/tests/fixtures
ASSETS       := $(if $(ROBINE_NAM_ASSETS_DIR),$(ROBINE_NAM_ASSETS_DIR),assets)
MODEL        ?= $(FIXTURES)/A2.nam
INPUT        ?= $(ASSETS)/inputs/Brit - Guitar.wav

.PHONY: all build test tools compare bench golden assets assets-all fmt clippy clean aidecl-check listen

all: build tools

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
	cd tools && cargo fmt --all

clippy:
	cargo clippy --release --all-targets -- -D warnings
	cd tools && cargo clippy --release --all-targets

## Measurement tools (separate workspace; builds the C++ upstream reference)
tools:
	tools/nam-reference-sys/vendor.sh
	cd tools && cargo build --release

## Sample-exact delta against the upstream reference (exit code != 0 on failure)
compare: tools
	$(NAM_LAB) compare -m "$(MODEL)" --seconds 5

compare-wav: tools
	$(NAM_LAB) compare -m "$(MODEL)" -i "$(INPUT)" --blocks 64

## Throughput / latency of robine-nam vs upstream fast and generic paths
bench: tools
	$(NAM_LAB) bench -m "$(MODEL)" --json target/bench.json

## Listening set: robine vs upstream renders, diff and blind A/B WAVs in assets/listening
listen: tools
	python3 -I scripts/listening_set.py $(LISTEN_ARGS)

## Regenerate the golden fixtures used by `cargo test`
golden: tools
	$(NAM_LAB) golden -m $(FIXTURES)/A2.nam -o $(FIXTURES) --name a2

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
	cd tools && cargo clean

## Validate the AI usage declaration (https://ai-declaration.org/)
aidecl-check:
	uvx --from git+https://github.com/ai-declaration/cli.git aidecl --strict validate aidecl.yaml
