# Makefile for Rust project using Cargo
#
# Note: `fastembed-static` and `fastembed-dynamic` are mutually exclusive
# ONNX Runtime linking strategies (ort cannot honor both in one build), so
# targets below never use `--all-features`. The no-default-features core is
# tested alongside the default build.

.PHONY: all build check run test test-core bench clippy clippy-fix fmt fmt-check doc update

all: fmt-check clippy

# Build the project with default features enabled in release mode
build:
	cargo build --workspace --release

# Check the project for compilation errors without producing binaries
check:
	cargo check --workspace

# Update dependencies to their latest compatible versions
update:
	cargo update

# Run the project with default features enabled in release mode
run:
	cargo run --release

# Run all tests with default features, then the pure-Rust core without them
test:
	cargo test --workspace
	cargo test -p qmd-rs --no-default-features

# Alias for the second half of `make test`
test-core:
	cargo test -p qmd-rs --no-default-features

# Run benchmarks with default features enabled
bench:
	cargo bench

# Run Clippy linter with nightly toolchain (check only, for CI)
# Uses workspace lints from Cargo.toml
clippy:
	cargo +nightly clippy --workspace \
		--all-targets \
		-- -D warnings

# Run Clippy linter with auto-fix (for development)
clippy-fix:
	cargo +nightly clippy --workspace \
		--fix \
		--all-targets \
		--allow-dirty \
		--allow-staged \
		-- -D warnings

# Format the code using rustfmt with nightly toolchain
fmt:
	cargo +nightly fmt

# Verify formatting without modifying files.
fmt-check:
	cargo fmt --check

# Generate documentation for all crates and open it in the browser
doc:
	cargo +nightly doc --no-deps --open
