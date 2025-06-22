# field‑edge‑rag‑sentry — Build & test helpers
# Usage:
#   make            # build llama.cpp + FAISS + CoreML vision smoke demo
#   make deps       # install Homebrew prereqs
#   make llama      # clone & compile llama.cpp with Metal
#   make test-faiss # run 2 k‑vector FAISS retrieval demo
#   make vision     # run realtime YOLO‑v5s CoreML webcam demo
#   make clean      # clean llama.cpp artifacts

BREW_PKGS = cmake faiss

.PHONY: all deps llama test-faiss vision clean

all: llama test-faiss vision

deps:
	brew install $(BREW_PKGS)

# ---------------------------------------------------------------------------
# Build llama.cpp with Metal backend (≈15‑25 tok/s on M‑series GPU)
# ---------------------------------------------------------------------------
llama: deps
	mkdir -p third_party
	@if [ ! -d third_party/llama.cpp ]; then \
		git clone --depth 1 https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp; \
	fi
	rm -rf third_party/llama.cpp/build
	cd third_party/llama.cpp && mkdir -p build && cd build && \
	cmake .. -DGGML_METAL=ON -DCMAKE_TOOLCHAIN_FILE=../cmake/arm64-apple-clang.cmake -DCMAKE_ASM_FLAGS="-arch arm64" && \
	cmake --build . --config Release

# ---------------------------------------------------------------------------
# FAISS smoke test — index 2 k random vectors & query nearest neighbours
# ---------------------------------------------------------------------------
test-faiss: deps scripts/faiss_smoke.py
	python scripts/faiss_smoke.py

# ---------------------------------------------------------------------------
# Realtime YOLO‑v5s CoreML webcam demo (≥15 FPS)
# ---------------------------------------------------------------------------
vision: deps scripts/vision_demo.py
	python scripts/vision_demo.py

clean:
	@if [ -d third_party/llama.cpp/build ]; then rm -rf third_party/llama.cpp/build; fi