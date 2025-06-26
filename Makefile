# field‑edge‑rag‑sentry — Build & test helpers
# Usage:
#   make            # build llama.cpp + FAISS + CoreML vision smoke demo
#   make deps       # install Homebrew prereqs
#   make llama      # clone & compile llama.cpp with Metal
#   make test-faiss # run 2 k‑vector FAISS retrieval demo
#   make vision     # run realtime YOLO‑v5s CoreML webcam demo
#   make test       # run all tests in the tests directory
#   make clean      # clean llama.cpp artifacts

BREW_PKGS = cmake faiss libomp

.PHONY: all deps llama test-faiss vision index generate lora rag-server rag-client eval clean test install-deps download-model

all: install-deps llama index generate test-faiss vision

deps:
	brew install $(BREW_PKGS)

# ---------------------------------------------------------------------------
# Install Python dependencies with proper Metal compilation
# ---------------------------------------------------------------------------
install-deps: deps
	pip install -r requirements.txt
	CMAKE_ARGS="-DGGML_METAL=ON -DGGML_NATIVE=OFF -DCMAKE_OSX_ARCHITECTURES=arm64" \
	pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

# ---------------------------------------------------------------------------
# Build llama.cpp with Metal backend (≈15‑25 tok/s on M‑series GPU)
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
# FAISS smoke test — index 2 k random vectors & query nearest neighbours
# ---------------------------------------------------------------------------
test-faiss: deps src/faiss_smoke.py
	python src/faiss_smoke.py

# ---------------------------------------------------------------------------
# Realtime YOLO‑v5s CoreML webcam demo (≥15 FPS)
# ---------------------------------------------------------------------------
vision: deps src/vision_demo.py
	python src/vision_demo.py

# ---------------------------------------------------------------------------
# Vision + RAG bridge
# ---------------------------------------------------------------------------
vision-rag: deps index download-model src/vision_rag.py
	python src/vision_rag.py

# ---------------------------------------------------------------------------
# Build FAISS index from doctrine PDFs
# ---------------------------------------------------------------------------
index: deps src/build_index.py
	python src/build_index.py

# ---------------------------------------------------------------------------
# Download Llama-2-7B-Chat Q4_0 model
# ---------------------------------------------------------------------------
download-model:
	mkdir -p models
	@if [ ! -f models/llama-2-7b-chat-q4_0.gguf ]; then \
		echo "Downloading Llama-2-7B-Chat Q4_0 model..."; \
		huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_0.gguf --local-dir models/ && \
		mv models/llama-2-7b-chat.Q4_0.gguf models/llama-2-7b-chat-q4_0.gguf || \
		curl -L -o models/llama-2-7b-chat-q4_0.gguf "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"; \
	else \
		echo "Model already exists at models/llama-2-7b-chat-q4_0.gguf"; \
	fi

# ---------------------------------------------------------------------------
# End-to-end answer generation (retrieval + llama.cpp)
# ---------------------------------------------------------------------------
generate: deps index download-model src/generate.py
	python src/generate.py "Explain mission-type orders."

# ---------------------------------------------------------------------------
# LoRA fine‑tuning on synthetic Q‑A
# ---------------------------------------------------------------------------
lora: deps index src/finetune_lora.py
	python src/finetune_lora.py

# ---------------------------------------------------------------------------
# ZeroMQ RAG service
# ---------------------------------------------------------------------------
rag-server: deps index download-model src/rag_server.py
	python src/rag_server.py

# Client helper
rag-client: deps src/rag_client.py
	python src/rag_client.py "What is the purpose of the Army's mission-type orders?"

# ---------------------------------------------------------------------------
# Ragas evaluation over 25‑Q dev set
# ---------------------------------------------------------------------------
eval: deps index download-model src/eval_ragas.py
	python src/eval_ragas.py

# ---------------------------------------------------------------------------
# Run all tests in the tests directory
# ---------------------------------------------------------------------------
test:
	python -m pytest tests/

clean:
	@if [ -d third_party/llama.cpp/build ]; then rm -rf third_party/llama.cpp/build; fi