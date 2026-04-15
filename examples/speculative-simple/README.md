# llama.cpp/examples/speculative-simple

Demonstration of basic greedy speculative decoding

```bash
./bin/llama-speculative-simple \
    -m  ../models/qwen2.5-32b-coder-instruct/ggml-model-q8_0.gguf \
    -md ../models/qwen2.5-1.5b-coder-instruct/ggml-model-q4_0.gguf \
    -f test.txt -c 0 -ngl 99 --color \
    --sampling-seq k --top-k 1 -fa --temp 0.0 \
    -ngld 99 --draft-max 16 --draft-min 5 --draft-p-min 0.9
```

Use `--spec-steward` to move speculative decode-driving work off the high-level orchestrator and onto two dedicated steward threads: one for draft decode and one for target/verify decode. This keeps `main` out of GGML worker 0 and gives draft and target stable, distinct worker-0 thread IDs while preserving the existing default behavior when the flag is omitted.

In non-OpenMP CPU builds, steward mode creates the example-owned target and draft threadpools in the paused state and pauses the relevant pool before each decode so worker-0 affinity is reapplied on the dedicated draft or target steward thread. In OpenMP builds, the orchestrator still stays out of the compute region, but worker affinity continues to follow the existing OpenMP implementation rather than the pthread pause/resume path.
