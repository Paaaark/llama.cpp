#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using ggml_threadpool_new_fn_t = struct ggml_threadpool * (*)(struct ggml_threadpool_params *);
using ggml_threadpool_pause_fn_t = void (*)(struct ggml_threadpool *);

struct speculative_threadpools {
    ggml_threadpool * target       = nullptr;
    ggml_threadpool * target_batch = nullptr;
    ggml_threadpool * draft        = nullptr;
    ggml_threadpool * draft_batch  = nullptr;

    ggml_threadpool_pause_fn_t pause_fn = nullptr;
    bool use_phase_pause = false;

    void pause_pool(ggml_threadpool * pool) const {
        if (use_phase_pause && pause_fn != nullptr && pool != nullptr) {
            pause_fn(pool);
        }
    }

    void pause_target_phase() const {
        pause_pool(target);
        if (target_batch != nullptr && target_batch != target) {
            pause_pool(target_batch);
        }
    }

    void pause_draft_phase() const {
        pause_pool(draft);
        if (draft_batch != nullptr && draft_batch != draft) {
            pause_pool(draft_batch);
        }
    }
};

class sequential_steward {
public:
    sequential_steward() : worker_([this]() { run(); }) {}

    ~sequential_steward() {
        stop();
    }

    template <typename Fn>
    auto submit(Fn && fn) -> std::future<std::invoke_result_t<std::decay_t<Fn>>> {
        using task_fn_t = std::decay_t<Fn>;
        using result_t = std::invoke_result_t<task_fn_t>;

        auto promise = std::make_shared<std::promise<result_t>>();
        auto future = promise->get_future();

        auto wrapped = [promise, fn = task_fn_t(std::forward<Fn>(fn))]() mutable {
            try {
                if constexpr (std::is_void_v<result_t>) {
                    fn();
                    promise->set_value();
                } else {
                    promise->set_value(fn());
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        };

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stopping_ || !task_.has_value(); });
            if (stopping_) {
                throw std::runtime_error("steward has stopped");
            }
            task_ = std::move(wrapped);
        }

        cv_.notify_all();
        return future;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
        }

        cv_.notify_all();

        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stopping_ || task_.has_value(); });

                if (stopping_ && !task_.has_value()) {
                    return;
                }

                task = std::move(*task_);
                task_.reset();
            }

            cv_.notify_all();
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<std::function<void()>> task_;
    bool stopping_ = false;
    std::thread worker_;
};

struct speculative_loop {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    const llama_vocab * vocab;
    const common_params & params;
    const speculative_threadpools & threadpools;

    common_sampler * smpl;
    common_speculative * spec;
    llama_batch batch_tgt;
    common_speculative_params params_spec;

    llama_tokens prompt_tgt;
    llama_token id_last = 0;

    int n_input   = 0;
    int n_past    = 0;
    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    int n_draft = 0;
    int n_draft_min = 0;

    bool has_eos = false;

    int64_t t_ttft_start = 0;
    int64_t t_enc_start  = 0;
    int64_t t_enc_end    = 0;
    int64_t t_dec_start  = 0;
    int64_t t_dec_end    = 0;
    int64_t t_ttft_end_us = 0;
    bool ttft_first_token_done = false;

    speculative_loop(
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const llama_vocab * vocab,
            const common_params & params,
            const speculative_threadpools & threadpools,
            int64_t t_ttft_start) :
        ctx_tgt(ctx_tgt),
        ctx_dft(ctx_dft),
        vocab(vocab),
        params(params),
        threadpools(threadpools),
        smpl(common_sampler_init(llama_get_model(ctx_tgt), params.sampling)),
        spec(common_speculative_init(ctx_tgt, ctx_dft, params.speculative.draft_deterministic)),
        batch_tgt(llama_batch_init(llama_n_batch(ctx_tgt), 0, 1)),
        n_draft(params.speculative.n_max),
        n_draft_min(params.speculative.n_min),
        t_ttft_start(t_ttft_start),
        t_enc_start(t_ttft_start) {
        params_spec.n_draft = n_draft;
        params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft;
        params_spec.p_min   = params.speculative.p_min;
        params_spec.early_stop = params.speculative.draft_early_stop;

        for (const auto & pair : params.speculative.replacements) {
            common_speculative_add_replacement_tgt_dft(spec, pair.first.c_str(), pair.second.c_str());
        }
    }

    ~speculative_loop() {
        common_sampler_free(smpl);
        common_speculative_free(spec);
        llama_batch_free(batch_tgt);
    }

    void prefill(std::vector<llama_token> & inp) {
        n_input = static_cast<int>(inp.size());

        threadpools.pause_target_phase();
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));
        llama_synchronize(ctx_tgt);
        t_enc_end = ggml_time_us();

        id_last = inp.back();

        prompt_tgt = llama_tokens(inp.begin(), inp.end() - 1);
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

        n_past = static_cast<int>(inp.size()) - 1;
        t_dec_start = ggml_time_us();
    }

    bool step() {
        threadpools.pause_draft_phase();
        llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);

        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);

        if (draft.size() < static_cast<size_t>(n_draft_min)) {
            draft.clear();
        }

        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + static_cast<int>(i), { 0 }, true);
        }

        threadpools.pause_target_phase();
        llama_decode(ctx_tgt, batch_tgt);

        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);

        if (!ttft_first_token_done) {
            t_ttft_end_us = ggml_time_us();
            ttft_first_token_done = true;
        }

        GGML_ASSERT(ids.size() > 0);

        n_past    += static_cast<int>(ids.size()) - 1;
        n_drafted += static_cast<int>(draft.size());
        n_accept  += static_cast<int>(ids.size()) - 1;
        n_predict += static_cast<int>(ids.size());

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", static_cast<int>(ids.size()) - 1, static_cast<int>(draft.size()), id_last);
        if (params.speculative.accept_stats) {
            LOG_INF("accept_step draft=%zu accept=%zu\n", draft.size(), ids.size() - 1);
        }

        LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);

        const bool done = (params.n_predict >= 0 && n_predict > params.n_predict) || has_eos;
        if (done) {
            t_dec_end = ggml_time_us();
        }

        return done;
    }

    void print_summary() const {
        LOG("\n\n");

        if (ttft_first_token_done && t_ttft_end_us > 0) {
            LOG_INF("ttft_ms = %.2f\n", (t_ttft_end_us - t_ttft_start) / 1000.0f);
        }
        LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, n_input   / ((t_enc_end - t_enc_start) / 1e6f));
        LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

        LOG_INF("\n");
        LOG_INF("n_draft   = %d\n", n_draft);
        LOG_INF("n_predict = %d\n", n_predict);
        LOG_INF("n_drafted = %d\n", n_drafted);
        LOG_INF("n_accept  = %d\n", n_accept);
        LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

        LOG_INF("\n");
        LOG_INF("draft:\n\n");

        llama_perf_context_print(ctx_dft);

        LOG_INF("\n");
        LOG_INF("target:\n\n");
        common_perf_print(ctx_tgt, smpl);
    }
};

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    //llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // save target CPU params before overwriting for draft (used for threadpool affinity)
    cpu_params cpuparams_tgt       = params.cpuparams;
    cpu_params cpuparams_batch_tgt = params.cpuparams_batch;

    // load the draft model (use full speculative cpuparams so draft gets its own threads + affinity)
    params.devices               = params.speculative.devices;
    params.model                 = params.speculative.model;
    params.n_ctx                 = params.speculative.n_ctx;
    params.n_batch               = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
    params.n_gpu_layers          = params.speculative.n_gpu_layers;
    params.cpuparams              = params.speculative.cpuparams;
    params.cpuparams_batch        = params.speculative.cpuparams_batch;
    params.tensor_buft_overrides  = params.speculative.tensor_buft_overrides;

    common_init_result llama_init_dft = common_init_from_params(params);

    //model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
        LOG_INF("the draft model '%s' is not compatible with the target model '%s'. tokens will be translated between the draft and target models.\n", params.speculative.model.path.c_str(), params.model.path.c_str());
    }

    speculative_threadpools threadpools;

    // attach threadpools with CPU affinity (target = main cpuparams, draft = speculative cpuparams)
    {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
            auto ggml_threadpool_new_fn = reinterpret_cast<ggml_threadpool_new_fn_t>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
            auto ggml_threadpool_pause_fn = reinterpret_cast<ggml_threadpool_pause_fn_t>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_pause"));

            if (ggml_threadpool_new_fn) {
                struct ggml_threadpool_params tpp_tgt       = ggml_threadpool_params_from_cpu_params(cpuparams_tgt);
                struct ggml_threadpool_params tpp_tgt_batch = ggml_threadpool_params_from_cpu_params(cpuparams_batch_tgt);
                struct ggml_threadpool_params tpp_dft       = ggml_threadpool_params_from_cpu_params(params.cpuparams);
                struct ggml_threadpool_params tpp_dft_batch = ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);

                if (params.speculative.use_steward) {
                    tpp_tgt.paused = true;
                    tpp_tgt_batch.paused = true;
                    tpp_dft.paused = true;
                    tpp_dft_batch.paused = true;
                }

                struct ggml_threadpool * threadpool_tgt = ggml_threadpool_new_fn(&tpp_tgt);
                struct ggml_threadpool * threadpool_tgt_batch = ggml_threadpool_params_match(&tpp_tgt, &tpp_tgt_batch) ? nullptr : ggml_threadpool_new_fn(&tpp_tgt_batch);
                struct ggml_threadpool * threadpool_dft = ggml_threadpool_new_fn(&tpp_dft);
                struct ggml_threadpool * threadpool_dft_batch = ggml_threadpool_params_match(&tpp_dft, &tpp_dft_batch) ? nullptr : ggml_threadpool_new_fn(&tpp_dft_batch);

                if (threadpool_tgt) {
                    llama_attach_threadpool(ctx_tgt, threadpool_tgt, threadpool_tgt_batch);
                }
                if (threadpool_dft) {
                    llama_attach_threadpool(ctx_dft, threadpool_dft, threadpool_dft_batch);
                }
                // threadpools are owned by the context and freed on llama_free

                threadpools.target       = threadpool_tgt;
                threadpools.target_batch = threadpool_tgt_batch;
                threadpools.draft        = threadpool_dft;
                threadpools.draft_batch  = threadpool_dft_batch;
                threadpools.pause_fn     = ggml_threadpool_pause_fn;
                threadpools.use_phase_pause = params.speculative.use_steward;
            }
        }
    }

    if (params.speculative.use_steward) {
        if (threadpools.target == nullptr || threadpools.draft == nullptr || threadpools.pause_fn == nullptr) {
            LOG_ERR("%s: --spec-steward requires CPU threadpool attach and pause support for both target and draft contexts\n", __func__);
            return 1;
        }
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // TTFT = time to first token: from start of generation until first token is available.
    // Start here (before target prefill); end after first sample_and_accept_n. Includes:
    // target prefill, first draft run (draft prefill + draft decodes), and first verification.
    const auto t_ttft_start = ggml_time_us();
    speculative_loop loop(ctx_tgt, ctx_dft, vocab, params, threadpools, t_ttft_start);

    if (params.speculative.use_steward) {
        sequential_steward steward;

        steward.submit([&loop, &inp]() {
            loop.prefill(inp);
        }).get();

        while (true) {
            const bool done = steward.submit([&loop]() {
                return loop.step();
            }).get();

            if (done) {
                break;
            }
        }
    } else {
        loop.prefill(inp);

        while (!loop.step()) {
        }
    }

    loop.print_summary();

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
