#pragma once

#include "llama.h"
#include "common.h"

#include <string>

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;  // max drafted tokens
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft
    bool early_stop = true; // stop drafting when top-token prob < p_min (false = always draft up to n_draft)
};

struct common_speculative_draft_request {
    bool vocab_dft_compatible = true;
    llama_tokens prompt_tokens;
    llama_token id_last = LLAMA_TOKEN_NULL;
    std::string prompt_text;
    std::string id_last_text;
};

struct common_speculative_draft_result {
    bool vocab_dft_compatible = true;
    llama_tokens tokens;
    std::string text;
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft,
        bool draft_deterministic = true  // true = greedy (top_k=1), false = top_k=10
);

void common_speculative_free(struct common_speculative * spec);

bool common_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest);

struct common_speculative_draft_request common_speculative_prepare_draft_request(
               struct common_speculative * spec,
                      const llama_tokens & prompt,
                             llama_token   id_last);

struct common_speculative_draft_result common_speculative_gen_draft_result(
               struct common_speculative * spec,
        struct common_speculative_params   params,
        const struct common_speculative_draft_request & request);

llama_tokens common_speculative_finalize_draft_result(
               struct common_speculative * spec,
        const struct common_speculative_draft_result & result,
                                         int           n_draft);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_gen_draft(
               struct common_speculative * spec,
        struct common_speculative_params   params,
                      const llama_tokens & prompt,
                             llama_token   id_last);
