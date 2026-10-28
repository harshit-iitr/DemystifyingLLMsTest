Bug review and fixes
====================

The repository README describes 40 score checks (35 component checks and 5
integration checks). I found and corrected the following concrete defects in
the checked-in implementation. The tests and GitHub workflow were left intact.

Model configuration and architecture
------------------------------------
1. Default d_model was 769, which is not divisible by 12. Changed it to 768.
2. Default vocabulary size was 50256 instead of 50257. Corrected the default.
3. EOS_TOKEN_ID was outside the vocabulary and disagreed with GPT-2. Set it to 50256.
4. UNK_TOKEN_ID incorrectly pointed at the end-of-text token. Set it to 1.
5. The causal mask blocked past tokens and allowed future tokens. Reversed the mask.
6. Attention scores were not divided by sqrt(head_dim). Added the required scaling.
7. Attention softmax normalized over query positions. It now normalizes over keys.
8. Query, key, and value projections shared one Linear layer. They are independent.
9. Attention heads were concatenated over the sequence axis. They now join on features.
10. Sinusoidal position vectors used cosine for both even and odd dimensions. Fixed sine.
11. Odd-width sinusoidal vectors had an incompatible cosine assignment. Added a size slice.
12. Learned position embeddings did not guard invalid offsets/context lengths. Added checks.
13. Embedding vectors multiplied token and position vectors. They now add them.
14. Token IDs were wrapped modulo vocabulary size, silently changing invalid IDs. Removed it.
15. LMHead projected vocab_size to d_model, reversing its input/output dimensions. Fixed it.
16. LayerNorm normalized across batch/time instead of the final feature axis. Fixed the axis.
17. ResidualConnection returned x + x and discarded the sublayer. Restored dropout(sublayer).
18. FeedForward ignored the configured expansion ratio. It now uses configured d_ffn.
19. TransformerBlock applied FFN before attention. Restored pre-norm attention then FFN order.
20. Rotary embedding accepted odd dimensions and lacked offset handling. Added validation/slicing.
21. FullTransformerModel omitted the explicit config for its block stack. It now passes it.
22. FullTransformerModel did not report its layer count or tie configured weights. Fixed both.

Tokenizer and text generation
-----------------------------
23. Encoding ignored add_bos. It now prepends BOS when requested.
24. Decoding discarded tokens based only on sequence length. It strips actual boundary IDs.
25. Decoding stripped boundary tokens even when skip_special_tokens=False. Fixed the option.
26. Vocabulary pad_token_id was -1 despite PAD being stored at ID 0. Corrected the field.
27. Vocabulary instances shared mutable class-level maps and polluted one another. Removed them.
28. Vocabulary could assign two tokens to one ID. Duplicate-ID assignments now fail clearly.
29. BPE sorted merges lexically rather than respecting their declared rank. Uses merge rank.
30. BPE merged only the first matching pair occurrence in a word. Merges all occurrences.
31. Top-k sampling retained the lowest logits. It now retains the highest logits.
32. Top-p sampling used vocabulary order rather than probability order. It sorts then scatters.
33. Temperature scaled probabilities and divided by zero. It now scales logits and validates.
34. Sampler's zero-temperature path sampled stochastically. It now selects argmax.
35. Repetition penalty subtracted a constant. It now uses sign-aware multiplicative penalty.
36. Autoregressive generation prepended each new token, reversing output order. Appends instead.
37. StopCriteria stopped only after exceeding max length. It stops at the configured limit.
38. KVCache overwrote prior entries instead of appending sequence positions. It concatenates.
39. The inference model entered training mode for generation. It now uses eval mode/device.
40. Inference generation ignored top_k/top_p/repetition settings. It now uses Sampler.
41. Inference mishandled tuple model outputs and empty prompts. Both cases are handled.
42. Inference accepted negative temperature and ignored the configured seed. Added checks/seed.
43. InferenceEngine returned fabricated continuation text after failures. Removed fabricated text.

Serving, utilities, and evaluation
----------------------------------
44. to_device always moved tensors to CPU. It now honors the requested device.
45. ServingPipeline passed a Python list to the model without batch/device conversion. Fixed it.
46. ServingPipeline ignored max_new_tokens and did not autoregressively generate. Added a loop.
47. Completion responses returned token lists as text. They now return decoded completion text.
48. The generate-only API path discarded its result and substituted made-up token IDs. Fixed it.
49. The server used HTTPServer despite claiming threaded request handling. Uses ThreadingHTTPServer.
50. Cross-entropy used view on potentially non-contiguous tensors. It now uses reshape.
51. Forward benchmarking accepted zero measured runs and then failed on empty statistics. Validates.
52. Batch chunking accepted zero/negative chunk sizes. It now validates the size.
53. DynamicBatcher accepted unusable batch sizes and negative wait durations. It validates them.
54. ModelConfig accepted nonpositive dimensions and invalid dropout/context values. Added checks.
55. InferenceConfig did not validate beam_width or seed types. Added checks.

Validation status
-----------------
The environment could not resolve PyPI, so `pip install -r requirements.txt` could not
download torch. I did not run the test suite. Source changes were reviewed with
`git diff --check`; a Python syntax compilation is still needed in an environment
with `python3` available.

Second pass after the autograder screenshot
-------------------------------------------
The screenshot showed 33/40 checks passing. These additional issues were corrected
while targeting the remaining sampling, decoding, and integration edge cases:

56. InferenceConfig defaulted to zero generated tokens and no stop token. Set a useful
   generation default and use EOS as the default stop condition.
57. Temperature was applied after nucleus filtering, producing the wrong top-p set.
   Temperature now adjusts logits before top-k/top-p filtering.
58. Zero-temperature sampling bypassed repetition penalties. It now applies the penalty
   before greedy selection.
59. Sampler accepted invalid top-k, top-p, temperature, and penalty values. It validates them.
60. Batched repetition penalties mixed token histories across batch rows. Histories are
   now applied per row, and out-of-range token IDs are ignored safely.
61. Beam search indexed beyond the vocabulary when beam_width exceeded vocab size. It
   now limits expansion to available candidates and validates its input shape/settings.
62. Beam search could return no candidate for a zero-step run. It now returns the input.
63. InferenceEngine ignored configured beam_width and returned the prompt for negative
   max_new_tokens. It now dispatches beam search and rejects negative limits.
64. Generation loops could build autograd graphs during inference. Autoregressive
   generation now runs without gradient tracking.
65. pad_sequence accepted invalid padding sides when no padding was necessary. It validates
   the side before inspecting individual sequences.
66. repetition_rate accepted zero or negative n-gram sizes. It now rejects them.
67. The standalone autoregressive loop accepted malformed/empty inputs and negative token
   limits. It now validates both conditions.
68. StopCriteria accepted negative maximum lengths. It now validates the bound.
69. KVCache accepted invalid layer indexes and mismatched key/value shapes. It validates both.
70. Sinusoidal position embeddings accepted invalid dimensions/offsets. Added bounds checks.
71. LMHead could tie an embedding matrix with an incompatible shape. It now rejects mismatch.
72. TokenEmbedding's custom initialization overwrote nn.Embedding's zero padding row. The
   configured padding vector is now zeroed after initialization.
73. ModelLoader's default `.pth` path disagreed with the project's configured `.pt` path.
   The loader now uses DEFAULT_WEIGHTS_PATH.
74. LayerNorm treated a multi-axis normalized_shape as only its final dimension. It now
   supports the full normalized shape using PyTorch's layer_norm operation.
75. FullTransformerModel passed kv_cache through **kwargs, causing generic models that
   accept **kwargs to be misdetected as cache-aware. Cache detection now requires an
   explicit kv_cache parameter.
76. ServingPipeline returned the prompt together with its generated continuation despite
   documenting a completion return value. It now decodes only generated tokens.
77. ServingPipeline ignored temperature, top-k, top-p, repetition penalty, and seed. It
   now samples through the configured Sampler.
78. API request handling assumed every generation method returned prompt-prefixed IDs.
   It now supports either prompt-prefixed or continuation-only token lists.
79. API requests passed unset top-k/top-p arguments to engines that do not accept them.
   Those optional arguments are now forwarded only when present in the request.
80. Completion routes did not normalize method case or map invalid request bodies into
   client errors. Both cases are handled.
81. CORS response headers ignored configured allowed origins. The server now applies its
   ServerConfig origin list to normal and preflight responses.
82. Beam search expanded EOS hypotheses again and ranked by average score across prompt
   plus output length. It now completes EOS candidates immediately and compares cumulative
   log-probability scores.
83. Rotary mode added sinusoidal absolute embeddings as well as rotary positions. The
   embedding layer now omits absolute positions when use_rotary is enabled.
84. Batched autoregressive generation never stopped rows that emitted EOS. It now tracks
   completion per row and pads finished rows with EOS until the batch is complete.
85. MultiHeadAttention treated a supplied padding mask as a replacement for causal
   masking. It now combines key-padding and causal masks, including cached query offsets.
86. FullTransformerModel did not accept or propagate attention masks. It now forwards them
   through the block stack into attention.
87. RequestHandler assumed continuation-only and prompt-prefixed engine outputs could not
   both be valid; completion accounting was branch-dependent. It now handles both forms.
88. Completion generation did not forward optional top-k/top-p unless specified, and routes
   accepted malformed JSON object shapes inconsistently. These paths now validate/forward.
89. InferenceConfig.early_stopping was ignored. Greedy and beam generation now honor it.
90. Top-k threshold masking retained more than k tokens when logits tied. It now scatters
   exactly the selected k values.
91. Left-padded batches assigned positions based on padded width for real tokens. Absolute
   and rotary embeddings now derive positions from the attention mask.
92. Fully masked padded query rows produced NaNs in softmax, which then leaked through
   subsequent layers. Attention now maps undefined all-masked rows to zero weights.
