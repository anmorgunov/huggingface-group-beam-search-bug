# `transformers` bfloat16 Group Beam Search Bug

a minimal reproduction repository for a `dtype` mismatch bug in `huggingface/transformers`.

a bug introduced in `transformers==4.45.0` (specifically in [PR 31292](https://github.com/huggingface/transformers/pull/31292)) causes a `RuntimeError` during group beam search when using a model in `bfloat16` or `float16`.


## Demo

This will work:
```bash
    uv run --with transformers==4.44.2 --with torch --with accelerate bfloat-gen.py
```

These will fail:
```bash
    uv run --with transformers==4.45.0 --with torch --with accelerate bfloat-gen.py
```

```bash
    uv run --with transformers==4.56.2 --with torch --with accelerate bfloat-gen.py
```

## The Root Cause

The underlying issue is a `dtype` mismatch during an in-place tensor assignment.

1.  the generation code starting from v4.45.0 upcasts model logits from `bfloat16` to `float32` before a `log_softmax` operation.
2.  this `float32` `dtype` propagates through the score calculation.
3.  the code then attempts to assign these `float32` scores back into a `bfloat16` tensor (`processed_score`) without downcasting first.
4.  this triggers the `RuntimeError: Index put requires the source and destination types match...` because pytorch's `index_put_` kernel does not perform implicit casting.

### Evolution of code

the bug was unintentionally introduced in v4.45.0 and can be traced through the following changes:

- v4.44.2 (and earlier): No upcasting.

the logits were used in their native dtype.
```py
# file: src/transformers/generation/utils.py
# line 3742
next_token_logits = outputs.logits[batch_group_indices, -1, :]
```

- v4.45.0: Upcasting is introduced.

.float() is added to the logit calculation for precision, creating the float32 tensor that eventually causes the conflict.
```py
# file: src/transformers/generation/utils.py
# line 3762
# .float() is needed to retain precision for later logits manipulations
next_token_logits = outputs.logits[:, -1, :].clone().float()
```

- v4.51.0: Code is refactored.
the logic is cleaned up from .float() to a more explicit .to() call, but the intentional upcasting to float32 remains.
```py
(3673) next_token_logits = outputs.logits[batch_group_indices, -1, :].to(
                    dtype=torch.float32, device=input_ids.device
                )
```


## The proposed fix

the fix is a one-line, surgical change in the `_group_beam_search` method of `generation/utils.py`. it involves casting the processed scores back to the destination tensor's `dtype` before the assignment.

**from (bugged):**

```py
# line 3689
processed_score[batch_group_indices] = next_token_scores_processed
```

**to (fixed):**
```py
# line 3689
processed_score[batch_group_indices] = next_token_scores_processed.to(processed_score.dtype)
```

this respects the need for fp32 stability in the calculation while ensuring `dtype` correctness for the assignment.
