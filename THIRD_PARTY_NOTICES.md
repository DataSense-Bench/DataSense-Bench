# Third-party notices

Original DataSense-Bench code is licensed under Apache-2.0.
Copyright 2026 DataSense-Bench authors.

## EnvScaler — MIT

Source: https://github.com/RUC-NLPIR/EnvScaler/tree/96ae8b02dc0187c911b8e2101e7bb6904271597b

The BFCL recipe and evaluator adaptations follow EnvScaler. The included code
has been adapted for the task-specific release layout and standalone execution.

MIT License

Copyright (c) 2026 RUC-NLPIR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Berkeley Function Calling Leaderboard / Gorilla — Apache-2.0

Source: https://github.com/ShishirPatil/gorilla

Files under `code/bfcl/post-training/patches/bfcl_eval/` adapt the BFCL evaluator
for the experiment's Qwen3 handling, local inference and fixed V3 data.
The Apache-2.0 license text is included in `LICENSE`.

## Qwen3 — Apache-2.0

Source: https://huggingface.co/Qwen/Qwen3-4B

`code/tblite/post-training/chat_template_allthink.jinja` adapts the Qwen3 chat
format for reasoning in every assistant turn. Tokenizer preparation uses the
user-supplied Qwen3 tokenizer. The Apache-2.0 license text is in `LICENSE`.

Datasets and model weights are downloaded from their respective providers and
remain subject to the licenses listed there.
