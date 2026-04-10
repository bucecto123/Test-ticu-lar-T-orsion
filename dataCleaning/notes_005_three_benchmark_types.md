three benchmark files exist in this codebase, each answering a distinct research question.

**benchmark_vectorized.py** — answers "does the math work?" no llm, no graph, no row loops. loads the entire dataset as a matrix, runs svd and monoid as numpy/pandas operations in one shot. computes metric 1 (spectral snr — how geometrically separated are attacks from benign?) and metric 3 (thermodynamic efficiency — how many flows does the router drop before they hit the llm?). this is the core quantitative evidence for the paper. runs in ~30 seconds.

**benchmark_stream.py** — answers "does the live pipeline produce the same results?" same math as above but processed row-by-row exactly as the real ids does it, going through the actual sensor functions. no llm still. the point is to verify the streaming inference path is consistent with the vectorized results — if the numbers diverge, there's a bug in the live code. also gives a real throughput number (flows/sec) which is a legitimate system performance metric.

**benchmark_vgs.py** — answers "is the llm actually grounded or is it hallucinating?" this one does call groq. samples 10 flows per attack class, runs them through the physicist, then programmatically checks whether the llm's output was derived strictly from the inputs given to it. computes metric 2 (vgs). this is the evidence that the neuro-symbolic constraint is working — that the llm is a translator, not a guesser.

in short: vectorized = math proof, stream = system proof, vgs = llm proof.
