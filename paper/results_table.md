# Results Table (LaTeX-Ready)

*Auto-generated from `paper/results/eval_results_claude_judge.json` by `scripts/populate_paper.py`.*

## Table 1 — Retrieval Quality (6 systems, 50 queries)

```latex
\begin{table*}[t]
\centering
\caption{Retrieval quality across six systems on the 50-query benchmark. P@10, R@10, nDCG@10 computed against graph-derived relevance labels. Means $\pm$ std. Best per column in \textbf{bold}.}
\label{tab:retrieval}
\begin{tabular}{lcccc}
\toprule
\textbf{System} & \textbf{P@10} & \textbf{R@10} & \textbf{nDCG@10} & \textbf{Lat.\ (s)} \\
\midrule
HippoRAG-4way  & $\textbf{0.210} \pm 0.301$ & $\textbf{0.044} \pm 0.148$ & $\textbf{0.231} \pm 0.361$ & 23.1 \\
GraphRAG       & $0.152 \pm 0.231$ & $0.016 \pm 0.036$ & $0.151 \pm 0.237$ & 19.2 \\
Vector-Only    & $0.114 \pm 0.190$ & $0.014 \pm 0.036$ & $0.121 \pm 0.209$ & 1.6 \\
BM25           & $0.106 \pm 0.166$ & $0.028 \pm 0.141$ & $0.116 \pm 0.192$ & 0.7 \\
PPR-Only       & $0.200 \pm 0.349$ & $0.035 \pm 0.107$ & $0.221 \pm 0.397$ & 2.6 \\
Graph-Only     & $0.058 \pm 0.169$ & $0.004 \pm 0.010$ & $0.072 \pm 0.191$ & 0.7 \\
\bottomrule
\end{tabular}
\end{table*}
```

## Table 2 — Answer Quality (Dual-Judge Evaluation)

Faithfulness and answer relevance scored by two independent LLM judges:
Ollama (`qwen2.5:7b`, local) and Claude (`claude-sonnet-4`).
Scores are means $\pm$ std on [0, 1]. Causal accuracy is scored only on multi-hop queries.

```latex
\begin{table*}[t]
\centering
\caption{Answer quality with dual-judge evaluation. Ollama and Claude judges are reported independently; agreement between judges is discussed in Section~\ref{sec:judge-agreement}.}
\label{tab:answer-quality}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{Ollama Judge}} & \multicolumn{2}{c}{\textbf{Claude Judge}} & & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{System} & \textbf{Faith.} & \textbf{Rel.} & \textbf{Faith.} & \textbf{Rel.} & \textbf{Causal} & \textbf{Ctx P.} \\
\midrule
HippoRAG-4way  & $0.685 \pm 0.271$ & $0.615 \pm 0.279$ & $0.310 \pm 0.322$ & $0.135 \pm 0.126$ & 0.650 & 0.219 \\
GraphRAG       & $0.642 \pm 0.301$ & $0.489 \pm 0.336$ & $0.255 \pm 0.265$ & $0.120 \pm 0.126$ & 0.537 & 0.158 \\
Vector-Only    & $0.608 \pm 0.337$ & $0.608 \pm 0.278$ & $0.410 \pm 0.356$ & $0.170 \pm 0.118$ & 0.588 & 0.119 \\
BM25           & $0.520 \pm 0.346$ & $0.507 \pm 0.289$ & $0.245 \pm 0.301$ & $0.105 \pm 0.125$ & 0.650 & 0.110 \\
PPR-Only       & $0.578 \pm 0.287$ & $0.517 \pm 0.320$ & $0.230 \pm 0.225$ & $0.055 \pm 0.105$ & 0.438 & 0.208 \\
Graph-Only     & $0.532 \pm 0.343$ & $0.395 \pm 0.232$ & $0.205 \pm 0.218$ & $0.010 \pm 0.049$ & 0.225 & 0.060 \\
\bottomrule
\end{tabular}
\end{table*}
```

## Table 3 — Performance by Query Type (Ollama Judge)

```latex
\begin{table}[t]
\centering
\caption{Ollama-judged faithfulness and relevance stratified by query type. Single-hop (n=20): fact retrieval from one report. Multi-hop (n=20): causal reasoning spanning multiple reports. Comparative (n=10): cross-corpus comparison.}
\label{tab:query-type}
\begin{tabular}{llcccc}
\toprule
\textbf{Query Type} & \textbf{System} & \textbf{Faith.} & \textbf{Rel.} & \textbf{n} \\
\midrule
\multirow{6}{*}{Single-Hop}      & HippoRAG-4way  & 0.634 & 0.556 & 20 \\
                                 & GraphRAG       & 0.581 & 0.447 & 20 \\
                                 & Vector-Only    & 0.575 & 0.569 & 20 \\
                                 & BM25           & 0.588 & 0.531 & 20 \\
                                 & PPR-Only       & 0.613 & 0.487 & 20 \\
                                 & Graph-Only     & 0.500 & 0.425 & 20 \\
\midrule
\multirow{6}{*}{Multi-Hop}       & HippoRAG-4way  & 0.766 & 0.694 & 20 \\
                                 & GraphRAG       & 0.781 & 0.556 & 20 \\
                                 & Vector-Only    & 0.744 & 0.650 & 20 \\
                                 & BM25           & 0.500 & 0.531 & 20 \\
                                 & PPR-Only       & 0.594 & 0.512 & 20 \\
                                 & Graph-Only     & 0.544 & 0.338 & 20 \\
\midrule
\multirow{6}{*}{Comparative}     & HippoRAG-4way  & 0.625 & 0.575 & 10 \\
                                 & GraphRAG       & 0.487 & 0.438 & 10 \\
                                 & Vector-Only    & 0.400 & 0.600 & 10 \\
                                 & BM25           & 0.425 & 0.412 & 10 \\
                                 & PPR-Only       & 0.475 & 0.588 & 10 \\
                                 & Graph-Only     & 0.575 & 0.450 & 10 \\
\bottomrule
\end{tabular}
\end{table}
```

## Table 4 — Pairwise Blind Comparisons

Each pair judged by Claude with blinded system labels. Reports win rate of the left system.

| Comparison | Left Wins | Right Wins | Ties | Total | Win Rate |
|---|---|---|---|---|---|
| hybrid_4way vs. baseline | 31 | 17 | 2 | 50 | 62.00% |
| hybrid_4way vs. bm25 | 32 | 14 | 4 | 50 | 64.00% |
| graphrag vs. baseline | 36 | 9 | 5 | 50 | 72.00% |

## Graph Statistics

| Metric | Value |
|---|---|
| Clean graph — nodes | 23948 |
| Clean graph — edges | 48479 |
| Raw graph — nodes | 29244 |
| Raw graph — edges | 43505 |
| Chunks indexed | 4710 |
| Reports ingested | 3040 |
| Mean entities / report | 21.4 |
| Mean relations / report | 19.5 |
| Communities (L0) | 311 |
| Communities (L1) | 393 |
| Communities (L2) | 296 |
