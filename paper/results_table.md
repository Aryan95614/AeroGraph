# Results Table (LaTeX-Ready)

## Main Results

```latex
\begin{table}[h]
\centering
\caption{Evaluation results: GraphRAG vs.\ vector-only baseline on 50-query benchmark.
Scores are averages across all queries of each type. Faithfulness and relevance
are scored by Claude as judge (0--1 scale). Causal chain accuracy is scored only
for multi-hop queries.}
\label{tab:main-results}
\begin{tabular}{lcccccc}
\toprule
\textbf{System} & \textbf{Faith.} & \textbf{Rel.} & \textbf{Ctx. Prec.} & \textbf{Ctx. Rec.} & \textbf{Causal Acc.} & \textbf{Latency (ms)} \\
\midrule
GraphRAG    & --   & --   & --   & --   & --   & --   \\
Baseline    & --   & --   & --   & --   & --   & --   \\
\bottomrule
\end{tabular}
\end{table}
```

**Note:** Values marked `--` are placeholders to be filled after running
`make eval`. The evaluation pipeline writes actual numbers to
`paper/results/eval_results.json`.

## Per Query Type Breakdown

```latex
\begin{table}[h]
\centering
\caption{Faithfulness and answer relevance by query type.}
\label{tab:query-type}
\begin{tabular}{llcccc}
\toprule
\textbf{Query Type} & \textbf{System} & \textbf{Faith.} & \textbf{Rel.} & \textbf{n} \\
\midrule
\multirow{2}{*}{Single-Hop}  & GraphRAG & -- & -- & 20 \\
                              & Baseline & -- & -- & 20 \\
\midrule
\multirow{2}{*}{Multi-Hop}   & GraphRAG & -- & -- & 20 \\
                              & Baseline & -- & -- & 20 \\
\midrule
\multirow{2}{*}{Comparative} & GraphRAG & -- & -- & 10 \\
                              & Baseline & -- & -- & 10 \\
\bottomrule
\end{tabular}
\end{table}
```

## Graph Statistics

| Metric | Value |
|--------|-------|
| Total nodes | -- |
| Total edges | -- |
| Entity types | -- |
| Reports indexed | -- |
| Chunks indexed | -- |

Values populated by `make paper` from runtime graph statistics.
