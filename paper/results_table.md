# Results Table (LaTeX-Ready)

## Main Results — 4-System Ablation

```latex
\begin{table}[h]
\centering
\caption{Ablation study: four retrieval systems on 50-query benchmark.
Scores are means $\pm$ std across all queries. Faithfulness and relevance
scored by Claude-as-judge (0--1). Causal chain accuracy scored only for
multi-hop queries. Reference similarity uses ROUGE-L F1.}
\label{tab:main-results}
\begin{tabular}{lccccccc}
\toprule
\textbf{System} & \textbf{Faith.} & \textbf{Rel.} & \textbf{Ctx P.} & \textbf{Ctx R.} & \textbf{Causal} & \textbf{Ref. Sim.} & \textbf{Lat. (s)} \\
\midrule
GraphRAG         & $0.39 \pm 0.34$ & $0.85 \pm 0.17$ & -- & 1.00 & 0.55 & -- & 19.8 \\
Vector-Only      & $0.31 \pm 0.30$ & $0.85 \pm 0.18$ & -- & 1.00 & 0.55 & -- & 18.5 \\
BM25             & $0.23 \pm 0.24$ & $0.78 \pm 0.23$ & -- & 1.00 & 0.56 & -- & 17.0 \\
Graph-Only       & $0.53 \pm 0.43$ & $0.60 \pm 0.29$ & -- & 1.00 & 0.54 & -- & 14.0 \\
\bottomrule
\end{tabular}
\end{table}
```

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
\multirow{4}{*}{Single-Hop}    & GraphRAG         & 0.37 & 0.82 & 20 \\
                               & Vector-Only      & 0.30 & 0.86 & 20 \\
                               & BM25             & 0.21 & 0.73 & 20 \\
                               & Graph-Only       & 0.37 & 0.69 & 20 \\
\midrule
\multirow{4}{*}{Multi-Hop}     & GraphRAG         & 0.30 & 0.91 & 20 \\
                               & Vector-Only      & 0.30 & 0.90 & 20 \\
                               & BM25             & 0.21 & 0.82 & 20 \\
                               & Graph-Only       & 0.71 & 0.52 & 20 \\
\midrule
\multirow{4}{*}{Comparative}   & GraphRAG         & 0.61 & 0.81 & 10 \\
                               & Vector-Only      & 0.34 & 0.72 & 10 \\
                               & BM25             & 0.31 & 0.76 & 10 \\
                               & Graph-Only       & 0.48 & 0.59 & 10 \\
\bottomrule
\end{tabular}
\end{table}
```

## Graph Statistics

| Metric | Value |
|--------|-------|
| Total nodes | 29244 |
| Total edges | 43505 |
| Entity types | 10 |
| Edge types | 8 |
| Reports indexed | 2460 |
| Chunks indexed | 4710 |
| Mean entities/report | 21.5 |
| Mean relations/report | 19.6 |

*Auto-populated by `scripts/populate_paper.py` from evaluation results.*
