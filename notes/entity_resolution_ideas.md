# Entity Resolution Approaches

## Problem
Same real-world entity appears under different names:
- "B737" vs "Boeing 737" vs "737-800"
- "LAX" vs "Los Angeles International"
- "windshear" vs "wind shear" vs "microburst"

## Approaches to try
1. Exact string matching after normalization (lowercase, strip whitespace)
2. Fuzzy matching with edit distance threshold
3. Embedding similarity with sentence-transformers
4. Taxonomy-based canonical mapping (ICAO codes, HFACS categories)

## Priority
Start with taxonomy-based for structured entities (airports, aircraft),
fall back to fuzzy for free-text entities (weather, human factors).
