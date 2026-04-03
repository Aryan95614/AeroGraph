# Taxonomy Mapping Research

## HFACS (Human Factors Analysis and Classification System)
- Level 1: Unsafe Acts (errors, violations)
- Level 2: Preconditions (physical/mental, CRM, environmental)
- Level 3: Unsafe Supervision
- Level 4: Organizational Influences

Need to map extracted "human_factor" entities to HFACS categories.
Most extractions land in L1/L2.

## ICAO Airport Codes
Use ICAO 4-letter codes as canonical. Map IATA 3-letter to ICAO.
FAA LIDs (3-letter domestic) also need mapping.

## Aircraft Type Designators
ICAO type designators (e.g., B738 for 737-800).
Many reports use informal names - need lookup table.

## Next steps
- Build canonical dictionaries for each taxonomy
- Implement exact match first, then fuzzy fallback
- Measure resolution rate on current graph
