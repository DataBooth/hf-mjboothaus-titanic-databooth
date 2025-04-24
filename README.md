# `hf-mjboothaus-titanic-databooth`

Titanic data quality dataset for Huggingface.co

# Dataset Card: Titanic Passenger List with Data Quality Annotations

## Dataset Description

**Purpose**: Demonstrate how data quality impacts analytics through the iconic Titanic dataset, featuring:
- **Original datasets** (with known age/class errors)
- **Corrected versions** (with reconciled passenger details)
- **Data quality annotations** (error flags, reconciliation sources)

**Homepage**: [Data Governance: Titanic Dataset and the Perils of Bad Data](https://www.databooth.com.au/posts/data-quality-titanic/)
**Repository**: `mjboothaus-titanic-databooth`  
**Tasks**: `data-cleaning`, `error-detection`, `survival-prediction`

## Dataset Versions
| Version | Description | Key Features |
|---------|-------------|--------------|
| `original` | Unmodified datasets | Contains age discrepancies (e.g., Algernon Barkworth recorded as 80) |
| `corrected-v1` | Age-reconciled data | Matches Encyclopedia Titanica records |
| `annotated` | Error-flagged version | Contains `is_age_discrepancy` and `data_source` columns |

## Data Fields (Corrected Version)
| Column | Type | Description | Common Errors |
|--------|------|-------------|---------------|
| `name` | string | Passenger name | - |
| `age` | float | **Corrected age** at voyage | Original had 143+ age errors >2 years |
| `pclass` | int | Passenger class (1-3) | Class misassignments in original |
| `survived` | int | Survival status | - |
| `is_age_discrepancy` | bool | True if original age error >2 years | - |
| `data_source` | string | Reconciliation source (ET) | - |

## Usage Example
```
from datasets import load_dataset

# Compare original vs corrected data
original = load_dataset("mjboothaus/titanic-databooth", name="original")
corrected = load_dataset("mjboothaus/titanic-databooth", name="corrected-v1")

# Find corrected records
discrepancies = corrected.filter(lambda x: x["is_age_discrepancy"])
print(f"Fixed {len(discrepancies)} age errors")
```

<!-- TODO: Also st.connnector class? And "plain" class too for DuckDB? -->

## Key Data Quality Issues
1. **Age Discrepancies**  
   - Original error: 80yo survivor (actual age 47)
   - 143+ passengers with >2 year age differences
   - Systemic bias from death age vs voyage age confusion

2. **Class Misassignments**  
   - Documented cabin class errors
   - Impacts fare/survival correlation analysis

## Reconciliation Process
1. **Source Alignment**: Cross-referenced with:
   - Encyclopedia Titanica
   - Titanic Facts Network
   - Historical voyage manifests

2. **Validation Methods**:
   - Age distribution analysis
   - Survival rate by age cohort
   - Source conflict resolution protocols

## Impact Analysis
| Metric | Original Data | Corrected Data |
|--------|---------------|----------------|
| Avg Age (Survivors) | 28.34 | 27.46 |
| Oldest Survivor | 80 (incorrect) | 64 (Mary Compton) |
| Class 1 Survival Rate | 62.96% | 63.01% (adjusted) |

## Suggested Use Cases
- **Data Quality Workshops**: Compare original/corrected versions
- **Governance Training**: Demonstrate error propagation
- **ML Robustness Tests**: Train models on both versions

## License
CC-BY-4.0 [TODO: CHECK]

## Citation
```
@dataset{titanic-databooth,
  author = {Michael J. Booth},
  title = {Titanic Data Quality Benchmark},
  year = {2025},
  publisher = {Hugging Face},
  version = {1.0.0}
}
```

## Acknowledgements

- **Encyclopedia Titanica** for reference data

**Key Features to Highlight**:
- **Version Control**: Clear lineage between original/corrected data
- **Error Documentation**: Specific examples with historical context
- **Impact Metrics**: Quantifiable differences between datasets
- **Educational Focus**: Designed for data governance training

<!-- **TODO**: Include a Jupyter / Marimo / Streamlit notebook  -->

**Code demonstrating**:
1. Age distribution comparisons
2. Survival rate analysis by data version
3. Simple ML model performance differences

## References:

- [1] https://www.databooth.com.au/posts/data-quality-titanic/
- [2] https://mjboothaus.wordpress.com/2017/07/11/did-a-male-octogenarian-really-survive-the-sinking-of-the-rms-titanic-2/

