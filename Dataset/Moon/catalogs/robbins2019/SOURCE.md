# Robbins lunar crater database - source

- **Product**: Robbins Lunar Crater Database, release 2018-08-15, PDS4 bundle `urn:nasa:pds:lunar_crater_database_robbins_2018`, version 1.0.
- **URL**: [Robbins lunar crater database download link](https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/lunar_crater_database_robbins_2018)
- **Download date**: 2026-11-18.
- **Integrity**: MD5 of `lunar_crater_database_robbins_2018.csv` = `13f2f9df2e71d7049e966ee405c7f014`, identical to the bundle's `bundle_checksums.txt`; re-checked by `python Experiments/qcai2027/qc_checks.py catalog` (→ `Experiments/qcai2027/results/checks/catalog.json`). The MD5 of all 17 bundle files was checked once before the removal of the unneeded files (not repeatable: those files are gone).
- **Citation**: Robbins, S. J. (2019). A new global database of lunar impact craters >1-2 km: 1. Crater locations and sizes, comparisons with published databases, and global analysis. *Journal of Geophysical Research: Planets*, 124, 871-892. https://doi.org/10.1029/2018JE005592 (the bundle documents cite it as 2018, vol. 123: the paper was published online in 2018 and in the 2019 volume).

## Files kept
| File | Content |
|---|---|
| `lunar_crater_database_robbins_2018.csv` | the catalogue: 1,296,796 craters, 21 comma-separated columns, one header line |
| `lunar_crater_database_robbins_2018.xml` | PDS4 label of the table (field names and order) |
| `lunar_crater_database_archive_description.pdf` | short description: sources, completeness (~1-2 km), method |

Removed from the original bundle (not needed): PDS4 schema files (`schema/`), collection inventories, bundle label, checksum files, bundle README.

## Columns used by the experiment
| Column | Meaning | Unit / range (checked on the data by `qc_checks.py catalog`) |
|---|---|---|
| `CRATER_ID` | unique crater id | unique for all rows |
| `LAT_CIRC_IMG` | latitude of the centre of the circle fit to the rim | degrees, planetocentric, -89.85 to 89.97 |
| `LON_CIRC_IMG` | longitude of the centre of the circle fit | degrees east, 0-360 |
| `DIAM_CIRC_IMG` | diameter of the circle fit | km, 1.0 to 2,491.9 |

Units are not written in the PDS4 label; they follow from the value ranges (minimum diameter 1.0 = the catalogue limit of 1 km; maximum ~2,500 km = South Pole-Aitken basin) and from the paper.

## Content in the region R of the plan (30°S-30°N, 0-90°E)
Computed by `python Experiments/qcai2027/qc_data.py craters` (→ `Experiments/qcai2027/results/craters_R_summary.json`): 131,182 craters with the centre in R, 128,382 with 1 ≤ D < 8 km (1-2 km 89,516; 2-4 km 31,562; 4-8 km 7,304), 2,800 with D ≥ 8 km, none with D < 1 km.
