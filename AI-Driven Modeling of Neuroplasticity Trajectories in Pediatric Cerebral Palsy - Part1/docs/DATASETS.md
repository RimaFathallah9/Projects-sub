# Dataset access tracker

Checked 2026-11-24 via web search. **Evidence level: search-result snippets, not primary
pages.** Every "Reported" item still needs confirmation on the linked official page before
it goes in a manuscript. "Unknown" means not found; it is not an assumption of absence.

## 1. CP-NET / Brain-CODE (Ontario Brain Institute) - best candidate
| Item | Status |
|---|---|
| Access | **Reported:** controlled data releases. Investigators submit requests through Brain-CODE Data Release Portals; reviewed by the Brain-CODE Data Access Committee and Steering Committee. Whether an independent/unaffiliated researcher qualifies: **Unknown** - must ask Brain-CODE. |
| CP-NET clinical database | **Reported:** children born 2009 or later with confirmed CP; risk factors, neuroimaging information, neurodevelopmental functioning. Size, visit schedule: **Unknown**. |
| Hemi-NET Clinical Database | **Reported:** 320 children/youth aged 2-18 with hemiplegic CP (Ontario). Scans listed: T1w, T2w, FLAIR, DTI, Angio, CT, PD/T2, MRS; formats NIfTI and CSV (so raw imaging appears to be included). Longitudinal structure: **Unknown**. |
| WGS release | **Reported:** controlled; 221 children with CP plus biological parents (656 samples), CRAM format. |
| Missingness, motor/cognitive instruments | **Unknown** - read the data dictionary after access. |
| Suitability | Potentially high (CP + MRI + genomics) but only if longitudinal visits exist and access is granted. Hemi-NET is hemiplegic only (not all CP). |

Sources: [Brain-CODE](https://braincode.ca) - [Controlled releases](https://www.braincode.ca/node/1860) - [OBI dataset news 2022](https://braininstitute.ca/news-events/2022/new-dataset-to-enhance-care-for-children-with-cerebral-palsy) - [OBI WGS news 2024](https://braininstitute.ca/news-events/2024/whole-genome-sequencing-data-from-canadian-led-cp-study-now-available) - [CONP entry](https://n2t.net/ark:/70798/d7wx86hhx5mj34339s)

## 2. CPOP (resolved: a national CP follow-up programme, not an imaging cohort)
**Reported:** CPOP is the Danish Cerebral Palsy Follow-up Programme (all 18 Danish paediatric departments; Atlas of Longitudinal Datasets lists >1,600 participants aged 0-15) and a Norwegian CP follow-up programme (used in a register-based cohort of 419 ambulant children with spastic bilateral CP). Sources disagree on country because both use the acronym; confirm which one you mean.
- Likely content: clinical follow-up (motor classification, orthopaedic measures), longitudinal. **MRI: not documented in what I found; assume none until shown.**
- Access: **Unknown** (national registry; usually application + ethics + legal basis; for EU/Nordic data, GDPR applies).
- Suitability: good for clinical/motor trajectories (E1, E2 clinical-only), not for multimodal imaging fusion.
Sources: [Atlas of Longitudinal Datasets - CPOP](https://atlaslongitudinaldatasets.ac.uk/datasets/cpop) - [Danish validation study 2025](https://journals.sagepub.com/doi/full/10.1177/14034948231219825) - [Norwegian cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC6947838/) - [Sweden CPUP longitudinal study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10741609/)
Related registries seen (not examined): Swiss CP Registry, Swedish CPUP, Canadian CP Registry, CP Research Network (US).

## 3. Other CP cohorts found (candidates, access not established)
| Cohort | Reported facts | Access |
|---|---|---|
| Australian Cerebral Palsy Child Study | Protocol: prospective, planned n=240, assessments at 18, 24, 30, 36, 48, 60 months corrected age, structural MRI. Planned n is not achieved n. | Unknown - [protocol](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3686628/) |
| NCT01903603 | Observational; MRI/fMRI/DTI and motor function at baseline, 6, 12 months; CP + healthy controls | Unknown - [ClinicalTrials.gov](https://clinicaltrials.gov/study/NCT01903603) |
| SCPE registry MRI study | 3,818 children, cross-sectional MRI classification; not longitudinal | Unknown - [PubMed 32120429](https://pubmed.ncbi.nlm.nih.gov/32120429/) |
These are the usual route: contact the corresponding author / data custodian.

## 4. OpenNeuro and open repositories
- **Reported:** no CP-specific dataset found. One perinatal-stroke dataset, [ds004498](https://doi.org/10.18112/openneuro.ds004498.v1.0.0): listings conflict (1 subject / 18 sessions vs 7 DWI) - not opened; at most a single-subject case dataset, **unsuitable for modelling**.
- ds004889 (SOOP) is adult stroke - not relevant.
- Zenodo/figshare: no hit, but not searched separately.
- NDA hosts PING (used in a unilateral-CP atlas study as a normative reference) - controlled access.
- Conclusion: **no open longitudinal pediatric CP imaging dataset identified.** Open data cannot carry this project.
- Still to do by hand: search openneuro.org directly (site search is not indexed by the web search tool).

## 5. ABCD (normative comparison only)
**Reported:** ~11,880 children aged 9-10 at baseline (counts vary by release: 11,874-11,880), 22 sites, annual in-person follow-up for at least 10 years; access via NIMH Data Archive (NDA) with data-use agreement, plus the DEAP online analysis portal. Not a CP cohort; age window (9+) misses early CP development. Use only for normative reference curves; harmonization (scanner, site) is a real problem.
Sources: [NDA ABCD](https://nda.nih.gov/abcd) - [methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8803401)

## Decision rules
1. Need a CP cohort with >= ~100 children and >= 3 visits; otherwise downscale (see EXPERIMENTS.md fallback).
2. Controlled-access data never enter the repo or any external service.
3. Synthetic data are for unit tests only and never reported as results.

## Next actions
- Email Brain-CODE: eligibility for independent researchers, whether Hemi-NET/CP-NET are longitudinal, data dictionary.
- Resolve what CPOP is.
- Search OpenNeuro/NDA/Zenodo directly.
- Read primary pages and upgrade "Reported" to "Confirmed".
