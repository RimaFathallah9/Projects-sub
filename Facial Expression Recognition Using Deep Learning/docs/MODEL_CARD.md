# Model card: FER-2013 expression classifiers (template)

> Fill every `TODO` from files in `outputs/` and `reports/`. Do not enter numbers by hand from memory.

## Model details
- Architectures: Custom CNN (from scratch), ResNet18 (ImageNet-pretrained, fine-tuned), MobileNetV2 (ImageNet-pretrained, fine-tuned). See `src/fer/models/`.
- Framework: PyTorch. Training configs: `configs/`. Git commit / environment: TODO (`summary.json -> environment`).
- Output: probability distribution over 7 categories: angry, disgust, fear, happy, sad, surprise, neutral.

## Intended use
Research and education: studying CNN architectures, transfer learning and explainability on a public benchmark.

## Out-of-scope / prohibited use
- Inferring a person's emotional state, intent, honesty, mental health, or suitability (hiring, education, policing, insurance, border control, surveillance).
- Any automated decision about an individual.
Facial movements are not reliable indicators of internal emotion and vary by culture, context and individual (see Barrett et al., 2019).

## Data
FER-2013: 48x48 grayscale faces, 7 labels. This checkout uses `data/train` (28,709) and `data/test` (7,178 = PublicTest + PrivateTest). Validation is 10% of train. Splits and leakage counts: `reports/eda.json`, `outputs/*/split_report.json`.

## Metrics (test split, mean ± std over seeds)
TODO: paste `reports/tables/comparison_test.md`. Also: per-class table, confusion matrices, ECE, parameters, MACs, latency.

## Explainability
TODO: summarise `gradcam_test.json` (randomisation sanity check, border-mass) and gallery figures.

## Known limitations
- Label noise and class imbalance (Disgust is rare): TODO quantify.
- Dataset lacks demographic annotations; subgroup performance (age, skin tone, gender presentation, pose, occlusion, lighting) is **unmeasured**, so no fairness claim can be made.
- Acted/web-posed expressions differ from spontaneous ones; domain shift expected on webcam/phone images.
- 48x48 grayscale input discards detail.

## Privacy
The demo processes uploads in memory only, requires explicit consent, and stores nothing.
