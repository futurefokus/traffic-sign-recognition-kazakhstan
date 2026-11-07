# Traffic Sign Recognition for Kazakhstan

Five-class cropped-sign classifier for Computer Vision Assignment 2.

## Run in Google Colab

For inference without training, open [Recovery Colab](https://colab.research.google.com/github/futurefokus/traffic-sign-recognition-kazakhstan/blob/main/Recovery_Colab.ipynb). Run its two code cells. It downloads the verified release bundle, restores the included architecture and saved weights, and launches Gradio. If a release download is unavailable, upload the saved project ZIP when prompted. A GPU is optional; recovery inference was tested on CPU for all five classes.

For experiments, open `Kazakhstan_Traffic_Sign_Recognition.ipynb` with a T4 runtime. It downloads the source collection, prepares a grouped split, trains the baseline and classifier heads, and evaluates them. The real application video is recorded separately; it is not synthesized from report figures.

## Run the saved model

With Python 3.11 or later, install `requirements.txt`. If you cloned the code repository, run `python fetch_assets.py` once to download and checksum-verify the saved assets from [release v1.0](https://github.com/futurefokus/traffic-sign-recognition-kazakhstan/releases/tag/v1.0). Then run `python app.py`. If you extracted the complete release ZIP, the architecture and weights are already present. No network download of DINOv2 code or weights is needed after extraction, and no training is run. CPU inference is supported; GPU is optional.

The Gradio share URL is temporary and depends on an active runtime. The recovery notebook recreates a new URL if Colab disconnects.

## Supported classes

| Source label | Kazakhstan code | English description |
|---|---|---|
| Main Road | 2.1 | Priority road |
| Give Way | 2.4 | Give way |
| STOP | 2.5 | Stop |
| No Entry | 3.1 | No entry |
| Pedestrian Crossing | 5.16.1 | Pedestrian crossing |

Russian sign names are shown only in prediction labels, as requested. Input must contain a single cropped sign. This is not a full-scene detector or a continuous video detector.

## Experiments

- Baseline: HOG plus standardized Logistic Regression.
- Modern component: frozen official DINOv2 ViT-S/14 plus a linear classifier.
- Ablation: original training crops versus originals plus one augmented copy.
- Evaluation: clean held-out crops, controlled darkening, controlled blur, macro-F1, accuracy, per-class precision/recall/F1, confusion matrix, and warm inference latency.
- Explanation: nearest original training examples by feature cosine similarity.

The seed is 42. Source filename variants and perceptually similar crops are grouped before splitting. This reduces obvious duplicates but does not guarantee physical-sign or capture-sequence independence. No independent Kazakhstan field test was performed. Scores are not calibrated probabilities. The original self-collected paper-sign requirement was not fulfilled.

## Score display and camera

The app displays float64 softmax outputs as decimal strings at full stored floating-point precision. It does not round scores to integer percentages, cap them at 99 percent, or recalibrate them. A score near one can still be wrong. Full stored precision does not mean infinite mathematical precision.

Camera preview was checked in the user's browser at 1280 by 720 after permission was granted. The camera is a snapshot input, not continuous traffic-sign detection. No private webcam frames are included in the public demonstration.

## Data and licenses

Source: https://github.com/TN02/Road-Signs-Dataset. The authors describe a mixture of GTSRB, RTSD, and 230 additional Baku photographs. The repository states MIT; retain its license and respect upstream source terms. The project does not claim all images were photographed in Azerbaijan or Kazakhstan. The bundle contains a selected subset of sign crops and a manifest; the complete 1.48 GB source archive is not duplicated.

DINOv2: https://github.com/facebookresearch/dinov2, Apache License 2.0 for the standard DINOv2 code and weights. Paper: https://arxiv.org/abs/2304.07193.

Sign-number reference: https://old.adilet.zan.kz/rus/docs/V2300033003.

## Deliverables

The [release](https://github.com/futurefokus/traffic-sign-recognition-kazakhstan/releases/tag/v1.0) contains the English seven-page report, a real three-to-five-minute recording of the running application, and the complete project ZIP with saved weights. The report includes the DINOv2 teacher/student pretraining diagram and this project's frozen-encoder pipeline. `metrics.json`, `classification_report.json`, `manifest.json`, and `figures/` contain the experiment evidence. `environment.txt` records installed versions. The video shows actual interaction with the web app; it is not a slideshow of report plots.

Use only for education. Do not use predictions to operate a vehicle.
