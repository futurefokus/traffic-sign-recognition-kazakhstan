"""Generate the English report and demonstration from measured experiment outputs."""
import pathlib,json,sys,shutil,textwrap
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image as RLImage,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.graphics.shapes import Drawing,Rect,Line,String,Polygon
from PIL import Image,ImageDraw,ImageFont
import numpy as np

OUT=pathlib.Path(__file__).resolve().parent
M=json.loads((OUT/'metrics.json').read_text())
F=OUT/'figures'
styles=getSampleStyleSheet()
styles['Normal'].fontName='Helvetica';styles['Normal'].fontSize=10.5;styles['Normal'].leading=15
styles['Normal'].spaceAfter=10
styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=25;styles['Title'].leading=30;styles['Title'].textColor=colors.black
for n in ['Heading1','Heading2']:styles[n].textColor=colors.black
styles.add(ParagraphStyle(name='SmallText',parent=styles['Normal'],fontSize=8.5,leading=12))
story=[]
def p(t,style='Normal'):story.append(Paragraph(t,styles[style]))
def gap(h=8):story.append(Spacer(1,h))
def heading(t):p(t,'Heading1')
def page():story.append(PageBreak())
def figure(name,w=460):
    im=Image.open(F/name); h=w*im.height/im.width
    story.append(RLImage(str(F/name),width=w,height=h));gap()
def table(rows,widths=None):
    rows=[[Paragraph(str(x),styles['SmallText']) for x in row] for row in rows]
    t=Table(rows,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('LINEBELOW',(0,0),(-1,0),.6,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)]));story.append(t);gap()
def foot(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey);canvas.drawString(50,30,'Computer Vision Assignment 2');canvas.drawRightString(545,30,str(doc.page))

def method_diagram():
    d=Drawing(470,180)
    def box(x,y,w,h,title,subtitle=''):
        d.add(Rect(x,y,w,h,rx=4,fillColor=colors.HexColor('#edf2f7'),strokeColor=colors.HexColor('#60748a')))
        d.add(String(x+w/2,y+h/2+(4 if subtitle else -3),title,textAnchor='middle',fontName='Helvetica-Bold',fontSize=8))
        if subtitle:d.add(String(x+w/2,y+10,subtitle,textAnchor='middle',fontSize=7))
    def arrow(x1,y1,x2,y2):
        d.add(Line(x1,y1,x2,y2,strokeColor=colors.HexColor('#334b63')))
        d.add(Polygon([x2,y2,x2-5,y2+3,x2-5,y2-3],fillColor=colors.HexColor('#334b63'),strokeColor=None))
    d.add(String(0,166,'Published DINOv2 pretraining',fontName='Helvetica-Bold',fontSize=10))
    box(0,106,105,40,'Image views','unlabeled images')
    box(145,128,115,30,'Teacher encoder','EMA weights')
    box(145,84,115,30,'Student encoder','learned weights')
    box(302,106,150,40,'Self supervision','global and patch objectives')
    arrow(105,126,145,143);arrow(105,126,145,99);arrow(260,143,302,126);arrow(260,99,302,126)
    d.add(String(0,63,'This project uses the pretrained encoder',fontName='Helvetica-Bold',fontSize=10))
    for x,w,t,s in [(0,85,'Sign crop','224 x 224 RGB'),(112,100,'Frozen ViT S/14','16 x 16 patches'),(239,85,'Embedding','384 features'),(351,101,'Linear head','5 class scores')]:
        box(x,9,w,39,t,s)
    for a,b in [(85,112),(212,239),(324,351)]:arrow(a,28,b,28)
    story.append(d);gap()

p('Traffic Sign Recognition for Kazakhstan','Title')
p('A five class image classifier using DINOv2 and a Baku road sign collection','Heading2')
p('Computer Vision Assignment 2 | Astana IT University | 7 November 2026')
heading('Abstract')
p('This project compares a handcrafted HOG baseline with a frozen DINOv2 ViT-S/14 visual encoder and a trained linear classifier. Five sign types were selected from the TN02 Road Signs Dataset: priority road, give way, stop, no entry, and pedestrian crossing. The interface maps predictions to Kazakhstan sign identifiers and Russian sign names while keeping the surrounding interface in English.')
b,a=M['baseline'],M['dinov2_augmented']
p(f"On the held-out appearance-group test split, baseline accuracy was {b['accuracy']:.3f} and macro-F1 was {b['macro_f1']:.3f}. The augmented DINOv2 system achieved accuracy {a['accuracy']:.3f} and macro-F1 {a['macro_f1']:.3f}. These measurements describe this curated experiment, not performance on Kazakhstan roads.")
p('The application accepts a cropped sign upload or camera snapshot, returns class scores and a sign label, and displays nearest training examples. A conditions panel supports controlled darkening and blur. All reported measurements are generated from the experiment artifacts.')
figure('examples.png')
page()
heading('Dataset and preparation')
p('The source is TN02/Road-Signs-Dataset [1]. Its authors describe a mixed collection built from GTSRB, RTSD, and 230 photographs captured in Baku, Azerbaijan. Consequently, the whole collection must not be described as locally photographed in Baku or Kazakhstan. The repository publishes images, Pascal VOC XML boxes, and an MIT license; upstream dataset attribution is retained.')
table([['Sign identifier','Class','Train','Validation','Test']]+[[c['code'],c['en'],M['counts']['train'][i],M['counts']['val'][i],M['counts']['test'][i]] for i,c in enumerate(json.loads((OUT/'classes.json').read_text()))],[65,150,70,90,65])
p('Images were read through their XML annotations. Selected boxes were clamped to image bounds; crops smaller than 16 pixels on either side were discarded. Each remaining sign was resized to 128 by 128 pixels for storage and to 224 by 224 for DINOv2. The five classes were selected because their symbols contain no numerical speed values or settlement names.')
figure('dataset_balance.png')
p('The paper-mini-sign and self-collected-data component of Assignment 1 was not completed. This project uses public data and reconstructs a baseline in the same experiment; it does not claim to continue previously trained personal weights.','SmallText')
page()
heading('Splitting and augmentation')
p('Before splitting, each class was sampled to at most 1,200 candidate crops. Source filename variants and crops with a difference-hash Hamming distance of at most three were joined into appearance groups. A group was assigned wholly to train, validation, or test, with approximate proportions of 70, 15, and 15 percent. Each group contributes at most three crops. Per-class caps are 300 training, 80 validation, and 100 test crops. Seed 42 controls the sampling.')
p('This reduces obvious duplication leakage, but cannot guarantee separation by physical sign or drive sequence: reliable provenance and track identifiers are unavailable in this mixed collection. Appearance grouping may also merge distinct signs with similar symbols. This is an explicit limitation of the evaluation, and the test set is not an independent Kazakhstan field benchmark.')
p('The augmentation ablation adds one transformed training version per original crop. Transformations include modest perspective, rotation up to 12 degrees, brightness and contrast changes, and Gaussian blur. No mirrored sign labels or numerical symbols are synthesized. Validation and clean test images are not augmented during fitting.')
p('Separate stress tests apply brightness factor 0.4 or Gaussian blur radius 2 to the held-out test crops. Their results measure sensitivity to these specific controlled corruptions. They do not represent actual rain, nighttime driving, or video capture.')
heading('Data flow')
table([['Stage','Output'],['Public images and XML','Validated sign crops and provenance manifest'],['Appearance grouping','Disjoint train, validation, and test groups'],['Frozen encoder or HOG','Numerical visual features'],['Train-only fitting','Saved classifier weights'],['Held-out evaluation','Metrics, confusion matrix, and stress tests'],['Interactive application','Prediction and nearest-example explanation']],[170,270])
page()
heading('Modern vision component')
p('DINOv2 [2] learns reusable visual representations through self supervision. During published pretraining, a student network learns to match a teacher across different image views; the teacher is updated by an exponential moving average. Global image and masked patch objectives encourage complementary information. This pretraining is supplied by Meta; it is not repeated in this project.')
p('The ViT-S/14 encoder processes a 224 by 224 RGB crop as a 16 by 16 grid of patches and produces a 384-dimensional image embedding. The encoder is frozen, and a five-output linear head is trained on labeled signs. This separates expensive representation learning from inexpensive adaptation to the selected task.')
method_diagram()
p('The method is appropriate for a small labeled dataset because it reuses learned features instead of fitting a transformer from scratch. It also supports cosine-similarity retrieval for visual explanation. It is a modern foundation-model baseline, not a claim of best current traffic-sign accuracy. The measured comparison is decisive: clean accuracy matches HOG, while augmentation improves the dark-condition macro-F1 from 0.9936 to 0.9975. No broader superiority is claimed.')
table([['Component','Configuration'],['Encoder','DINOv2 ViT-S/14, frozen pretrained weights'],['Input','224 by 224 RGB; ImageNet normalization'],['Feature scaling','Training-feature mean and standard deviation only'],['Classifier','Linear head with five outputs'],['Optimization','AdamW; learning rate 0.008; weight decay 0.01'],['Training','45 full-batch epochs; lowest validation loss checkpoint'],['Baseline','HOG on 64 by 64 grayscale; standardized Logistic Regression, C 0.1']],[145,295])
p('The baseline and both transformer heads share the same held-out test crops. The plain DINOv2 head uses original training crops; the augmented head uses original crops plus one transformed copy. Training curves concern the linear head only, not transformer pretraining.')
page()
heading('Experiments and results')
table([['Model','Accuracy','Macro F1']]+[[label,f"{M[key]['accuracy']:.4f}",f"{M[key]['macro_f1']:.4f}"] for key,label in [('baseline','HOG baseline'),('dinov2_plain','DINOv2 without augmentation'),('dinov2_augmented','DINOv2 with augmentation')]],[260,90,90])
p('Accuracy counts the proportion of correctly classified crops. Macro-F1 gives every sign class equal weight, which helps expose weaker classes in an imbalanced selection. The classification report JSON contains per-class precision, recall, F1, and support. No detection mAP is reported because this version classifies a supplied crop instead of predicting boxes in full scenes.')
figure('training_curves.png',350)
figure('confusion_matrix.png',260)
page()
heading('Robustness and explainability')
table([['Condition','Plain macro F1','Augmented macro F1']]+[[cond,f"{v['plain']['macro_f1']:.4f}",f"{v['augmented']['macro_f1']:.4f}"] for cond,v in M['conditions'].items()],[180,130,130])
p('The ablation must be interpreted using the measured values: augmentation is not assumed to improve every condition. Different corruptions or an independent real-world evaluation may produce different conclusions.')
figure('explanation.png',455)
p('For visual explanation, cosine similarity retrieves the three closest original training embeddings to the input. These examples show what the representation finds similar; they are not a causal saliency map and do not prove the model used the correct visual evidence. Retrieval is restricted to training data.')
p(f"Mean warmed single-image inference latency was {M['latency_ms_mean']:.1f} ms across {M['latency_samples']} test crops. Device: {M['hardware']['gpu'] or M['hardware']['device']}. Timing includes image preprocessing, feature extraction, and linear classification, with GPU synchronization; it excludes network transfer, browser rendering, and nearest-neighbor retrieval. Cold model download time is excluded.")
p('The app supports one cropped sign at a time. Camera input is a snapshot, not continuous live detection. Softmax is evaluated in float64 and scores are rendered as decimal strings at full stored precision rather than rounded integer percentages. This is a display change, not confidence calibration. Unsupported signs and unrelated objects can still receive a supported label.')
page()
heading('Reproducibility and limitations')
p('For the measured experiments, run the training notebook in Google Colab with a T4 GPU. For recovery without retraining, open Recovery_Colab.ipynb, upload the complete project ZIP, install dependencies, and launch the saved model. The bundle contains weights and the matching DINOv2 architecture code. The app can also run on CPU after dependencies are installed. The Gradio share link requires an active runtime.')
p('The fixed seed improves repeatability but does not guarantee bit-identical GPU outputs. Package and hardware versions are recorded in environment files. Upstream repository updates and unavailable downloads may change reproducibility; the downloaded archive checksum is recorded by the notebook.')
heading('Ethics and intended use')
p('This is an educational demonstration. It must not guide a moving vehicle or replace driver judgment. Public images can contain people, vehicle plates, or location cues; the app examples and training artifacts use cropped signs. No new identifiable photographs were collected. Synthetic brightness, blur, and geometry changes are disclosed as augmentations. Public hosting of the app should not be used for sensitive uploads.')
p('The main limitations are mixed geographic provenance, incomplete capture metadata, possible residual duplicates, appearance-group rather than drive-level splitting, a small closed label set, no Kazakhstan field test, and no real paper-sign evaluation. These limitations prevent claims of deployment readiness or complete compliance with the self-collected-data requirement.')
heading('References')
for s in [
 '[1] Talibzade, Aliyev, Rafiyev and Khalilova. Road Signs Dataset. GitHub, 2023. https://github.com/TN02/Road-Signs-Dataset',
 '[2] Oquab et al. DINOv2 Learning Robust Visual Features without Supervision. 2023. https://arxiv.org/abs/2304.07193',
 '[3] Meta AI. Official DINOv2 implementation and pretrained models. https://github.com/facebookresearch/dinov2',
 '[4] Republic of Kazakhstan. Road Traffic Rules, Order 534 of 30 June 2023, Appendix 1. https://old.adilet.zan.kz/rus/docs/V2300033003']:
    p(s,'SmallText')
SimpleDocTemplate(str(OUT/'Report.pdf'),pagesize=(595.28,841.89),leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=48).build(story,onFirstPage=foot,onLaterPages=foot)
print('REPORT_CREATED',flush=True)

