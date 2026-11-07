"""Colab training and evaluation for five Kazakhstan-compatible traffic signs."""
import os, sys, json, time, random, pathlib, hashlib, re, shutil, platform
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from skimage.feature import hog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)
ROOT=pathlib.Path('/content/traffic_sign_project')
DATA=ROOT/'Road-Signs-Dataset-main'
OUT=ROOT/'submission'; OUT.mkdir(exist_ok=True,parents=True)
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
CLASSES=[
 {'source':'Main Road','code':'2.1','en':'Priority road','ru':'Главная дорога'},
 {'source':'Give Way','code':'2.4','en':'Give way','ru':'Уступите дорогу'},
 {'source':'STOP','code':'2.5','en':'Stop','ru':'Движение без остановки запрещено'},
 {'source':'No Entry','code':'3.1','en':'No entry','ru':'Въезд запрещен'},
 {'source':'Pedestrian Crossing','code':'5.16.1','en':'Pedestrian crossing','ru':'Пешеходный переход'}]
names=[c['en'] for c in CLASSES]
lookup={c['source'].lower():i for i,c in enumerate(CLASSES)}
(OUT/'classes.json').write_text(json.dumps(CLASSES,ensure_ascii=False,indent=2),encoding='utf-8')
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
hardware={'device':DEVICE,'gpu':torch.cuda.get_device_name(0) if DEVICE=='cuda' else None,
 'python':platform.python_version(),'torch':torch.__version__}
print('Hardware:',hardware,flush=True)

def dhash(im):
    a=np.asarray(im.convert('L').resize((9,8)))
    return int(''.join('1' if v else '0' for v in (a[:,1:]>a[:,:-1]).flatten()),2)

records=[]; raw=Counter(); bad=0
for ann in sorted((DATA/'Annotations').glob('*.xml')):
    try:
        x=ET.parse(ann).getroot(); fn=x.findtext('filename') or ann.stem+'.jpg'
        path=DATA/'JPEGImages'/pathlib.Path(fn).name
        objects=[o for o in x.findall('object') if o.findtext('name','').strip().lower() in lookup]
        if not objects: continue
        with Image.open(path) as inp:
            im=inp.convert('RGB'); w,h=im.size
            for o in objects:
                label=lookup[o.findtext('name').strip().lower()]; raw[label]+=1
                b=o.find('bndbox')
                x1,y1,x2,y2=[int(float(b.findtext(k))) for k in ['xmin','ymin','xmax','ymax']]
                x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
                if min(x2-x1,y2-y1)<16: continue
                crop=im.crop((x1,y1,x2,y2)).resize((128,128))
                base=re.split(r'\.rf\.',path.stem)[0]
                records.append({'label':label,'image':crop,'source':path.name,'base':base,'hash':dhash(crop)})
    except Exception: bad+=1
print('Raw selected boxes:',dict(raw),'usable:',len(records),'invalid files:',bad,flush=True)
assert all(any(r['label']==i for r in records) for i in range(5)), 'A selected class has no usable images'

# Group source filename variants and similar sign appearances before splitting.
# Tracks/physical-sign IDs are unavailable in this mixed collection.
selected=[]
for label in range(5):
    rows=[r for r in records if r['label']==label]
    random.shuffle(rows); rows=rows[:1200]
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b: parent[b]=a
    for i,r in enumerate(rows):
        for j in range(i):
            if r['base']==rows[j]['base'] or (r['hash']^rows[j]['hash']).bit_count()<=3:
                union(i,j)
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[find(i)].append(r)
    usable=list(groups.values()); random.shuffle(usable)
    assert len(usable)>=5, f'Too few independent appearance groups in {names[label]}'
    n=len(usable); nt=max(1,round(n*.15)); nv=max(1,round(n*.15))
    for gi,group in enumerate(usable):
        split='test' if gi<nt else 'val' if gi<nt+nv else 'train'
        # Cap each appearance group and each class/split to keep the run small.
        for r in group[:3]:
            r['group']=f'{label}_{gi}'; r['split']=split; selected.append(r)
cap={'train':300,'val':80,'test':100}
final=[]
for split in cap:
    for label in range(5):
        rows=[r for r in selected if r['split']==split and r['label']==label]
        random.shuffle(rows); final.extend(rows[:cap[split]])
records=final
for i,r in enumerate(records):
    rel=f"crops/{r['split']}/{r['label']}/{i:05d}.jpg"
    p=OUT/rel; p.parent.mkdir(parents=True,exist_ok=True); r['image'].save(p,quality=95)
    r['path']=rel
manifest=[{k:v for k,v in r.items() if k not in ('image','hash')} for r in records]
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
stats={s:[sum(r['split']==s and r['label']==i for r in records) for i in range(5)] for s in cap}
print('Split counts:',stats,flush=True)
train=[r for r in records if r['split']=='train']; val=[r for r in records if r['split']=='val']; test=[r for r in records if r['split']=='test']
assert set(r['group'] for r in train).isdisjoint(r['group'] for r in test)
yt=np.array([r['label'] for r in test])

fig,ax=plt.subplots(figsize=(8,3.5)); xx=np.arange(5)
for off,s in enumerate(['train','val','test']): ax.bar(xx+(off-1)*.25,stats[s],.25,label=s)
ax.set_xticks(xx,names,rotation=12); ax.set_ylabel('Crops'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'dataset_balance.png',dpi=160); plt.close(fig)
fig,axs=plt.subplots(2,5,figsize=(10,4))
for i in range(5):
    rows=[r for r in train if r['label']==i]
    for j in range(2): axs[j,i].imshow(rows[j%len(rows)]['image']); axs[j,i].axis('off'); axs[j,i].set_title(names[i],fontsize=9)
fig.tight_layout();fig.savefig(FIG/'examples.png',dpi=160);plt.close(fig)

def hog_feature(im):
    a=np.asarray(im.convert('L').resize((64,64)),dtype=np.float32)/255.
    return hog(a,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2))
baseline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=400,C=.1,random_state=SEED))
baseline.fit(np.stack([hog_feature(r['image']) for r in train]),[r['label'] for r in train])
pred_b=baseline.predict(np.stack([hog_feature(r['image']) for r in test]))
joblib.dump(baseline,OUT/'baseline.joblib')
print('Baseline macro-F1:',f1_score(yt,pred_b,average='macro'),flush=True)

print('Loading official DINOv2 ViT-S/14 pretrained encoder...',flush=True)
encoder=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',pretrained=True).to(DEVICE).eval()
for p in encoder.parameters(): p.requires_grad=False
norm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
augment=transforms.Compose([transforms.RandomPerspective(.15,p=.6),transforms.RandomRotation(12,fill=110),transforms.ColorJitter(.35,.3,.15,.03),transforms.GaussianBlur(3,sigma=(.1,1.0))])
@torch.inference_mode()
def features(images):
    out=[]
    for i in range(0,len(images),32):
        batch=torch.stack([norm(im) for im in images[i:i+32]]).to(DEVICE)
        out.append(encoder(batch).float().cpu())
    return torch.cat(out)
print('Extracting visual features...',flush=True)
X=features([r['image'] for r in train]); V=features([r['image'] for r in val]); T=features([r['image'] for r in test])
A=features([augment(r['image']) for r in train])
y=torch.tensor([r['label'] for r in train]); vy=torch.tensor([r['label'] for r in val])

def fit_head(x,y,tag):
    torch.manual_seed(SEED)
    mean=x.mean(0); std=x.std(0).clamp_min(.01)
    model=torch.nn.Linear(x.shape[1],5); opt=torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.01)
    curve=[]; best=None; bestloss=float('inf')
    for epoch in range(45):
        model.train(); opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model((x-mean)/std),y); loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            vl=torch.nn.functional.cross_entropy(model((V-mean)/std),vy).item()
        curve.append([float(loss.item()),vl])
        if vl<bestloss: bestloss=vl;best={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    with torch.no_grad(): pred=model((T-mean)/std).argmax(1).numpy()
    return model,mean,std,pred,curve
plain,pm,ps,pred_p,curve_p=fit_head(X,y,'plain')
model,mean,std,pred_m,curve=fit_head(torch.cat([X,A]),torch.cat([y,y]),'augmented')
torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std,'classes':CLASSES},OUT/'dinov2_head.pt')
torch.save(encoder.state_dict(),OUT/'dinov2_vits14_backbone.pt')
torch.save({'features':X,'paths':[r['path'] for r in train],'labels':y},OUT/'retrieval.pt')

def metrics(pred): return {'accuracy':float(accuracy_score(yt,pred)),'macro_f1':float(f1_score(yt,pred,average='macro'))}
results={'baseline':metrics(pred_b),'dinov2_plain':metrics(pred_p),'dinov2_augmented':metrics(pred_m)}
robust={}
for cond in ['dark','blur']:
    ims=[ImageEnhance.Brightness(r['image']).enhance(.4) if cond=='dark' else r['image'].filter(ImageFilter.GaussianBlur(2)) for r in test]
    F=features(ims)
    with torch.no_grad():
        p0=plain((F-pm)/ps).argmax(1).numpy();p1=model((F-mean)/std).argmax(1).numpy()
    robust[cond]={'plain':metrics(p0),'augmented':metrics(p1)}
results['conditions']=robust
(OUT/'classification_report.json').write_text(json.dumps(classification_report(yt,pred_m,target_names=names,output_dict=True,zero_division=0),indent=2))
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(yt,pred_m,display_labels=names,ax=ax,xticks_rotation=30,colorbar=False);fig.tight_layout();fig.savefig(FIG/'confusion_matrix.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.5));ax.plot(np.array(curve)[:,0],label='Train');ax.plot(np.array(curve)[:,1],label='Validation');ax.set(xlabel='Epoch',ylabel='Cross entropy');ax.legend();fig.tight_layout();fig.savefig(FIG/'training_curves.png',dpi=170);plt.close(fig)

@torch.inference_mode()
def predict_image(im,condition='Normal'):
    im=im.convert('RGB')
    if condition=='Dark':im=ImageEnhance.Brightness(im).enhance(.4)
    if condition=='Blur':im=im.filter(ImageFilter.GaussianBlur(2))
    if DEVICE=='cuda':torch.cuda.synchronize()
    start=time.perf_counter(); f=features([im]); logits=model((f-mean)/std); probs=logits.double().softmax(1)[0].numpy()
    if DEVICE=='cuda':torch.cuda.synchronize()
    ms=(time.perf_counter()-start)*1000
    k=int(probs.argmax()); sim=torch.nn.functional.cosine_similarity(f,X); idx=sim.topk(min(3,len(train))).indices.tolist()
    return im,k,probs,ms,idx
lat=[]
predict_image(test[0]['image'])
for r in test[:20]:lat.append(predict_image(r['image'])[3])
results['latency_ms_mean']=float(np.mean(lat));results['latency_samples']=len(lat)
results['hardware']=hardware;results['counts']=stats;results['seed']=SEED;results['invalid_xml_or_images']=bad
(OUT/'metrics.json').write_text(json.dumps(results,indent=2))
print('RESULTS',json.dumps(results),flush=True)
fig,axs=plt.subplots(1,4,figsize=(10,3)); query,k,pr,ms,idx=predict_image(test[0]['image'])
axs[0].imshow(query);axs[0].set_title('Test query')
for a,j in zip(axs[1:],idx):a.imshow(train[j]['image']);a.set_title('Train: '+names[train[j]['label']],fontsize=9)
for a in axs:a.axis('off')
fig.tight_layout();fig.savefig(FIG/'explanation.png',dpi=170);plt.close(fig)

def build_app():
    import gradio as gr
    def infer(im,condition):
        if im is None:raise gr.Error('Upload or capture one cropped traffic sign.')
        shown,k,probs,ms,idx=predict_image(im,condition)
        c=CLASSES[k]
        rows=[[names[i],repr(float(probs[i]))] for i in np.argsort(-probs)]
        return f"{c['code']} {c['ru']}",rows,f'{ms:.1f} ms | {hardware["gpu"] or DEVICE} | Browser upload/snapshot',[(train[j]['image'],names[train[j]['label']]) for j in idx],shown
    with gr.Blocks(title='Traffic Sign Recognition for Kazakhstan') as demo:
        gr.Markdown('# Traffic Sign Recognition for Kazakhstan\nUpload a **cropped sign** or take a camera snapshot. Five supported sign types.\nPredictions use Kazakhstan sign codes and Russian sign names. This is an educational classifier, not a road safety system.')
        with gr.Row():
            with gr.Column():
                im=gr.Image(type='pil',sources=['upload','webcam'],label='One traffic sign')
                cond=gr.Radio(['Normal','Dark','Blur'],value='Normal',label='Simulate conditions')
                run=gr.Button('Recognize sign',variant='primary')
            with gr.Column():
                label=gr.Textbox(label='Sign identification')
                scores=gr.Dataframe(headers=['Class','Softmax score (0 to 1)'],datatype=['str','str'],type='array',interactive=False,label='Model scores - full stored precision')
                gr.Markdown('Scores are decimal model outputs, not calibrated confidence estimates.')
                latency=gr.Textbox(label='Latency and environment');shown=gr.Image(label='Processed input')
        gallery=gr.Gallery(label='Visual explanation - nearest training examples',columns=3)
        gr.Examples([[str(OUT/r['path']),'Normal'] for r in [next(t for t in test if t['label']==i) for i in range(5)]],inputs=[im,cond])
        gr.Markdown('Data: TN02/Road-Signs-Dataset, a mixture of public datasets and 230 Baku photographs. No claim of Kazakhstan field validation. Synthetic weather transformations are explicitly simulated. Unsupported objects can receive incorrect predictions.')
        if pathlib.Path('/content/Traffic_Sign_Project.zip').exists():
            gr.File(value='/content/Traffic_Sign_Project.zip',label='Download complete project')
            gr.File(value=str(OUT/'Report.pdf'),label='Download English report')
            if (OUT/'Demo.mp4').exists(): gr.File(value=str(OUT/'Demo.mp4'),label='Download real application demonstration')
        run.click(infer,[im,cond],[label,scores,latency,gallery,shown])
    return demo

ENGLISH_JS="""() => {
 const translations = {'Перетащите изображение сюда':'Drop an image here','Нажмите для загрузки':'Click to upload','- или -':'- or -','Загрузка':'Loading','Очистить':'Clear','Скачать':'Download','Редактировать':'Edit','Отмена':'Cancel','Сохранить':'Save','Настройки':'Settings','Создано с помощью Gradio':'Built with Gradio','Использовать через API':'Use via API'};
 const translate = () => {
   const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
   let node; while (node = walk.nextNode()) { const key=node.nodeValue.trim(); if(translations[key]) node.nodeValue=node.nodeValue.replace(key,translations[key]); }
   document.querySelectorAll('[aria-label],[title]').forEach(el => { for(const attr of ['aria-label','title']) { const val=el.getAttribute(attr); if(translations[val]) el.setAttribute(attr,translations[val]); } });
 };
 translate(); new MutationObserver(translate).observe(document.body,{childList:true,subtree:true,characterData:true});
}"""

def force_english(demo):
    """Keep built-in Gradio labels English, independent of browser locale."""
    class EnglishLocale:
        def __init__(self, app): self.app=app
        async def __call__(self, scope, receive, send):
            if scope['type']=='http':
                scope=dict(scope)
                scope['headers']=[(k,v) for k,v in scope.get('headers',[]) if k.lower()!=b'accept-language']+[(b'accept-language',b'en')]
            await self.app(scope,receive,send)
    demo.app.middleware_stack=EnglishLocale(demo.app.middleware_stack)

print('TRAINING_COMPLETE',flush=True)
