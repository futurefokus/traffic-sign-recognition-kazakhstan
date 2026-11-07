"""Run from the extracted submission directory: python app.py."""
import pathlib,json,time,os
import numpy as np
from PIL import ImageEnhance,ImageFilter,Image
import torch
from torchvision import transforms
import gradio as gr

ROOT=pathlib.Path(__file__).resolve().parent
device=os.environ.get('TRAFFIC_SIGN_DEVICE','cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)
architecture=ROOT/'dinov2_source'
if not (architecture/'hubconf.py').exists():
    raise FileNotFoundError('Missing bundled DINOv2 architecture. Extract the complete project ZIP before launching.')
enc=torch.hub.load(str(architecture),'dinov2_vits14',source='local',pretrained=False).to(device).eval()
enc.load_state_dict(torch.load(ROOT/'dinov2_vits14_backbone.pt',map_location=device,weights_only=True))
ck=torch.load(ROOT/'dinov2_head.pt',map_location='cpu',weights_only=True)
head=torch.nn.Linear(384,5);head.load_state_dict(ck['state_dict']);head.eval()
mean,std=ck['mean'],ck['std'];classes=ck['classes']
retr=torch.load(ROOT/'retrieval.pt',map_location='cpu',weights_only=True)
norm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
@torch.inference_mode()
def infer(im,condition):
    if im is None:raise gr.Error('Upload or capture one cropped sign.')
    im=im.convert('RGB')
    if condition=='Dark':im=ImageEnhance.Brightness(im).enhance(.4)
    if condition=='Blur':im=im.filter(ImageFilter.GaussianBlur(2))
    if device=='cuda':torch.cuda.synchronize()
    t=time.perf_counter();f=enc(norm(im).unsqueeze(0).to(device)).float().cpu()
    probs=head((f-mean)/std).double().softmax(1)[0].numpy()
    if device=='cuda':torch.cuda.synchronize()
    ms=(time.perf_counter()-t)*1000;k=int(probs.argmax());c=classes[k]
    ids=torch.nn.functional.cosine_similarity(f,retr['features']).topk(3).indices.tolist()
    examples=[(Image.open(ROOT/retr['paths'][i]).copy(),classes[int(retr['labels'][i])]['en']) for i in ids]
    env=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'
    rows=[[classes[i]['en'],repr(float(probs[i]))] for i in np.argsort(-probs)]
    return f"{c['code']} {c['ru']}",rows,f'{ms:.1f} ms | {env}',examples,im
with gr.Blocks(title='Traffic Sign Recognition for Kazakhstan') as demo:
    gr.Markdown('# Traffic Sign Recognition for Kazakhstan\nUpload or capture **one cropped sign**. Five supported classes. Educational demonstration.')
    with gr.Row():
        with gr.Column():
            image=gr.Image(type='pil',sources=['upload','webcam'],label='One traffic sign')
            condition=gr.Radio(['Normal','Dark','Blur'],value='Normal',label='Simulate conditions')
            run=gr.Button('Recognize sign',variant='primary')
        with gr.Column():
            label=gr.Textbox(label='Sign identification')
            scores=gr.Dataframe(headers=['Class','Softmax score (0 to 1)'],datatype=['str','str'],type='array',interactive=False,label='Model scores - full stored precision')
            gr.Markdown('Scores are displayed as decimal numbers without integer percentage rounding. They are not calibrated confidence estimates.')
            latency=gr.Textbox(label='Latency and environment');shown=gr.Image(label='Processed input')
    gallery=gr.Gallery(label='Nearest training examples',columns=3)
    run.click(infer,[image,condition],[label,scores,latency,gallery,shown])
    manifest=json.loads((ROOT/'manifest.json').read_text())
    samples=[next(r for r in manifest if r['split']=='test' and r['label']==i) for i in range(5)]
    gr.Examples([[str(ROOT/r['path']),'Normal'] for r in samples],inputs=[image,condition])
    gr.Markdown('Data: TN02/Road-Signs-Dataset; mixed public data and additional Baku photos. Kazakhstan field performance is unverified. Unsupported inputs may receive incorrect labels. Dark and Blur are simulated transformations.')
def force_english(demo):
    class EnglishLocale:
        def __init__(self, app): self.app=app
        async def __call__(self,scope,receive,send):
            if scope['type']=='http':
                scope=dict(scope)
                scope['headers']=[(k,v) for k,v in scope.get('headers',[]) if k.lower()!=b'accept-language']+[(b'accept-language',b'en')]
            await self.app(scope,receive,send)
    demo.app.middleware_stack=EnglishLocale(demo.app.middleware_stack)
ENGLISH_JS="""() => {
 const translations = {'Перетащите изображение сюда':'Drop an image here','Нажмите для загрузки':'Click to upload','- или -':'- or -','Загрузка':'Loading','Очистить':'Clear','Скачать':'Download','Редактировать':'Edit','Отмена':'Cancel','Сохранить':'Save','Настройки':'Settings','Создано с помощью Gradio':'Built with Gradio','Использовать через API':'Use via API'};
 const translate=()=>{const walk=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);let node;while(node=walk.nextNode()){const key=node.nodeValue.trim();if(translations[key])node.nodeValue=node.nodeValue.replace(key,translations[key]);}document.querySelectorAll('[aria-label],[title]').forEach(el=>{for(const attr of ['aria-label','title']){const val=el.getAttribute(attr);if(translations[val])el.setAttribute(attr,translations[val]);}});};
 translate();new MutationObserver(translate).observe(document.body,{childList:true,subtree:true,characterData:true});
}"""
if __name__=='__main__':
    import threading
    demo.launch(prevent_thread_lock=True,js=ENGLISH_JS,css='footer {display:none !important}')
    force_english(demo)
    threading.Event().wait()
