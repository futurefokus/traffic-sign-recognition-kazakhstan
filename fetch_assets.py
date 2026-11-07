"""Download and verify the published saved-model bundle. No training is run."""
import hashlib,pathlib,urllib.request,zipfile,io
ROOT=pathlib.Path(__file__).resolve().parent
BASE='https://github.com/futurefokus/traffic-sign-recognition-kazakhstan/releases/download/v1.0/'
def main():
    print('Downloading saved model assets from the project release...',flush=True)
    raw=urllib.request.urlopen(BASE+'Traffic_Sign_Project.zip',timeout=180).read()
    expected=urllib.request.urlopen(BASE+'Traffic_Sign_Project.zip.sha256',timeout=30).read().decode().split()[0]
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected:raise RuntimeError('Archive checksum mismatch; no files were extracted.')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            name=info.filename
            if info.is_dir():continue
            if name.startswith(('dinov2_source/','crops/')) or name in ['dinov2_vits14_backbone.pt','dinov2_head.pt','retrieval.pt','classes.json','manifest.json','metrics.json','baseline.joblib']:
                target=(ROOT/name).resolve()
                if ROOT not in target.parents:raise RuntimeError('Unsafe archive path')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
    print('Saved assets ready. Run python app.py.',flush=True)
if __name__=='__main__':main()
