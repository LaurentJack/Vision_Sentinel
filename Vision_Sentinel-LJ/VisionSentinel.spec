# VisionSentinel.spec
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
hidden = []
hidden += collect_submodules('ultralytics')
hidden += collect_submodules('torch')
hidden += collect_submodules('torchvision')
hidden += collect_submodules('insightface')
hidden += collect_submodules('onnxruntime')
hidden += collect_submodules('cv2')

datas = []
datas += collect_data_files('ultralytics')
datas += collect_data_files('insightface')
datas += collect_data_files('onnxruntime')
datas += [('app/config.yaml', 'app')]
datas += [('app/data/facebank', 'app/data/facebank')]

block_cipher = None

a = Analysis(
    ['app/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='VisionSentinel',
    console=False,   # True si tu veux voir les logs console
    icon=None
)
