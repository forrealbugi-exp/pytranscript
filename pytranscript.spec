# -*- mode: python ; coding: utf-8 -*-
"""Spec dla PyInstaller – buduje pytranscript.app dla macOS."""

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'config', 'audio', 'vad', 'transcription', 'postprocess',
        'languages', 'i18n', 'storage',
        'soundcard', 'numpy',
        'mlx_whisper',
        'silero_vad_lite',
        'ollama',
        'huggingface_hub',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'scipy', 'pandas', 'PIL',
        'IPython', 'jupyter', 'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pytranscript',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='pytranscript',
)

app = BUNDLE(
    coll,
    name='pytranscript.app',
    icon=None,
    bundle_identifier='com.pytranscript.app',
    info_plist={
        'NSMicrophoneUsageDescription':
            'Aplikacja potrzebuje dostępu do mikrofonu, aby transkrybować mowę.',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1',
    },
)