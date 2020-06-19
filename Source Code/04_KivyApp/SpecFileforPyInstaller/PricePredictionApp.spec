# -*- mode: python ; coding: utf-8 -*-
from kivy_deps import sdl2, glew
import wcwidth
import os
import importlib
block_cipher = None


a = Analysis(['..\\kivy\\app.py'],
             pathex=['D:\\Course and Learning\\MDS\\WQD7005 Data Mining\\PricePredictionApp'],
             binaries=[],
             datas=[
                 (os.path.dirname(wcwidth.__file__), 'wcwidth'),
                 (os.path.join(os.path.dirname(importlib.import_module('tensorflow').__file__),
                              "lite\\experimental\\microfrontend\\python\\ops\\_audio_microfrontend_op.so"),
                 "tensorflow\\lite\\experimental\\microfrontend\\python\\ops\\")
             ],
             hiddenimports=['tensorflow', 'pkg_resources.py2_warn', 'kivy.garden', 'kivy.garden.matplotlib.backend_kivy',
                            'tensorflow.python.keras.engine.base_layer_v1', 'tensorflow.python.ops.while_v2'
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='PricePredictionApp',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe, Tree('D:\\Course and Learning\\MDS\\WQD7005 Data Mining\\kivy'),
               a.binaries,
               a.zipfiles,
               a.datas,
               *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               upx_exclude=[],
               name='PricePredictionApp')
