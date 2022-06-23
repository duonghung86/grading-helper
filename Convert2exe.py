import PyInstaller.__main__

PyInstaller.__main__.run([
    'grading_helper_v1.py',
    # '--onefile',
    '--nowindow',
    '''--add-data=gh_gui.ui;.''',
    '''--add-data=mnist-8.onnx;.''',
    '--noconfirm',
    #'--log-level=WARN'
])