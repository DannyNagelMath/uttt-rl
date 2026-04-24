import os, sys, ctypes

if hasattr(sys, '_MEIPASS'):
    torch_lib = os.path.join(sys._MEIPASS, 'torch', 'lib')
    if os.path.isdir(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')

        # Pre-load torch DLLs by absolute path so they're in the Windows module
        # cache before torch/__init__.py calls LoadLibraryExW with restricted
        # search flags — Windows skips the file search for already-loaded modules.
        _load_order = [
            'torch_global_deps.dll',
            'libiomp5md.dll',
            'c10.dll',
            'torch_cpu.dll',
            'torch.dll',
            'torch_python.dll',
            'shm.dll',
        ]
        for dll in _load_order:
            path = os.path.join(torch_lib, dll)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path)
                except OSError:
                    pass
