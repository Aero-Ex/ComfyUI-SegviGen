from typing import *

BACKEND = 'sdpa' 
DEBUG = False

def __detect_backend():
    global BACKEND
    try:
        import flash_attn_interface
        BACKEND = 'flash_attn_3'
    except ImportError:
        try:
            import flash_attn
            BACKEND = 'flash_attn'
        except ImportError:
            try:
                import xformers
                BACKEND = 'xformers'
            except ImportError:
                BACKEND = 'sdpa'

def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    __detect_backend()
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
