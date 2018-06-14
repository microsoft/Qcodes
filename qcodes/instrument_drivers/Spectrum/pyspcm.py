import os
import platform
import sys
from ctypes import c_int8, c_int16, c_int32, c_int64, c_uint8, c_uint16, c_uint32, c_uint64, c_char_p, POINTER, c_void_p, cdll
from ctypes import *

# load registers for easier access
from py_header.regs import *

# load registers for easier access
from py_header.spcerr import *

SPCM_DIR_PCTOCARD = 0
SPCM_DIR_CARDTOPC = 1

SPCM_BUF_DATA      = 1000 # main data buffer for acquired or generated samples
SPCM_BUF_ABA       = 2000 # buffer for ABA data, holds the A-DATA (slow samples)
SPCM_BUF_TIMESTAMP = 3000 # buffer for timestamps


# determine bit width of os
oPlatform = platform.architecture()
if (oPlatform[0] == '64bit'):
    bIs64Bit = 1
else:
    bIs64Bit = 0

# define pointer aliases
int8  = c_int8
int16 = c_int16
int32 = c_int32
int64 = c_int64

ptr8  = POINTER (int8)
ptr16 = POINTER (int16)
ptr32 = POINTER (int32)
ptr64 = POINTER (int64)

uint8  = c_uint8
uint16 = c_uint16
uint32 = c_uint32
uint64 = c_uint64

uptr8  = POINTER (uint8)
uptr16 = POINTER (uint16)
uptr32 = POINTER (uint32)
uptr64 = POINTER (uint64)

# Windows
if os.name == 'nt':
    sys.stdout.write("Windows found")

    # define card handle type
    if (bIs64Bit):
        # for unknown reasons c_void_p gets messed up on Win7/64bit, but this works:
        drv_handle = POINTER(c_uint64)
    else:
        drv_handle = c_void_p # type: ignore

    # Load DLL into memory.
    # use windll because all driver access functions use _stdcall calling convention under windows
    if (bIs64Bit == 1):
        spcmDll = windll.LoadLibrary ("c:\\windows\\system32\\spcm_win64.dll") # type: ignore
    else:
        spcmDll = windll.LoadLibrary ("c:\\windows\\system32\\spcm_win32.dll") # type: ignore

    # load spcm_hOpen
    if (bIs64Bit):
        spcm_hOpen = getattr (spcmDll, "spcm_hOpen")
    else:
        spcm_hOpen = getattr (spcmDll, "_spcm_hOpen@4")
    spcm_hOpen.argtype = [c_char_p]
    spcm_hOpen.restype = drv_handle 

    # load spcm_vClose
    if (bIs64Bit):
        spcm_vClose = getattr (spcmDll, "spcm_vClose")
    else:
        spcm_vClose = getattr (spcmDll, "_spcm_vClose@4")
    spcm_vClose.argtype = [drv_handle]
    spcm_vClose.restype = None

    # load spcm_dwGetErrorInfo
    if (bIs64Bit):
        spcm_dwGetErrorInfo_i32 = getattr (spcmDll, "spcm_dwGetErrorInfo_i32")
    else:
        spcm_dwGetErrorInfo_i32 = getattr (spcmDll, "_spcm_dwGetErrorInfo_i32@16")
    spcm_dwGetErrorInfo_i32.argtype = [drv_handle, uptr32, ptr32, c_char_p]
    spcm_dwGetErrorInfo_i32.restype = uint32

    # load spcm_dwGetParam_i32
    if (bIs64Bit):
        spcm_dwGetParam_i32 = getattr (spcmDll, "spcm_dwGetParam_i32")
    else:
        spcm_dwGetParam_i32 = getattr (spcmDll, "_spcm_dwGetParam_i32@12")
    spcm_dwGetParam_i32.argtype = [drv_handle, int32, ptr32]
    spcm_dwGetParam_i32.restype = uint32

    # load spcm_dwGetParam_i64
    if (bIs64Bit):
        spcm_dwGetParam_i64 = getattr (spcmDll, "spcm_dwGetParam_i64")
    else:
        spcm_dwGetParam_i64 = getattr (spcmDll, "_spcm_dwGetParam_i64@12")
    spcm_dwGetParam_i64.argtype = [drv_handle, int32, ptr64]
    spcm_dwGetParam_i64.restype = uint32

    # load spcm_dwSetParam_i32
    if (bIs64Bit):
        spcm_dwSetParam_i32 = getattr (spcmDll, "spcm_dwSetParam_i32")
    else:
        spcm_dwSetParam_i32 = getattr (spcmDll, "_spcm_dwSetParam_i32@12")
    spcm_dwSetParam_i32.argtype = [drv_handle, int32, int32]
    spcm_dwSetParam_i32.restype = uint32

    # load spcm_dwSetParam_i64
    if (bIs64Bit):
        spcm_dwSetParam_i64 = getattr (spcmDll, "spcm_dwSetParam_i64")
    else:
        spcm_dwSetParam_i64 = getattr (spcmDll, "_spcm_dwSetParam_i64@16")
    spcm_dwSetParam_i64.argtype = [drv_handle, int32, int64]
    spcm_dwSetParam_i64.restype = uint32

    # load spcm_dwSetParam_i64m
    if (bIs64Bit):
        spcm_dwSetParam_i64m = getattr (spcmDll, "spcm_dwSetParam_i64m")
    else:
        spcm_dwSetParam_i64m = getattr (spcmDll, "_spcm_dwSetParam_i64m@16")
    spcm_dwSetParam_i64m.argtype = [drv_handle, int32, int32, int32]
    spcm_dwSetParam_i64m.restype = uint32

    # load spcm_dwDefTransfer_i64
    if (bIs64Bit):
        spcm_dwDefTransfer_i64 = getattr (spcmDll, "spcm_dwDefTransfer_i64")
    else:
        spcm_dwDefTransfer_i64 = getattr (spcmDll, "_spcm_dwDefTransfer_i64@36")
    spcm_dwDefTransfer_i64.argtype = [drv_handle, uint32, uint32, uint32, c_void_p, uint64, uint64]
    spcm_dwDefTransfer_i64.restype = uint32

    # load spcm_dwInvalidateBuf
    if (bIs64Bit):
        spcm_dwInvalidateBuf = getattr (spcmDll, "spcm_dwInvalidateBuf")
    else:
        spcm_dwInvalidateBuf = getattr (spcmDll, "_spcm_dwInvalidateBuf@8")
    spcm_dwInvalidateBuf.argtype = [drv_handle, uint32]
    spcm_dwInvalidateBuf.restype = uint32

    # load spcm_dwGetContBuf_i64
    if (bIs64Bit):
        spcm_dwGetContBuf_i64 = getattr (spcmDll, "spcm_dwGetContBuf_i64")
    else:
        spcm_dwGetContBuf_i64 = getattr (spcmDll, "_spcm_dwGetContBuf_i64@16")
    spcm_dwGetContBuf_i64.argtype = [drv_handle, uint32, POINTER(c_void_p), uptr64]
    spcm_dwGetContBuf_i64.restype = uint32


elif os.name == 'posix':
    sys.stdout.write("Linux found")

    # define card handle type
    drv_handle = c_void_p # type: ignore

    # Load DLL into memory.
    # use cdll because all driver access functions use cdecl calling convention under linux 
    spcmDll = cdll.LoadLibrary ("libspcm_linux.so") # type: ignore

    # load spcm_hOpen
    spcm_hOpen = getattr (spcmDll, "spcm_hOpen")
    spcm_hOpen.argtype = [c_char_p]
    spcm_hOpen.restype = drv_handle 

    # load spcm_vClose
    spcm_vClose = getattr (spcmDll, "spcm_vClose")
    spcm_vClose.argtype = [drv_handle]
    spcm_vClose.restype = None

    # load spcm_dwGetErrorInfo
    spcm_dwGetErrorInfo_i32 = getattr (spcmDll, "spcm_dwGetErrorInfo_i32")
    spcm_dwGetErrorInfo_i32.argtype = [drv_handle, uptr32, ptr32, c_char_p]
    spcm_dwGetErrorInfo_i32.restype = uint32

    # load spcm_dwGetParam_i32
    spcm_dwGetParam_i32 = getattr (spcmDll, "spcm_dwGetParam_i32")
    spcm_dwGetParam_i32.argtype = [drv_handle, int32, ptr32]
    spcm_dwGetParam_i32.restype = uint32

    # load spcm_dwGetParam_i64
    spcm_dwGetParam_i64 = getattr (spcmDll, "spcm_dwGetParam_i64")
    spcm_dwGetParam_i64.argtype = [drv_handle, int32, ptr64]
    spcm_dwGetParam_i64.restype = uint32

    # load spcm_dwSetParam_i32
    spcm_dwSetParam_i32 = getattr (spcmDll, "spcm_dwSetParam_i32")
    spcm_dwSetParam_i32.argtype = [drv_handle, int32, int32]
    spcm_dwSetParam_i32.restype = uint32

    # load spcm_dwSetParam_i64
    spcm_dwSetParam_i64 = getattr (spcmDll, "spcm_dwSetParam_i64")
    spcm_dwSetParam_i64.argtype = [drv_handle, int32, int64]
    spcm_dwSetParam_i64.restype = uint32

    # load spcm_dwSetParam_i64m
    spcm_dwSetParam_i64m = getattr (spcmDll, "spcm_dwSetParam_i64m")
    spcm_dwSetParam_i64m.argtype = [drv_handle, int32, int32, int32]
    spcm_dwSetParam_i64m.restype = uint32

    # load spcm_dwDefTransfer_i64
    spcm_dwDefTransfer_i64 = getattr (spcmDll, "spcm_dwDefTransfer_i64")
    spcm_dwDefTransfer_i64.argtype = [drv_handle, uint32, uint32, uint32, c_void_p, uint64, uint64]
    spcm_dwDefTransfer_i64.restype = uint32

    # load spcm_dwInvalidateBuf
    spcm_dwInvalidateBuf = getattr (spcmDll, "spcm_dwInvalidateBuf")
    spcm_dwInvalidateBuf.argtype = [drv_handle, uint32]
    spcm_dwInvalidateBuf.restype = uint32

    # load spcm_dwGetContBuf_i64
    spcm_dwGetContBuf_i64 = getattr (spcmDll, "spcm_dwGetContBuf_i64")
    spcm_dwGetContBuf_i64.argtype = [drv_handle, uint32, POINTER(c_void_p), uptr64]
    spcm_dwGetContBuf_i64.restype = uint32

else:
    raise Exception ('Operating system not supported by pySpcm')
