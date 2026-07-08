use std::ffi::{c_char, c_void, CStr};
use std::ptr::NonNull;
use std::sync::OnceLock;

use libloading::Library;

use crate::{Error, Result};

const CUDA_BACKEND_UNAVAILABLE: i32 = 20000;

#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    embed_native_has_cuda_dylib
))]
const CUDA_DYLIB_BLOB: &[u8] = include_bytes!(env!("GREPPY_EMBED_NATIVE_CUDA_DYLIB"));

#[cfg(not(all(
    any(target_os = "linux", target_os = "windows"),
    embed_native_has_cuda_dylib
)))]
const CUDA_DYLIB_BLOB: &[u8] = &[];

type GpCudaErrorString = unsafe extern "C" fn(i32) -> *const c_char;
type GpCudaInit = unsafe extern "C" fn(i32, *mut *mut c_void, *mut *mut c_void) -> i32;
type GpCudaDestroy = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type GpCudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type GpCudaFree = unsafe extern "C" fn(*mut c_void) -> i32;
type GpCudaMemcpyH2DAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> i32;
type GpCudaMemcpyD2HAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> i32;
type GpCudaMemsetAsync = unsafe extern "C" fn(*mut c_void, i32, usize, *mut c_void) -> i32;
type GpCudaStreamSync = unsafe extern "C" fn(*mut c_void) -> i32;
type GpCudaMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> i32;

type GpEmbedQ6k =
    unsafe extern "C" fn(*const c_void, *const u32, *mut f32, i32, i32, f32, *mut c_void) -> i32;
type GpRmsNorm =
    unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, i32, f32, *mut c_void) -> i32;
type GpRmsNormAdd = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    i32,
    i32,
    f32,
    *mut c_void,
) -> i32;
type GpRmsNormHeads = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    *mut c_void,
) -> i32;
type GpSplitHeads =
    unsafe extern "C" fn(*const f32, *mut f32, i32, i32, i32, i32, i32, *mut c_void) -> i32;
type GpRopeNeox =
    unsafe extern "C" fn(*const f32, *mut f32, i32, i32, i32, i32, f32, *mut c_void) -> i32;
type GpAttentionScores =
    unsafe extern "C" fn(*mut c_void, *const f32, *const f32, *mut f32, i32, i32, i32, i32) -> i32;
type GpSoftmaxMask =
    unsafe extern "C" fn(*mut f32, *const u32, i32, i32, i32, i32, f32, *mut c_void) -> i32;
type GpAttentionValues =
    unsafe extern "C" fn(*mut c_void, *const f32, *const f32, *mut f32, i32, i32, i32, i32) -> i32;
type GpMergeHeads =
    unsafe extern "C" fn(*const f32, *mut f32, i32, i32, i32, i32, *mut c_void) -> i32;
type GpGeglu = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, *mut c_void) -> i32;
type GpMeanPool =
    unsafe extern "C" fn(*const f32, *const u32, *mut f32, i32, i32, i32, *mut c_void) -> i32;
type GpL2Norm = unsafe extern "C" fn(*const f32, *mut f32, i32, i32, *mut c_void) -> i32;
type GpMmqMatmul = unsafe extern "C" fn(
    i32,
    *const c_void,
    *const f32,
    *mut f32,
    *mut c_void,
    *mut c_void,
    i64,
    i64,
    i64,
    i64,
    *mut c_void,
) -> i32;

struct CudaApi {
    _lib: Library,
    gp_cuda_error_string: GpCudaErrorString,
    gp_cuda_init: GpCudaInit,
    gp_cuda_destroy: GpCudaDestroy,
    gp_cuda_malloc: GpCudaMalloc,
    gp_cuda_free: GpCudaFree,
    gp_cuda_memcpy_h2d_async: GpCudaMemcpyH2DAsync,
    gp_cuda_memcpy_d2h_async: GpCudaMemcpyD2HAsync,
    gp_cuda_memset_async: GpCudaMemsetAsync,
    gp_cuda_stream_sync: GpCudaStreamSync,
    gp_cuda_mem_get_info: GpCudaMemGetInfo,
    gp_embed_q6k: GpEmbedQ6k,
    gp_rms_norm: GpRmsNorm,
    gp_rms_norm_add: GpRmsNormAdd,
    gp_rms_norm_heads: GpRmsNormHeads,
    gp_split_heads: GpSplitHeads,
    gp_rope_neox: GpRopeNeox,
    gp_attention_scores: GpAttentionScores,
    gp_softmax_mask: GpSoftmaxMask,
    gp_attention_values: GpAttentionValues,
    gp_merge_heads: GpMergeHeads,
    gp_geglu: GpGeglu,
    gp_mean_pool: GpMeanPool,
    gp_l2_norm: GpL2Norm,
    gp_mmq_matmul: GpMmqMatmul,
}

unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

static CUDA_API: OnceLock<std::result::Result<CudaApi, String>> = OnceLock::new();

fn cuda_api() -> Result<&'static CudaApi> {
    match CUDA_API.get_or_init(load_cuda_api) {
        Ok(api) => Ok(api),
        Err(err) => Err(Error::InvalidGguf(format!(
            "CUDA backend unavailable: {err}"
        ))),
    }
}

fn load_cuda_api() -> std::result::Result<CudaApi, String> {
    let lib = load_cuda_library()?;
    unsafe {
        Ok(CudaApi {
            gp_cuda_error_string: load_symbol(&lib, b"gp_cuda_error_string\0")?,
            gp_cuda_init: load_symbol(&lib, b"gp_cuda_init\0")?,
            gp_cuda_destroy: load_symbol(&lib, b"gp_cuda_destroy\0")?,
            gp_cuda_malloc: load_symbol(&lib, b"gp_cuda_malloc\0")?,
            gp_cuda_free: load_symbol(&lib, b"gp_cuda_free\0")?,
            gp_cuda_memcpy_h2d_async: load_symbol(&lib, b"gp_cuda_memcpy_h2d_async\0")?,
            gp_cuda_memcpy_d2h_async: load_symbol(&lib, b"gp_cuda_memcpy_d2h_async\0")?,
            gp_cuda_memset_async: load_symbol(&lib, b"gp_cuda_memset_async\0")?,
            gp_cuda_stream_sync: load_symbol(&lib, b"gp_cuda_stream_sync\0")?,
            gp_cuda_mem_get_info: load_symbol(&lib, b"gp_cuda_mem_get_info\0")?,
            gp_embed_q6k: load_symbol(&lib, b"gp_embed_q6k\0")?,
            gp_rms_norm: load_symbol(&lib, b"gp_rms_norm\0")?,
            gp_rms_norm_add: load_symbol(&lib, b"gp_rms_norm_add\0")?,
            gp_rms_norm_heads: load_symbol(&lib, b"gp_rms_norm_heads\0")?,
            gp_split_heads: load_symbol(&lib, b"gp_split_heads\0")?,
            gp_rope_neox: load_symbol(&lib, b"gp_rope_neox\0")?,
            gp_attention_scores: load_symbol(&lib, b"gp_attention_scores\0")?,
            gp_softmax_mask: load_symbol(&lib, b"gp_softmax_mask\0")?,
            gp_attention_values: load_symbol(&lib, b"gp_attention_values\0")?,
            gp_merge_heads: load_symbol(&lib, b"gp_merge_heads\0")?,
            gp_geglu: load_symbol(&lib, b"gp_geglu\0")?,
            gp_mean_pool: load_symbol(&lib, b"gp_mean_pool\0")?,
            gp_l2_norm: load_symbol(&lib, b"gp_l2_norm\0")?,
            gp_mmq_matmul: load_symbol(&lib, b"gp_mmq_matmul\0")?,
            _lib: lib,
        })
    }
}

fn load_cuda_library() -> std::result::Result<Library, String> {
    if let Ok(path) = std::env::var("EMBED_NATIVE_CUDA_LIBRARY") {
        let path = path.trim();
        if !path.is_empty() {
            return unsafe { Library::new(path) }
                .map_err(|e| format!("failed to load {path}: {e}"));
        }
    }

    if CUDA_DYLIB_BLOB.is_empty() {
        return Err("CUDA backend library was not built into this binary".into());
    }

    let ext = if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    };
    let path = std::env::temp_dir().join(format!(
        "greppy_embed_native_cuda_{}.{}",
        std::process::id(),
        ext
    ));
    std::fs::write(&path, CUDA_DYLIB_BLOB).map_err(|e| {
        format!(
            "failed to write bundled CUDA backend {}: {e}",
            path.display()
        )
    })?;
    unsafe { Library::new(&path) }.map_err(|e| {
        format!(
            "failed to load bundled CUDA backend {}: {e}",
            path.display()
        )
    })
}

unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> std::result::Result<T, String> {
    let symbol = unsafe { lib.get::<T>(name) }.map_err(|e| {
        format!(
            "missing CUDA backend symbol {}: {e}",
            String::from_utf8_lossy(name).trim_end_matches('\0')
        )
    })?;
    Ok(*symbol)
}

pub struct CudaDevice {
    stream: NonNull<c_void>,
    blas: NonNull<c_void>,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn new(device: i32) -> Result<Self> {
        let api = cuda_api()?;
        let mut stream = std::ptr::null_mut();
        let mut blas = std::ptr::null_mut();
        check(
            unsafe { (api.gp_cuda_init)(device, &mut stream, &mut blas) },
            "cuda init",
        )?;
        let stream = NonNull::new(stream)
            .ok_or_else(|| Error::InvalidGguf("CUDA stream creation returned null".into()))?;
        let blas = NonNull::new(blas)
            .ok_or_else(|| Error::InvalidGguf("cuBLAS creation returned null".into()))?;
        Ok(Self { stream, blas })
    }

    pub fn stream(&self) -> *mut c_void {
        self.stream.as_ptr()
    }

    pub fn blas(&self) -> *mut c_void {
        self.blas.as_ptr()
    }

    pub fn sync(&self) -> Result<()> {
        let api = cuda_api()?;
        check(
            unsafe { (api.gp_cuda_stream_sync)(self.stream()) },
            "cuda stream sync",
        )
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let api = cuda_api()?;
        let mut free = 0usize;
        let mut total = 0usize;
        check(
            unsafe { (api.gp_cuda_mem_get_info)(&mut free, &mut total) },
            "cuda mem info",
        )?;
        Ok((free, total))
    }

    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        let api = cuda_api()?;
        let mut ptr = std::ptr::null_mut();
        check(
            unsafe { (api.gp_cuda_malloc)(&mut ptr, bytes) },
            "cuda malloc",
        )?;
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            Error::InvalidGguf(format!("cuda malloc returned null ({bytes} bytes)"))
        })?;
        Ok(DeviceBuffer { ptr, bytes })
    }

    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<DeviceBuffer> {
        let buf = self.alloc(bytes.len().max(1))?;
        self.copy_h2d(&buf, bytes)?;
        Ok(buf)
    }

    pub fn copy_h2d<T>(&self, dst: &DeviceBuffer, src: &[T]) -> Result<()> {
        let api = cuda_api()?;
        let bytes = std::mem::size_of_val(src);
        if bytes > dst.bytes {
            return Err(Error::InvalidGguf(format!(
                "cuda h2d copy {bytes} bytes exceeds dst {} bytes",
                dst.bytes
            )));
        }
        check(
            unsafe {
                (api.gp_cuda_memcpy_h2d_async)(
                    dst.ptr(),
                    src.as_ptr() as *const c_void,
                    bytes,
                    self.stream(),
                )
            },
            "cuda memcpy h2d",
        )
    }

    pub fn copy_d2h<T>(&self, dst: &mut [T], src: &DeviceBuffer) -> Result<()> {
        let api = cuda_api()?;
        let bytes = std::mem::size_of_val(dst);
        if bytes > src.bytes {
            return Err(Error::InvalidGguf(format!(
                "cuda d2h copy {bytes} bytes exceeds src {} bytes",
                src.bytes
            )));
        }
        check(
            unsafe {
                (api.gp_cuda_memcpy_d2h_async)(
                    dst.as_mut_ptr() as *mut c_void,
                    src.ptr(),
                    bytes,
                    self.stream(),
                )
            },
            "cuda memcpy d2h",
        )?;
        self.sync()
    }

    pub fn memset(&self, dst: &DeviceBuffer, value: i32) -> Result<()> {
        let api = cuda_api()?;
        check(
            unsafe { (api.gp_cuda_memset_async)(dst.ptr(), value, dst.bytes, self.stream()) },
            "cuda memset",
        )
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if let Ok(api) = cuda_api() {
            let _ = unsafe { (api.gp_cuda_destroy)(self.stream(), self.blas()) };
        }
    }
}

pub struct DeviceBuffer {
    ptr: NonNull<c_void>,
    bytes: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    pub fn as_f32(&self) -> *mut f32 {
        self.ptr() as *mut f32
    }

    pub fn as_u32(&self) -> *mut u32 {
        self.ptr() as *mut u32
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Ok(api) = cuda_api() {
            let _ = unsafe { (api.gp_cuda_free)(self.ptr()) };
        }
    }
}

pub fn check(code: i32, what: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let msg = cuda_error_string(code);
    Err(Error::InvalidGguf(format!("{what}: {msg} ({code})")))
}

fn cuda_error_string(code: i32) -> String {
    if code == CUDA_BACKEND_UNAVAILABLE {
        return "CUDA backend unavailable".into();
    }
    let Ok(api) = cuda_api() else {
        return format!("CUDA error code {code}");
    };
    unsafe {
        let raw = (api.gp_cuda_error_string)(code);
        if raw.is_null() {
            format!("CUDA error code {code}")
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_embed_q6k(
    weights: *const c_void,
    ids: *const u32,
    dst: *mut f32,
    rows: i32,
    hidden: i32,
    scale: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_embed_q6k)(weights, ids, dst, rows, hidden, scale, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_rms_norm(
    src: *const f32,
    weight: *const f32,
    dst: *mut f32,
    rows: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_rms_norm)(src, weight, dst, rows, dim, eps, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_rms_norm_add(
    src: *const f32,
    add: *const f32,
    weight: *const f32,
    dst: *mut f32,
    rows: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_rms_norm_add)(src, add, weight, dst, rows, dim, eps, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_rms_norm_heads(
    src: *const f32,
    weight: *const f32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    row_width: i32,
    eps: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_rms_norm_heads)(
                src, weight, dst, batch, seq, heads, head_dim, row_width, eps, stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_split_heads(
    src: *const f32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    row_width: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_split_heads)(src, dst, batch, seq, heads, head_dim, row_width, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_rope_neox(
    src: *const f32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    base_freq: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_rope_neox)(src, dst, batch, seq, heads, head_dim, base_freq, stream)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_attention_scores(
    blas: *mut c_void,
    q: *const f32,
    k: *const f32,
    scores: *mut f32,
    batch: i32,
    heads: i32,
    seq: i32,
    head_dim: i32,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_attention_scores)(blas, q, k, scores, batch, heads, seq, head_dim)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_softmax_mask(
    scores: *mut f32,
    mask: *const u32,
    batch: i32,
    heads: i32,
    seq: i32,
    sliding_window: i32,
    scale: f32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_softmax_mask)(
                scores,
                mask,
                batch,
                heads,
                seq,
                sliding_window,
                scale,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_attention_values(
    blas: *mut c_void,
    scores: *const f32,
    v: *const f32,
    out: *mut f32,
    batch: i32,
    heads: i32,
    seq: i32,
    head_dim: i32,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_attention_values)(blas, scores, v, out, batch, heads, seq, head_dim)
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_merge_heads(
    src: *const f32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_merge_heads)(src, dst, batch, seq, heads, head_dim, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_geglu(
    gate: *const f32,
    up: *const f32,
    dst: *mut f32,
    total: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_geglu)(gate, up, dst, total, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_mean_pool(
    hidden: *const f32,
    mask: *const u32,
    dst: *mut f32,
    batch: i32,
    seq: i32,
    hidden_dim: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_mean_pool)(hidden, mask, dst, batch, seq, hidden_dim, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

pub unsafe fn gp_l2_norm(
    src: *const f32,
    dst: *mut f32,
    rows: i32,
    dim: i32,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.gp_l2_norm)(src, dst, rows, dim, stream) },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gp_mmq_matmul(
    dtype: i32,
    weights: *const c_void,
    src: *const f32,
    dst: *mut f32,
    q8_scratch: *mut c_void,
    fixup_scratch: *mut c_void,
    ncols_x: i64,
    stride_row_x: i64,
    nrows_x: i64,
    ncols_dst: i64,
    stream: *mut c_void,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.gp_mmq_matmul)(
                dtype,
                weights,
                src,
                dst,
                q8_scratch,
                fixup_scratch,
                ncols_x,
                stride_row_x,
                nrows_x,
                ncols_dst,
                stream,
            )
        },
        Err(_) => CUDA_BACKEND_UNAVAILABLE,
    }
}
