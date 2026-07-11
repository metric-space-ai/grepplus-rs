//! Minimal GGUF reader for EmbeddingGemma.
//!
//! This is adapted from candle-core's `quantized::gguf_file` parser, but kept
//! local so `greppy-embed-native` has no candle dependency.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::quant::GgmlDType;
use crate::{Error, Result};

const DEFAULT_ALIGNMENT: usize = 32;
const GGUF_MAX_STRING_LENGTH: usize = 1 << 30;
const GGUF_MAX_ARRAY_ELEMENTS: usize = 1 << 30;
const GGUF_MAX_TENSOR_DIMS: u32 = 4;
const GGUF_MAX_VALUE_DEPTH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read(reader: &mut Reader<'_>) -> Result<Self> {
        let magic = reader.u32()?;
        match magic {
            0x4655_4747 | 0x4747_5546 => {}
            _ => return Err(Error::InvalidGguf(format!("unknown magic 0x{magic:08x}"))),
        }
        let version = reader.u32()?;
        match version {
            1 => Ok(Self::GgufV1),
            2 => Ok(Self::GgufV2),
            3 => Ok(Self::GgufV3),
            _ => Err(Error::InvalidGguf(format!(
                "unsupported GGUF version {version}"
            ))),
        }
    }

    fn length_prefix_size(self) -> usize {
        match self {
            Self::GgufV1 => 4,
            Self::GgufV2 | Self::GgufV3 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    String,
    Array,
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(Error::InvalidGguf(format!("unknown value type {v}"))),
        }
    }

    fn min_disk_size(self, magic: VersionedMagic) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String => magic.length_prefix_size(),
            Self::Array => 4 + magic.length_prefix_size(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(u64::from(*v)),
            Self::U16(v) => Some(u64::from(*v)),
            Self::U32(v) => Some(u64::from(*v)),
            Self::U64(v) => Some(*v),
            Self::Bool(v) => Some(u64::from(*v)),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|v| u32::try_from(v).ok())
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    fn read(
        reader: &mut Reader<'_>,
        value_type: ValueType,
        magic: VersionedMagic,
        depth: usize,
    ) -> Result<Self> {
        if depth > GGUF_MAX_VALUE_DEPTH {
            return Err(Error::InvalidGguf(format!(
                "value nesting depth exceeds max {GGUF_MAX_VALUE_DEPTH}"
            )));
        }
        match value_type {
            ValueType::U8 => Ok(Self::U8(reader.u8()?)),
            ValueType::I8 => Ok(Self::I8(reader.i8()?)),
            ValueType::U16 => Ok(Self::U16(reader.u16()?)),
            ValueType::I16 => Ok(Self::I16(reader.i16()?)),
            ValueType::U32 => Ok(Self::U32(reader.u32()?)),
            ValueType::I32 => Ok(Self::I32(reader.i32()?)),
            ValueType::U64 => Ok(Self::U64(reader.u64()?)),
            ValueType::I64 => Ok(Self::I64(reader.i64()?)),
            ValueType::F32 => Ok(Self::F32(reader.f32()?)),
            ValueType::F64 => Ok(Self::F64(reader.f64()?)),
            ValueType::Bool => match reader.u8()? {
                0 => Ok(Self::Bool(false)),
                1 => Ok(Self::Bool(true)),
                b => Err(Error::InvalidGguf(format!("unexpected bool value {b}"))),
            },
            ValueType::String => Ok(Self::String(read_string(reader, magic)?)),
            ValueType::Array => {
                let elem_type = ValueType::from_u32(reader.u32()?)?;
                let len = read_length(reader, magic)?;
                if len > GGUF_MAX_ARRAY_ELEMENTS {
                    return Err(Error::InvalidGguf(format!(
                        "array length {len} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
                    )));
                }
                let needed = len.saturating_mul(elem_type.min_disk_size(magic));
                if needed > reader.remaining() {
                    return Err(Error::InvalidGguf(format!(
                        "array of {len} elements needs at least {needed} bytes, only {} remain",
                        reader.remaining()
                    )));
                }
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(Self::read(reader, elem_type, magic, depth + 1)?);
                }
                Ok(Self::Array(values))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: GgmlDType,
    pub shape: Vec<usize>,
    pub offset: usize,
}

impl TensorInfo {
    pub fn element_count(&self) -> usize {
        self.checked_element_count().unwrap_or(usize::MAX)
    }

    pub fn checked_element_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                Error::InvalidGguf(format!(
                    "tensor shape {:?} element count overflows",
                    self.shape
                ))
            })
        })
    }

    pub fn byte_len(&self) -> Result<usize> {
        let elems = self.checked_element_count()?;
        let block_size = self.dtype.block_size();
        if elems % block_size != 0 {
            return Err(Error::InvalidGguf(format!(
                "tensor with shape {:?} has {elems} elements not divisible by {} for {}",
                self.shape, block_size, self.dtype
            )));
        }
        (elems / block_size)
            .checked_mul(self.dtype.type_size())
            .ok_or_else(|| {
                Error::InvalidGguf(format!(
                    "tensor with shape {:?} byte length overflows",
                    self.shape
                ))
            })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub dtype: GgmlDType,
    pub shape: &'a [usize],
    pub raw_bytes: &'a [u8],
}

impl TensorView<'_> {
    pub fn element_count(&self) -> usize {
        self.checked_element_count().unwrap_or(usize::MAX)
    }

    pub fn checked_element_count(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                Error::InvalidGguf(format!(
                    "tensor shape {:?} element count overflows",
                    self.shape
                ))
            })
        })
    }

    pub fn to_f32(&self) -> Result<Vec<f32>> {
        crate::quant::dequantize(self.dtype, self.raw_bytes, self.checked_element_count()?)
    }
}

pub struct GgufModel {
    _mmap: Mmap,
    metadata: HashMap<String, Value>,
    tensors: HashMap<String, TensorInfo>,
    tensor_data_offset: usize,
    magic: VersionedMagic,
}

impl GgufModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: the map is read-only. Model files in the HF cache are
        // content-addressed and treated as immutable while loaded.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: Mmap) -> Result<Self> {
        let mut reader = Reader::new(&mmap);
        let magic = VersionedMagic::read(&mut reader)?;
        let tensor_count = read_length(&mut reader, magic)?;
        let metadata_count = read_length(&mut reader, magic)?;
        if tensor_count > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(Error::InvalidGguf(format!(
                "tensor_count {tensor_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
            )));
        }
        if metadata_count > GGUF_MAX_ARRAY_ELEMENTS {
            return Err(Error::InvalidGguf(format!(
                "metadata_count {metadata_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}"
            )));
        }
        let prefix = magic.length_prefix_size();
        let needed = metadata_count
            .saturating_mul(prefix + 4 + 1)
            .saturating_add(tensor_count.saturating_mul(prefix + 4 + 4 + 8));
        if needed > reader.remaining() {
            return Err(Error::InvalidGguf(format!(
                "header declares {tensor_count} tensors and {metadata_count} metadata entries, \
                 needs at least {needed} bytes, only {} remain",
                reader.remaining()
            )));
        }

        let mut metadata = HashMap::new();
        metadata.try_reserve(metadata_count).map_err(|e| {
            Error::InvalidGguf(format!(
                "metadata table allocation for {metadata_count} entries failed: {e}"
            ))
        })?;
        for _ in 0..metadata_count {
            let key = read_string(&mut reader, magic)?;
            let value_type = ValueType::from_u32(reader.u32()?)?;
            let value = Value::read(&mut reader, value_type, magic, 0)?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::new();
        tensors.try_reserve(tensor_count).map_err(|e| {
            Error::InvalidGguf(format!(
                "tensor table allocation for {tensor_count} entries failed: {e}"
            ))
        })?;
        for _ in 0..tensor_count {
            let name = read_string(&mut reader, magic)?;
            let n_dims = reader.u32()?;
            if n_dims > GGUF_MAX_TENSOR_DIMS {
                return Err(Error::InvalidGguf(format!(
                    "tensor '{name}' has {n_dims} dimensions, max is {GGUF_MAX_TENSOR_DIMS}"
                )));
            }
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(match magic {
                    VersionedMagic::GgufV1 => usize::try_from(reader.u32()?).map_err(|_| {
                        Error::InvalidGguf(format!("tensor '{name}' dimension does not fit usize"))
                    })?,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.usize_u64()?,
                });
            }
            shape.reverse();
            let dtype = GgmlDType::from_u32(reader.u32()?)?;
            let offset = reader.usize_u64()?;
            tensors.insert(
                name,
                TensorInfo {
                    dtype,
                    shape,
                    offset,
                },
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 {
            return Err(Error::InvalidGguf(
                "general.alignment must be non-zero".to_string(),
            ));
        }
        let tensor_data_offset = reader.position().div_ceil(alignment) * alignment;
        if tensor_data_offset > mmap.len() {
            return Err(Error::InvalidGguf(format!(
                "tensor data offset {tensor_data_offset} exceeds file size {}",
                mmap.len()
            )));
        }

        let model = Self {
            _mmap: mmap,
            metadata,
            tensors,
            tensor_data_offset,
            magic,
        };
        for (name, info) in &model.tensors {
            model.tensor_bytes(name, info)?;
        }
        Ok(model)
    }

    pub fn magic(&self) -> VersionedMagic {
        self.magic
    }

    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }

    pub fn tensor_infos(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }

    pub fn tensor_data_offset(&self) -> usize {
        self.tensor_data_offset
    }

    pub fn file_len(&self) -> usize {
        self._mmap.len()
    }

    pub fn metadata_str(&self, key: &str) -> Result<&str> {
        self.metadata
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| Error::InvalidGguf(format!("metadata key '{key}' is not a string")))
    }

    pub fn metadata_u32(&self, key: &str) -> Result<u32> {
        self.metadata
            .get(key)
            .and_then(Value::as_u32)
            .ok_or_else(|| Error::InvalidGguf(format!("metadata key '{key}' is not a u32")))
    }

    pub fn metadata_f32(&self, key: &str) -> Result<f32> {
        self.metadata
            .get(key)
            .and_then(Value::as_f32)
            .ok_or_else(|| Error::InvalidGguf(format!("metadata key '{key}' is not an f32")))
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        Ok(TensorView {
            dtype: info.dtype,
            shape: &info.shape,
            raw_bytes: self.tensor_bytes(name, info)?,
        })
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let tensor = self.tensor(name)?;
        Ok((tensor.shape.to_vec(), tensor.to_f32()?))
    }

    fn tensor_bytes(&self, name: &str, info: &TensorInfo) -> Result<&[u8]> {
        let len = info.byte_len()?;
        let start = self
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| Error::InvalidGguf(format!("tensor '{name}' offset overflows")))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::InvalidGguf(format!("tensor '{name}' byte length overflows")))?;
        if end > self._mmap.len() {
            return Err(Error::InvalidGguf(format!(
                "tensor '{name}' needs bytes [{start}, {end}), file size is {}",
                self._mmap.len()
            )));
        }
        Ok(&self._mmap[start..end])
    }
}

fn read_length(reader: &mut Reader<'_>, magic: VersionedMagic) -> Result<usize> {
    match magic {
        VersionedMagic::GgufV1 => usize::try_from(reader.u32()?)
            .map_err(|_| Error::InvalidGguf("GGUF v1 length does not fit usize".into())),
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.usize_u64(),
    }
}

fn read_string(reader: &mut Reader<'_>, magic: VersionedMagic) -> Result<String> {
    let len = read_length(reader, magic)?;
    if len > GGUF_MAX_STRING_LENGTH {
        return Err(Error::InvalidGguf(format!(
            "string length {len} exceeds max {GGUF_MAX_STRING_LENGTH}"
        )));
    }
    let bytes = reader.bytes(len)?;
    let trimmed = bytes
        .iter()
        .rposition(|&b| b != 0)
        .map(|idx| &bytes[..=idx])
        .unwrap_or_default();
    Ok(String::from_utf8_lossy(trimmed).into_owned())
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| Error::InvalidGguf("reader position overflow".to_string()))?;
        if end > self.data.len() {
            return Err(Error::InvalidGguf(format!(
                "read of {len} bytes at {} exceeds file size {}",
                self.pos,
                self.data.len()
            )));
        }
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let bytes = self.bytes(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(bytes);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn usize_u64(&mut self) -> Result<usize> {
        let value = self.u64()?;
        usize::try_from(value)
            .map_err(|_| Error::InvalidGguf(format!("u64 value {value} does not fit usize")))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }
}
