//! Structured-data editing: `data set` / `data ensure` for JSON, TOML and
//! YAML, format-preserving.
//!
//! JSON: a span tokenizer locates the VALUE bytes of a `$.a.b[2].c` path and
//! replaces exactly those bytes — every other byte (whitespace, key order,
//! comments in JSON5-ish files are out of scope) stays untouched.
//! TOML: `toml_edit` (lossless document model).
//! YAML: scalar values addressed by mapping path, replaced in-line; only
//! plain scalar targets are supported — anything else refuses rather than
//! reformatting the document.

use std::path::Path;

use crate::certificate::{Certificate, SelectorClass, SelectorEngine, Status};
use crate::txn::{PlannedOp, Snapshot};
use crate::verbs::{run_pipeline_public, single_refusal_certificate, VerbOptions};
use greppy_core::{Error, Result};

/// Path segment of a `$.a.b[2].c` style path.
#[derive(Debug, Clone, PartialEq)]
enum Seg {
    Key(String),
    Index(usize),
}

fn parse_path(path: &str) -> Result<Vec<Seg>> {
    let body = path
        .strip_prefix("$.")
        .or_else(|| path.strip_prefix('$'))
        .unwrap_or(path);
    let mut out = Vec::new();
    for raw in body.split('.') {
        if raw.is_empty() {
            continue;
        }
        let mut rest = raw;
        // key part before any [n]
        if let Some(idx) = rest.find('[') {
            let (key, brackets) = rest.split_at(idx);
            if !key.is_empty() {
                out.push(Seg::Key(key.to_string()));
            }
            rest = brackets;
            while let Some(inner) = rest.strip_prefix('[') {
                let end = inner
                    .find(']')
                    .ok_or_else(|| Error::Invalid(format!("unclosed index in path: {path}")))?;
                let n: usize = inner[..end]
                    .parse()
                    .map_err(|_| Error::Invalid(format!("non-numeric index in path: {path}")))?;
                out.push(Seg::Index(n));
                rest = &inner[end + 1..];
            }
        } else {
            out.push(Seg::Key(rest.to_string()));
        }
    }
    if out.is_empty() {
        return Err(Error::Invalid(format!("empty data path: {path}")));
    }
    Ok(out)
}

/// `greppy edit data set|ensure`.
pub fn data_set(
    workspace_root: &Path,
    file: &Path,
    path: &str,
    value_json: &str,
    ensure: bool,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    let segs = parse_path(path)?;
    // validate the new value is valid JSON scalar/structure
    let new_value: serde_json::Value = serde_json::from_str(value_json)
        .map_err(|e| Error::Invalid(format!("--value-json is not valid JSON: {e}")))?;

    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let replacement: Option<(usize, usize, String)> = match ext.as_str() {
        "json" => json_value_span(&snapshot.content, &segs)
            .map(|(s, e)| (s, e, value_json.trim().to_string())),
        "toml" => return toml_set(workspace_root, snapshot, &segs, &new_value, ensure, options),
        "yaml" | "yml" => {
            yaml_scalar_span(&snapshot.content, &segs).map(|(s, e)| (s, e, yaml_scalar(&new_value)))
        }
        _ => {
            return Ok(single_refusal_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::DataPath,
                SelectorClass::StructuredData,
                Status::NotFound,
                options,
            ))
        }
    };
    let Some((start, end, new_text)) = replacement else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::DataPath,
            SelectorClass::StructuredData,
            Status::NotFound,
            options,
        ));
    };
    let current = &snapshot.content[start..end];
    if current == new_text.as_bytes() {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::DataPath,
            SelectorClass::StructuredData,
            Status::AlreadySatisfied,
            options,
        ));
    }
    if ensure {
        // ensure semantics: only write when different; identical handled above
    }
    let ops = vec![PlannedOp {
        id: "data-set".into(),
        range: (start, end),
        replacement: new_text.into_bytes(),
    }];
    let cert = run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::DataPath,
        SelectorClass::StructuredData,
        None,
        options,
    )?;
    // postcondition: the result must still parse as the format
    if cert.status == Status::Applied {
        let live = std::fs::read(file).unwrap_or_default();
        let ok = match ext.as_str() {
            "json" => serde_json::from_slice::<serde_json::Value>(&live).is_ok(),
            _ => true,
        };
        if !ok {
            // this cannot normally happen (value validated), but the honest
            // reaction to an invalid result is loud failure, not silence
            return Err(Error::Invalid(
                "data set produced an unparsable document; file was published - restore from VCS"
                    .into(),
            ));
        }
    }
    Ok(cert)
}

fn toml_set(
    workspace_root: &Path,
    snapshot: Snapshot,
    segs: &[Seg],
    new_value: &serde_json::Value,
    _ensure: bool,
    options: &VerbOptions,
) -> Result<Certificate> {
    let text = String::from_utf8_lossy(&snapshot.content).into_owned();
    let mut doc: toml_edit::DocumentMut = text
        .parse()
        .map_err(|e| Error::Parse(format!("TOML parse: {e}")))?;
    {
        let mut item: &mut toml_edit::Item = doc.as_item_mut();
        for seg in &segs[..segs.len() - 1] {
            item = match seg {
                Seg::Key(k) => &mut item[k.as_str()],
                Seg::Index(i) => &mut item[*i],
            };
        }
        let last = &segs[segs.len() - 1];
        let target = match last {
            Seg::Key(k) => &mut item[k.as_str()],
            Seg::Index(i) => &mut item[*i],
        };
        let new_item = json_to_toml(new_value)?;
        if target.to_string().trim() == new_item.to_string().trim() {
            return Ok(single_refusal_certificate(
                workspace_root,
                &snapshot,
                SelectorEngine::DataPath,
                SelectorClass::StructuredData,
                Status::AlreadySatisfied,
                options,
            ));
        }
        *target = new_item;
    }
    let new_content = doc.to_string().into_bytes();
    let ops = vec![PlannedOp {
        id: "data-set".into(),
        range: (0, snapshot.content.len()),
        replacement: new_content,
    }];
    run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::DataPath,
        SelectorClass::StructuredData,
        None,
        options,
    )
}

fn json_to_toml(v: &serde_json::Value) -> Result<toml_edit::Item> {
    Ok(match v {
        serde_json::Value::String(s) => toml_edit::value(s.as_str()),
        serde_json::Value::Bool(b) => toml_edit::value(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                toml_edit::value(i)
            } else {
                toml_edit::value(n.as_f64().unwrap_or(0.0))
            }
        }
        _ => {
            return Err(Error::Invalid(
                "TOML data set supports scalar values only".into(),
            ))
        }
    })
}

fn yaml_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Very small YAML scalar addressing: follow mapping keys by indentation,
/// return the value span on the matched line. Sequences and block scalars
/// refuse (None).
fn yaml_scalar_span(content: &[u8], segs: &[Seg]) -> Option<(usize, usize)> {
    let text = std::str::from_utf8(content).ok()?;
    let mut depth = 0usize;
    let mut expected_indent = 0usize;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let stripped = line.trim_end_matches('\n');
        let indent = stripped.len() - stripped.trim_start().len();
        let body = stripped.trim_start();
        if !body.is_empty() && !body.starts_with('#') && indent == expected_indent {
            if let Some((key, rest)) = body.split_once(':') {
                let want = match &segs[depth] {
                    Seg::Key(k) => k.as_str(),
                    Seg::Index(_) => return None,
                };
                if key.trim() == want {
                    if depth + 1 == segs.len() {
                        let value = rest.trim();
                        if value.is_empty() || value.starts_with('|') || value.starts_with('>') {
                            return None;
                        }
                        let value_off = line.rfind(value)?;
                        return Some((offset + value_off, offset + value_off + value.len()));
                    }
                    depth += 1;
                    expected_indent = indent + 2;
                }
            }
        }
        offset += line.len();
    }
    None
}

/// Locate the byte span of the value at `segs` in raw JSON text.
fn json_value_span(content: &[u8], segs: &[Seg]) -> Option<(usize, usize)> {
    let text = std::str::from_utf8(content).ok()?;
    let mut pos = skip_ws(text, 0)?;
    for seg in segs {
        match seg {
            Seg::Key(key) => {
                if text.as_bytes().get(pos) != Some(&b'{') {
                    return None;
                }
                pos += 1;
                loop {
                    pos = skip_ws(text, pos)?;
                    if text.as_bytes().get(pos) == Some(&b'}') {
                        return None; // key not present
                    }
                    let (k, after_key) = parse_json_string(text, pos)?;
                    let colon = skip_ws(text, after_key)?;
                    if text.as_bytes().get(colon) != Some(&b':') {
                        return None;
                    }
                    let value_start = skip_ws(text, colon + 1)?;
                    let value_end = skip_json_value(text, value_start)?;
                    if k == *key {
                        pos = value_start;
                        break;
                    }
                    pos = skip_ws(text, value_end)?;
                    if text.as_bytes().get(pos) == Some(&b',') {
                        pos += 1;
                    } else {
                        return None;
                    }
                }
            }
            Seg::Index(n) => {
                if text.as_bytes().get(pos) != Some(&b'[') {
                    return None;
                }
                pos += 1;
                let mut i = 0usize;
                loop {
                    pos = skip_ws(text, pos)?;
                    if text.as_bytes().get(pos) == Some(&b']') {
                        return None;
                    }
                    let value_end = skip_json_value(text, pos)?;
                    if i == *n {
                        break;
                    }
                    pos = skip_ws(text, value_end)?;
                    if text.as_bytes().get(pos) == Some(&b',') {
                        pos += 1;
                        i += 1;
                    } else {
                        return None;
                    }
                }
            }
        }
    }
    let end = skip_json_value(text, pos)?;
    Some((pos, end))
}

fn skip_ws(text: &str, mut pos: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    (pos <= bytes.len()).then_some(pos)
}

fn parse_json_string(text: &str, pos: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    if bytes.get(pos) != Some(&b'"') {
        return None;
    }
    let mut i = pos + 1;
    let mut out = String::new();
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                out.push(text[i + 1..].chars().next()?);
                i += 2;
            }
            b'"' => return Some((out, i + 1)),
            _ => {
                let c = text[i..].chars().next()?;
                out.push(c);
                i += c.len_utf8();
            }
        }
    }
    None
}

fn skip_json_value(text: &str, pos: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    match bytes.get(pos)? {
        b'"' => parse_json_string(text, pos).map(|(_, end)| end),
        b'{' | b'[' => {
            let open = bytes[pos];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 0usize;
            let mut i = pos;
            while i < bytes.len() {
                match bytes[i] {
                    b'"' => {
                        let (_, end) = parse_json_string(text, i)?;
                        i = end;
                        continue;
                    }
                    b if b == open => depth += 1,
                    b if b == close => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            let mut i = pos;
            while i < bytes.len() && !b",}] \t\r\n".contains(&bytes[i]) {
                i += 1;
            }
            Some(i)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn json_set_preserves_formatting() {
        let dir = ws();
        let f = dir.path().join("config.json");
        std::fs::write(
            &f,
            b"{\n  \"server\": {\n    \"port\": 9000,\n    \"host\": \"x\"\n  }\n}\n",
        )
        .unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.server.port",
            "8080",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("\"port\": 8080"), "{out}");
        assert!(out.contains("\"host\": \"x\""), "{out}");
        // byte-preserving: same shape, only the value changed
        assert!(out.starts_with("{\n  \"server\": {\n"), "{out}");
    }

    #[test]
    fn json_missing_path_refuses() {
        let dir = ws();
        let f = dir.path().join("c.json");
        std::fs::write(&f, b"{\"a\": 1}\n").unwrap();
        let cert = data_set(dir.path(), &f, "$.b", "2", false, &VerbOptions::default()).unwrap();
        assert_eq!(cert.status, Status::NotFound);
    }

    #[test]
    fn json_array_index() {
        let dir = ws();
        let f = dir.path().join("c.json");
        std::fs::write(&f, b"{\"items\": [1, 2, 3]}\n").unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.items[1]",
            "99",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(std::fs::read_to_string(&f).unwrap().contains("[1, 99, 3]"));
    }

    #[test]
    fn ensure_is_idempotent() {
        let dir = ws();
        let f = dir.path().join("c.json");
        std::fs::write(&f, b"{\"port\": 8080}\n").unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.port",
            "8080",
            true,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
    }

    #[test]
    fn toml_set_scalar() {
        let dir = ws();
        let f = dir.path().join("Cargo.toml");
        std::fs::write(&f, b"[package]\nname = \"x\"\nversion = \"0.1.0\"\n").unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.package.version",
            "\"0.2.0\"",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("version = \"0.2.0\""), "{out}");
        assert!(out.contains("name = \"x\""), "{out}");
    }

    #[test]
    fn yaml_scalar_set() {
        let dir = ws();
        let f = dir.path().join("c.yaml");
        std::fs::write(&f, b"server:\n  port: 9000\n  host: x\n").unwrap();
        let cert = data_set(
            dir.path(),
            &f,
            "$.server.port",
            "8080",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("port: 8080"), "{out}");
        assert!(out.contains("host: x"), "{out}");
    }
}
