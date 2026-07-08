//! Source-span heuristics shared by the CLI (`--code` printing) and the
//! embedding indexer (what text a code-span vector is computed from).
//!
//! The indexer stores a definition's `start_line`/`end_line` as the
//! **declaration line** of the symbol (`end_line == start_line`), because
//! the captured tree-sitter node is the name identifier, not the full item.
//! Anything that wants the definition's BODY — printing source under a
//! nav result, or embedding the span so semantic search can match the
//! body's behaviour, not just the signature — must extend the span
//! forward. This module holds that shared extension logic.

/// Scan forward from `start_idx` (0-based index into `lines`) and return
/// the 0-based index of the line on which the definition ends, by
/// balancing curly braces.
///
/// We balance ONLY curly braces `{}`: a function's parameter list `( … )`
/// closes on the signature line and would otherwise look like the body
/// ending. String/char literals and `//` line comments are skipped so a
/// brace inside them cannot skew the depth. A `;` before any `{` opens
/// terminates a block-less item (unit struct, type alias, bare
/// signature). If no clean close is found within the scan window, the
/// window end is returned when a block was opened (emit *something*
/// bounded), else the start line.
pub fn definition_end_idx(lines: &[&str], start_idx: usize) -> usize {
    /// Upper bound on how far we scan forward for a definition's end, so
    /// a missing closing delimiter cannot make us read an entire huge
    /// file. Callers apply their own tighter print/embed caps on top.
    const MAX_SCAN_LINES: usize = 400;

    let last = lines.len().saturating_sub(1);
    let scan_end = std::cmp::min(last, start_idx + MAX_SCAN_LINES);
    let mut depth: i32 = 0;
    let mut opened = false;
    for (idx, raw) in lines.iter().enumerate().take(scan_end + 1).skip(start_idx) {
        let mut in_str: Option<char> = None;
        let mut prev = '\0';
        let bytes: Vec<char> = raw.chars().collect();
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];
            if let Some(q) = in_str {
                // Inside a string/char literal: only an unescaped matching
                // quote closes it.
                if c == q && prev != '\\' {
                    in_str = None;
                }
                prev = c;
                i += 1;
                continue;
            }
            // A `//` line comment: ignore the rest of the line.
            if c == '/' && i + 1 < bytes.len() && bytes[i + 1] == '/' {
                break;
            }
            match c {
                '"' | '\'' => in_str = Some(c),
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' if opened => {
                    depth -= 1;
                    if depth <= 0 {
                        return idx;
                    }
                }
                ';' if !opened => {
                    // A `;`-terminated item with no block (unit struct,
                    // type alias, bare signature): this line is the end.
                    return idx;
                }
                _ => {}
            }
            prev = c;
            i += 1;
        }
    }
    // No clean close found within the window: fall back to the start line
    // (single line) so we never over-read, but at least emit something.
    if opened {
        scan_end
    } else {
        start_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(src: &str) -> usize {
        let lines: Vec<&str> = src.lines().collect();
        definition_end_idx(&lines, 0)
    }

    #[test]
    fn balances_simple_block() {
        assert_eq!(idx("fn f() {\n    body();\n}\nfn g() {}"), 2);
    }

    #[test]
    fn semicolon_item_ends_on_its_line() {
        assert_eq!(idx("type A = B;\nfn f() {}"), 0);
    }

    #[test]
    fn brace_in_string_is_ignored() {
        assert_eq!(idx("fn f() {\n    let s = \"}\";\n    done();\n}"), 3);
    }

    #[test]
    fn brace_in_line_comment_is_ignored() {
        assert_eq!(idx("fn f() {\n    // } not the end\n    done();\n}"), 3);
    }
}
