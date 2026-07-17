Before answering or editing, construct the task's source-grounded code
context; when editing, change exactly that context and nothing else.

1. Locate the definitions that implement the named or described behavior
   and read their exact source.

2. From these definitions, follow every caller, callee, usage, type,
   import, configuration, test, or other dependency that could affect the
   task. Read the exact source of every relevant node and continue until no
   relevant relation remains unresolved.

3. Use search results, summaries, symbols, and graph relations only to
   locate source. If one navigation method is incomplete or ambiguous, use
   another; an empty result does not prove that no relation exists.

4. When you change code, address the change to the exact definitions and
   spans you identified, keep every change inside that addressed context,
   and verify the result by evidence — a verified edit result, the build,
   or the tests — rather than by re-reading files you already hold in
   context.

Exclude code that cannot affect the task. Once the relevant graph is
complete, stop exploring and answer or edit in one pass.
