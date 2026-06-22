"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You generate SQLite SQL for an analytics assistant.

Rules:
- Use only tables and columns from the provided schema.
- Return exactly one read-only SQLite SELECT query.
- Do not include markdown fences, prose, comments, or explanations.
- Prefer explicit JOINs using the schema's foreign-key relationships.
- Quote identifiers with double quotes when they contain spaces, punctuation,
  mixed case, or could be reserved words.
- Use DISTINCT when a join can repeat the same entity or attribute and the
  question asks for the value/list rather than every matching event row.
- If the question asks for a top/bottom/list subset, include the appropriate
  ORDER BY and LIMIT.
"""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Database schema:
{schema}

Question:
{question}

Write the SQL query."""


VERIFY_SYSTEM = """You verify whether a SQLite query result plausibly answers a
natural-language question.

Return only JSON with this exact shape:
{{"ok": true, "issue": ""}}

Use ok=false when:
- the SQL execution errored,
- the query references the wrong table or column,
- the result columns do not answer the question,
- the result is empty and the question likely expects matching rows,
- the result contains repeated identical rows where the question asks for a
  unique value/list,
- the SQL ignored an obvious filter, aggregation, ordering, or limit.

Keep issue short and actionable."""

VERIFY_USER = """Database schema:
{schema}

Question:
{question}

SQL:
{sql}

Execution result:
{execution}

Is the SQL result a plausible answer?"""


REVISE_SYSTEM = """You revise failed SQLite SQL.

Rules:
- Use only tables and columns from the provided schema.
- Return exactly one read-only SQLite SELECT query.
- Do not include markdown fences, prose, comments, or explanations.
- Address the verifier issue directly.
"""

REVISE_USER = """Database schema:
{schema}

Question:
{question}

Previous SQL:
{sql}

Previous execution result:
{execution}

Verifier issue:
{issue}

Write a corrected SQL query."""
