import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.execution import ExecutionResult
from agent.graph import (
    MAX_ITERATIONS,
    AgentState,
    _parse_verify_reply,
    revise_node,
    route_after_verify,
    verify_node,
)
from agent.server import build_trace_metadata


class GraphNodeTests(unittest.TestCase):
    def test_build_trace_metadata_adds_report_defaults_and_preserves_request_tags(
        self,
    ) -> None:
        metadata = build_trace_metadata({"run_type": "eval", "db": "financial"})

        self.assertEqual(metadata["backend"], "vllm")
        self.assertEqual(metadata["model"], "Qwen/Qwen3-30B-A3B-Instruct-2507")
        self.assertEqual(metadata["run_type"], "eval")
        self.assertEqual(metadata["tuning_iteration"], "baseline")
        self.assertEqual(metadata["db"], "financial")

    def test_parse_verify_reply_extracts_fenced_json(self) -> None:
        ok, issue = _parse_verify_reply(
            '```json\n{"ok": false, "issue": "SQL errored"}\n```'
        )

        self.assertFalse(ok)
        self.assertEqual(issue, "SQL errored")

    def test_parse_verify_reply_accepts_prose_wrapped_json(self) -> None:
        ok, issue = _parse_verify_reply(
            'The answer is questionable: {"ok": true, "issue": ""}'
        )

        self.assertTrue(ok)
        self.assertEqual(issue, "")

    def test_parse_verify_reply_defaults_to_revision_on_invalid_json(self) -> None:
        ok, issue = _parse_verify_reply("looks fine to me")

        self.assertFalse(ok)
        self.assertIn("Could not parse verifier JSON", issue)

    def test_route_after_verify_ends_when_ok_or_at_iteration_cap(self) -> None:
        self.assertEqual(
            route_after_verify(AgentState("q", "db", verify_ok=True)), "end"
        )
        self.assertEqual(
            route_after_verify(
                AgentState("q", "db", verify_ok=False, iteration=MAX_ITERATIONS)
            ),
            "end",
        )

    def test_route_after_verify_revises_when_not_ok_under_cap(self) -> None:
        self.assertEqual(
            route_after_verify(AgentState("q", "db", verify_ok=False, iteration=1)),
            "revise",
        )

    def test_revise_node_returns_clean_sql_and_records_history(self) -> None:
        state = AgentState(
            question="Which circuits hosted Australian Grand Prix?",
            db_id="formula_1",
            schema='CREATE TABLE "races" ("name" TEXT);',
            sql="SELECT * FROM missing_table",
            execution=ExecutionResult(ok=False, error="no such table: missing_table"),
            verify_issue="SQL references a table that is not in the schema.",
            iteration=1,
            history=[{"node": "generate_sql", "sql": "SELECT * FROM missing_table"}],
        )

        with patch("agent.graph.llm") as llm_factory:
            llm_factory.return_value.invoke.return_value = SimpleNamespace(
                content='```sql\nSELECT "name" FROM "races";\n```'
            )

            result = revise_node(state)

        self.assertEqual(result["sql"], 'SELECT "name" FROM "races";')
        self.assertEqual(result["iteration"], 2)
        self.assertEqual(result["history"][-1]["node"], "revise")
        self.assertEqual(result["history"][-1]["sql"], 'SELECT "name" FROM "races";')

    def test_verify_node_rejects_repeated_rows_without_calling_llm(self) -> None:
        state = AgentState(
            question="What is the coordinates location of the circuits for Australian grand prix?",
            db_id="formula_1",
            schema='CREATE TABLE "circuits" ("lat" REAL, "lng" REAL);',
            sql='SELECT c."lat", c."lng" FROM "circuits" c;',
            execution=ExecutionResult(
                ok=True,
                rows=[(-34.9272, 138.617), (-34.9272, 138.617)],
                columns=["lat", "lng"],
                row_count=2,
            ),
            history=[
                {
                    "node": "generate_sql",
                    "sql": 'SELECT c."lat", c."lng" FROM "circuits" c;',
                }
            ],
        )

        with patch("agent.graph.llm") as llm_factory:
            result = verify_node(state)

        llm_factory.assert_not_called()
        self.assertFalse(result["verify_ok"])
        self.assertIn("repeated identical rows", result["verify_issue"])
        self.assertEqual(result["history"][-1]["node"], "verify")
        self.assertFalse(result["history"][-1]["ok"])


if __name__ == "__main__":
    unittest.main()
