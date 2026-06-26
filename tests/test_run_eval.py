import unittest
from unittest.mock import Mock, patch

from evals import run_eval


class EvalRunnerTests(unittest.TestCase):
    def test_eval_one_scores_final_and_iteration_sql(self) -> None:
        question = {
            "question": "How many rows?",
            "db_id": "toy",
            "gold_sql": "SELECT 1",
        }
        response = {
            "sql": "SELECT 1",
            "ok": True,
            "iterations": 2,
            "history": [
                {"node": "generate_sql", "sql": "SELECT 0"},
                {"node": "verify", "ok": False, "issue": "wrong"},
                {"node": "revise", "sql": "SELECT 1"},
                {"node": "verify", "ok": True, "issue": ""},
            ],
        }

        with (
            patch("evals.run_eval.httpx.post") as post,
            patch("evals.run_eval.run_sql") as run_sql,
        ):
            post.return_value = Mock(
                json=Mock(return_value=response),
                raise_for_status=Mock(),
            )
            run_sql.side_effect = [
                (True, [(1,)], None),
                (True, [(1,)], None),
                (True, [(0,)], None),
                (True, [(1,)], None),
            ]

            result = run_eval.eval_one(question, "http://agent/answer")

        post.assert_called_once_with(
            "http://agent/answer",
            json={
                "question": "How many rows?",
                "db": "toy",
                "tags": {"run_type": "eval", "db": "toy"},
            },
            timeout=120.0,
        )
        self.assertTrue(result["correct"])
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(
            [attempt["correct"] for attempt in result["attempts"]],
            [False, True],
        )

    def test_eval_one_returns_error_record_when_agent_request_fails(self) -> None:
        question = {
            "question": "How many rows?",
            "db_id": "toy",
            "gold_sql": "SELECT 1",
        }

        with patch("evals.run_eval.httpx.post", side_effect=httpx_error("boom")):
            result = run_eval.eval_one(question, "http://agent/answer")

        self.assertFalse(result["correct"])
        self.assertEqual(result["error"], "RuntimeError: boom")
        self.assertEqual(result["attempts"], [])

    def test_summarize_carries_forward_stopped_iterations(self) -> None:
        results = [
            {
                "correct": True,
                "iterations": 1,
                "attempts": [{"iteration": 1, "correct": True}],
            },
            {
                "correct": True,
                "iterations": 2,
                "attempts": [
                    {"iteration": 1, "correct": False},
                    {"iteration": 2, "correct": True},
                ],
            },
            {
                "correct": False,
                "iterations": 3,
                "attempts": [
                    {"iteration": 1, "correct": False},
                    {"iteration": 2, "correct": False},
                    {"iteration": 3, "correct": False},
                ],
            },
        ]

        summary = run_eval.summarize(results)

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["correct"], 2)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["per_iteration"]["1"]["correct"], 1)
        self.assertEqual(summary["per_iteration"]["2"]["correct"], 2)
        self.assertEqual(summary["per_iteration"]["3"]["correct"], 2)


def httpx_error(message: str) -> RuntimeError:
    return RuntimeError(message)


if __name__ == "__main__":
    unittest.main()
