import unittest

from agent.schema import render_schema


class SchemaRenderingTests(unittest.TestCase):
    def test_render_schema_includes_bird_column_descriptions(self) -> None:
        schema = render_schema("financial")

        self.assertIn('"A14" INTEGER NOT NULL -- no. of entrepreneurs per 1000 inhabitants', schema)
        self.assertIn('"A15" INTEGER -- no. of committed crimes 1995', schema)
        self.assertIn('"A16" INTEGER NOT NULL -- no. of committed crimes 1996', schema)

    def test_render_schema_ignores_null_bird_column_descriptions(self) -> None:
        for db_id in ("debit_card_specializing", "european_football_2"):
            with self.subTest(db_id=db_id):
                schema = render_schema(db_id)

                self.assertIn(f"-- Database: {db_id}", schema)
                self.assertNotIn("-- None", schema)


if __name__ == "__main__":
    unittest.main()
