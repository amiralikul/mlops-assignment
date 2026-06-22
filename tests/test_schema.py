import unittest

from agent.schema import render_schema


class SchemaRenderingTests(unittest.TestCase):
    def test_render_schema_includes_bird_column_descriptions(self) -> None:
        schema = render_schema("financial")

        self.assertIn('"A14" INTEGER NOT NULL -- no. of entrepreneurs per 1000 inhabitants', schema)
        self.assertIn('"A15" INTEGER -- no. of committed crimes 1995', schema)
        self.assertIn('"A16" INTEGER NOT NULL -- no. of committed crimes 1996', schema)


if __name__ == "__main__":
    unittest.main()
