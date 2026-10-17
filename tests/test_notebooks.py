"""Prevent sensitive notebook results from entering the source tree."""
import json
from pathlib import Path
import unittest


class NotebookSourceTests(unittest.TestCase):
    def test_research_notebooks_contain_no_saved_outputs_or_attachments(self):
        directory = Path(__file__).resolve().parents[1] / "notebooks"
        for name in ("01_eda.ipynb", "02_detection.ipynb", "03_evaluation.ipynb"):
            with self.subTest(notebook=name):
                notebook = json.loads((directory / name).read_text(encoding="utf-8"))
                for cell in notebook["cells"]:
                    self.assertFalse(cell.get("attachments"), name)
                    self.assertFalse(cell.get("outputs"), name)
                    self.assertIsNone(cell.get("execution_count"), name)
                    self.assertNotIn("data:image/", "".join(cell["source"]), name)
