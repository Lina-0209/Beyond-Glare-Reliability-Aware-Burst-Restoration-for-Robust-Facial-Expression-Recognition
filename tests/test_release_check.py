"""Only explicitly registered aggregate plots may enter the release tree."""
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_release


class ReleaseCheckTests(unittest.TestCase):
    def check_tree(self, relative, content, registered=None, symlink=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if symlink:
                actual = root / "target.txt"
                actual.write_text("safe", encoding="utf-8")
                target.symlink_to(actual)
            else:
                target.write_bytes(content)
            with patch.object(check_release, "ROOT", root), patch.object(
                check_release, "DOCUMENTATION_FIGURES", registered or {}
            ), redirect_stdout(io.StringIO()) as output:
                status = check_release.main()
            return status, output.getvalue()

    def test_registered_unchanged_figure_passes(self):
        name = "docs/figures/aggregate.png"
        content = b"reviewed aggregate fixture"
        status, _ = self.check_tree(name, content, {name: hashlib.sha256(content).hexdigest()})
        self.assertEqual(status, 0)

    def test_modified_registered_figure_fails(self):
        name = "docs/figures/aggregate.png"
        status, output = self.check_tree(name, b"modified", {name: "0" * 64})
        self.assertEqual(status, 1)
        self.assertIn("differs from recorded provenance", output)

    def test_unregistered_image_fails(self):
        status, _ = self.check_tree("docs/figures/face.png", b"unregistered")
        self.assertEqual(status, 1)

    def test_registered_symlink_still_fails(self):
        name = "docs/figures/aggregate.png"
        status, output = self.check_tree(name, b"", {name: "0" * 64}, symlink=True)
        self.assertEqual(status, 1)
        self.assertIn("symlink", output)

    def test_private_absolute_path_still_fails(self):
        status, _ = self.check_tree("README.md", b"local path: /data/" + b"example/private")
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
