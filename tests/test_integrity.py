"""Integrity checks across checkout line endings and actual content changes."""
from pathlib import Path
import hashlib
import importlib.util
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("release_integrity", SCRIPTS / "verify_integrity.py")
integrity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integrity)


class ReleaseIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record(self, payload, normalization):
        canonical = payload.replace(b"\r\n", b"\n") if normalization == "lf" else payload
        return {"bytes": len(canonical), "sha256": hashlib.sha256(canonical).hexdigest(),
                "normalization": normalization}

    def test_text_accepts_lf_crlf_and_mixed_endings(self):
        expected = {"sample.txt": self.record(b"alpha\nbeta\n", "lf")}
        for payload in [b"alpha\nbeta\n", b"alpha\r\nbeta\r\n", b"alpha\r\nbeta\n"]:
            with self.subTest(payload=payload):
                (self.root / "sample.txt").write_bytes(payload)
                self.assertEqual(integrity.verify_files(self.root, expected), [])

    def test_text_rejects_changed_values_and_whitespace(self):
        expected = {"sample.txt": self.record(b"alpha\nbeta\n", "lf")}
        for payload in [b"alpha\nbetX\n", b"alpha \nbeta\n", b"alpha\nbeta", b"alpha\rbeta\r"]:
            with self.subTest(payload=payload):
                (self.root / "sample.txt").write_bytes(payload)
                self.assertTrue(integrity.verify_files(self.root, expected))

    def test_binary_is_checked_byte_for_byte(self):
        payload = b"\x00\xff\r\nDATA"
        expected = {"sample.npz": self.record(payload, "none")}
        path = self.root / "sample.npz"
        path.write_bytes(payload)
        self.assertEqual(integrity.verify_files(self.root, expected), [])
        path.write_bytes(payload.replace(b"\r\n", b"\n"))
        self.assertTrue(integrity.verify_files(self.root, expected))

    def test_missing_and_outside_paths_are_rejected(self):
        record = self.record(b"x", "none")
        for name in ["missing.txt", "../outside.txt"]:
            self.assertTrue(integrity.verify_files(self.root, {name: record}))

    def test_unknown_normalization_is_rejected(self):
        (self.root / "sample.txt").write_bytes(b"x")
        record = self.record(b"x", "none")
        record["normalization"] = "unsupported"
        self.assertTrue(integrity.verify_files(self.root, {"sample.txt": record}))


if __name__ == "__main__":
    unittest.main()
