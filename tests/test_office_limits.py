"""Resource-cap guards for parsing untrusted office/PDF files (F2).

.docx/.xlsx are zip+XML containers; a few-KB zip-bomb can decompress to
gigabytes and OOM-kill the process during a corpus scan. These tests verify the
pre-parse screen rejects bombs before openpyxl/python-docx ever decompress them.
"""
import zipfile
from pathlib import Path

import pytest

from graphify import detect


def _write_zip(path, name, payload):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, payload)


def test_file_within_size_cap(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 1024)
    assert detect._file_within_size_cap(f) is True          # within default cap
    assert detect._file_within_size_cap(f, cap=512) is False  # over an explicit small cap
    assert detect._file_within_size_cap(tmp_path / "missing") is False


def test_zip_ratio_bomb_rejected(tmp_path):
    """A tiny file that expands far past the ratio threshold is rejected."""
    bomb = tmp_path / "bomb.xlsx"
    _write_zip(bomb, "xl/worksheets/sheet1.xml", b"0" * (5 * 1024 * 1024))  # 5 MiB of zeros -> tiny zip
    assert bomb.stat().st_size < 100 * 1024  # compressed to well under 100 KiB
    assert detect._zip_within_caps(bomb) is False


def test_legit_zip_passes(tmp_path):
    ok = tmp_path / "ok.docx"
    _write_zip(ok, "word/document.xml", b"<xml>hello world</xml>" * 20)
    assert detect._zip_within_caps(ok) is True


def test_non_zip_rejected(tmp_path):
    notzip = tmp_path / "fake.xlsx"
    notzip.write_bytes(b"this is not a zip file")
    assert detect._zip_within_caps(notzip) is False


def test_converters_return_empty_for_bomb(tmp_path):
    """The live converters bail out (return "") on a bomb before parsing."""
    for ext in (".docx", ".xlsx"):
        bomb = tmp_path / f"bomb{ext}"
        _write_zip(bomb, "x.xml", b"0" * (5 * 1024 * 1024))
        assert detect.docx_to_markdown(bomb) == ""
        assert detect.xlsx_to_markdown(bomb) == ""


def test_legit_multi_member_passes_streaming(tmp_path):
    """A normal multi-member office zip passes the streaming-ceiling pass."""
    ok = tmp_path / "ok.xlsx"
    with zipfile.ZipFile(ok, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b"<types/>")
        zf.writestr("xl/workbook.xml", b"<workbook/>" * 100)
        zf.writestr("xl/worksheets/sheet1.xml", b"<sheetData>rows</sheetData>" * 500)
    assert detect._zip_within_caps(ok) is True


def test_streaming_ceiling_rejects_oversized_actual(tmp_path, monkeypatch):
    """With a low decompressed cap, content whose actual bytes exceed it is rejected.

    This exercises the authoritative bounded-decompression pass: the function
    reads real decompressed bytes (not the attacker-declared central-directory
    sizes) and stops once the ceiling is crossed.
    """
    monkeypatch.setattr(detect, "_OFFICE_MAX_DECOMPRESSED_BYTES", 64 * 1024)  # 64 KiB
    f = tmp_path / "big.xlsx"
    # ~512 KiB of incompressible data: low ratio (passes the ratio pre-filter),
    # but real decompressed size far exceeds the 64 KiB ceiling.
    import os as _os
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/x.xml", _os.urandom(512 * 1024))
    assert detect._zip_within_caps(f) is False


def test_pdf_over_cap_returns_empty(tmp_path, monkeypatch):
    """A PDF larger than the raw cap is skipped before pypdf opens it."""
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.4\n" + b"x" * 4096)
    # shrink the cap via the helper's default by patching the module constant and
    # calling through a wrapper that reads it fresh
    monkeypatch.setattr(detect, "_OFFICE_MAX_RAW_BYTES", 100)
    monkeypatch.setattr(detect, "_file_within_size_cap",
                        lambda p, cap=100: p.stat().st_size <= cap if p.exists() else False)
    assert detect.extract_pdf_text(big) == ""


def test_pdf_over_cap_returns_empty_pdfplumber(tmp_path, monkeypatch):
    """Size cap fires before pdfplumber opens the file (same guard as pypdf path)."""
    pytest.importorskip("pdfplumber")
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.4\n" + b"x" * 4096)
    monkeypatch.setenv("GRAPHIFY_PDF_BACKEND", "pdfplumber")
    monkeypatch.setattr(detect, "_OFFICE_MAX_RAW_BYTES", 100)
    monkeypatch.setattr(detect, "_file_within_size_cap",
                        lambda p, cap=100: p.stat().st_size <= cap if p.exists() else False)
    assert detect.extract_pdf_text(big) == ""


def test_pdfplumber_backend_extracts_text(tmp_path, monkeypatch):
    """pdfplumber backend returns non-empty text from a real PDF."""
    pdfplumber = pytest.importorskip("pdfplumber")
    real_pdf = Path(__file__).parents[3] / "doc" / "Worldpay_ISO_8583_Reference_Guide_V2.46.pdf"
    if not real_pdf.exists():
        pytest.skip("doc/Worldpay PDF not present")
    monkeypatch.setenv("GRAPHIFY_PDF_BACKEND", "pdfplumber")
    text = detect.extract_pdf_text(real_pdf)
    assert len(text) > 100, "expected substantial text from the Worldpay PDF"


def test_pypdf_backend_extracts_text():
    """pypdf backend (default) returns non-empty text from the same real PDF."""
    pytest.importorskip("pypdf")
    real_pdf = Path(__file__).parents[3] / "doc" / "Worldpay_ISO_8583_Reference_Guide_V2.46.pdf"
    if not real_pdf.exists():
        pytest.skip("doc/Worldpay PDF not present")
    text = detect.extract_pdf_text(real_pdf)
    assert len(text) > 100, "expected substantial text from the Worldpay PDF"


def test_pdfplumber_extracts_table_content(monkeypatch):
    """pdfplumber backend captures table-cell text that pypdf may miss."""
    pdfplumber = pytest.importorskip("pdfplumber")
    real_pdf = Path(__file__).parents[3] / "doc" / "Worldpay_ISO_8583_Reference_Guide_V2.46.pdf"
    if not real_pdf.exists():
        pytest.skip("doc/Worldpay PDF not present")
    monkeypatch.setenv("GRAPHIFY_PDF_BACKEND", "pdfplumber")
    text = detect.extract_pdf_text(real_pdf)
    # The Worldpay guide has ISO 8583 field tables; assert some field-related token present
    assert any(tok in text for tok in ("ISO", "8583", "Field", "Bitmap")), (
        "expected ISO 8583 table content in pdfplumber output"
    )


# ---------------------------------------------------------------------------
# extract_paper_no_llm — deterministic graph extraction
# ---------------------------------------------------------------------------

class TestExtractPaperNoLlm:
    PDF = Path(__file__).parents[3] / "doc" / "Worldpay_ISO_8583_Reference_Guide_V2.46.pdf"

    def _result(self, monkeypatch, backend="pdfplumber"):
        if backend == "pdfplumber":
            pytest.importorskip("pdfplumber")
        else:
            pytest.importorskip("pypdf")
        if not self.PDF.exists():
            pytest.skip("doc/Worldpay PDF not present")
        monkeypatch.setenv("GRAPHIFY_PDF_BACKEND", backend)
        return detect.extract_paper_no_llm(self.PDF)

    def test_returns_nodes_and_edges(self, monkeypatch):
        r = self._result(monkeypatch)
        assert r["nodes"], "expected at least one node"
        assert r["edges"], "expected at least one edge"

    def test_root_document_node_present(self, monkeypatch):
        r = self._result(monkeypatch)
        ids = {n["id"] for n in r["nodes"]}
        # root node id is derived from the PDF stem
        assert any("worldpay" in nid.lower() or "iso" in nid.lower() for nid in ids)

    def test_section_nodes_extracted(self, monkeypatch):
        r = self._result(monkeypatch)
        # at least some section nodes beyond the root
        assert len(r["nodes"]) > 1, "expected section nodes in addition to root"

    def test_table_nodes_extracted_pdfplumber(self, monkeypatch):
        r = self._result(monkeypatch, backend="pdfplumber")
        table_nodes = [n for n in r["nodes"] if "table" in n["id"]]
        assert table_nodes, "pdfplumber backend should extract table nodes"

    def test_all_edges_reference_valid_nodes(self, monkeypatch):
        r = self._result(monkeypatch)
        ids = {n["id"] for n in r["nodes"]}
        for e in r["edges"]:
            assert e["source"] in ids, f"edge source {e['source']!r} has no node"
            assert e["target"] in ids, f"edge target {e['target']!r} has no node"

    def test_size_cap_returns_empty(self, monkeypatch):
        if not self.PDF.exists():
            pytest.skip("doc/Worldpay PDF not present")
        monkeypatch.setattr(detect, "_file_within_size_cap", lambda p, **kw: False)
        r = detect.extract_paper_no_llm(self.PDF)
        assert r == {"nodes": [], "edges": []}

    def test_pypdf_backend_also_produces_nodes(self, monkeypatch):
        r = self._result(monkeypatch, backend="pypdf")
        assert r["nodes"], "pypdf backend should also produce nodes"
        assert r["edges"], "pypdf backend should also produce edges"
