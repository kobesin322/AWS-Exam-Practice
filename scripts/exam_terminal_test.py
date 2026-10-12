"""Checks for the terminal exam: domain mix, scoring, and a scripted sitting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exam_terminal as exam  # noqa: E402

POOL = json.loads((ROOT / "assets" / "data" / "questions.json").read_text(encoding="utf-8"))
EXPECTED = {
    "cloud-concepts": 8,
    "security-compliance": 9,
    "technology-services": 11,
    "billing-support": 4,
}


def rng(seed: int):
    source = __import__("random").Random(seed)

    def roll() -> float:
        return source.random()

    return roll


def test_engine() -> None:
    drawn = exam.build_exam(POOL["questions"], POOL["meta"], rng(7))
    again = exam.build_exam(POOL["questions"], POOL["meta"], rng(7))
    assert [item["id"] for item in drawn] == [item["id"] for item in again]
    assert len(drawn) == 32
    assert len({item["stemKey"] for item in drawn}) == 32
    counts: dict[str, int] = {}
    for question in drawn:
        counts[question["domain"]] = counts.get(question["domain"], 0) + 1
    assert counts == EXPECTED
    assert exam.is_correct(["E", "B"], ["B", "E"])
    assert not exam.is_correct(["B"], ["B", "E"])
    pool = exam.record_result([], "q1", ["A"], False, "2026-10-09T00:00:00+00:00")
    pool = exam.record_result(pool, "q1", ["C"], False, "2026-10-09T01:00:00+00:00")
    assert pool[0]["timesMissed"] == 2
    assert pool[0]["selected"] == ["C"]
    assert exam.record_result(pool, "q1", ["B"], True) == []
    assert exam.passed(22, 32, 0.7) is False
    assert exam.passed(23, 32, 0.7) is True
    assert exam.parse_entry("b, e", {"A", "B", "E"}) == ("answer", ["B", "E"])
    assert exam.parse_entry("b", {"A", "B", "E"}) == ("answer", ["B"])
    assert exam.parse_entry("e", {"A", "B", "E"}) == ("answer", ["E"])
    assert exam.parse_entry("end", {"A", "B", "E"}) == ("command", "end")
    assert exam.parse_entry("n", {"A"}) == ("command", "n")
    assert exam.parse_entry("q", {"A", "B"}) == ("command", "q")
    assert exam.parse_entry("g 12", {"A"}) == ("command", "g 12")
    assert exam.parse_entry("goto 3", {"A"}) == ("command", "goto 3")


def test_scripted_session() -> None:
    drawn = exam.build_exam(POOL["questions"], POOL["meta"], rng(7))
    first = drawn[0]
    wrong = next(choice["letter"] for choice in first["choices"] if choice["letter"] not in first["answers"])
    script = "\n".join(["1", wrong, "end", "y", "", "3", "2", "b", "b", "q", ""]) + "\n"
    with tempfile.TemporaryDirectory() as folder:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "exam_terminal.py"),
                "--pool",
                str(ROOT / "assets" / "data" / "questions.json"),
                "--seed",
                "7",
                "--data-dir",
                folder,
            ],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            cwd=ROOT,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        assert "Wrong." in output
        assert "Saved to the wrong pool." in output
        assert "0 / 32" in output
        assert "under the 70% pass mark" in output
        saved = json.loads((Path(folder) / "wrong-pool.json").read_text(encoding="utf-8"))
        assert saved[0]["id"] == first["id"]
        assert saved[0]["selected"] == [wrong]
        assert "Your answer:" in output


MLA_COUNTS = {
    "data-preparation": 18,
    "model-development": 17,
    "deployment": 14,
    "monitoring-security": 16,
}


def test_modules_stay_separate() -> None:
    catalog = exam.load_catalog(ROOT)
    assert [item["id"] for item in catalog] == ["clf-c02", "mla-c01"]
    assert "specialty" not in json.dumps(catalog).lower()
    mla = json.loads((ROOT / "assets" / "data" / "mla-c01.json").read_text(encoding="utf-8"))
    clf = json.loads((ROOT / "assets" / "data" / "questions.json").read_text(encoding="utf-8"))
    assert mla["meta"]["examCode"] == "MLA-C01"
    assert mla["meta"]["examSize"] == 65
    assert mla["meta"]["minutes"] == 130
    assert mla["meta"]["passMark"] == 0.72
    assert clf["meta"]["examCode"] == "CLF-C02"
    counts: dict[str, int] = {}
    for question in mla["questions"]:
        assert question["id"].startswith("mla_")
        assert "Specialty" not in question["source"]
        assert "MLA-C01" in question["source"]
        counts[question["domain"]] = counts.get(question["domain"], 0) + 1
    assert {domain["id"]: domain["examCount"] for domain in mla["meta"]["domains"]} == MLA_COUNTS
    for domain in mla["meta"]["domains"]:
        assert counts[domain["id"]] >= domain["examCount"]
    drawn = exam.build_exam(mla["questions"], mla["meta"], rng(7))
    drawn_counts: dict[str, int] = {}
    for question in drawn:
        drawn_counts[question["domain"]] = drawn_counts.get(question["domain"], 0) + 1
    assert len(drawn) == 65
    assert drawn_counts == MLA_COUNTS
    assert {item["id"] for item in clf["questions"]}.isdisjoint({item["id"] for item in mla["questions"]})
    assert exam.default_data_dir("clf-c02") != exam.default_data_dir("mla-c01")
    assert exam.default_data_dir("mla-c01").name == "mla-c01"


def test_module_menu() -> None:
    script = "\n".join(["2", "m", "q"]) + "\n"
    with tempfile.TemporaryDirectory() as folder:
        env = dict(**{key: value for key, value in __import__("os").environ.items()})
        env["LOCALAPPDATA"] = folder
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "exam_terminal.py")],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            cwd=ROOT,
            env=env,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        assert "Choose an exam" in output
        assert "MLA-C01" in output
        assert "Machine Learning Specialty is not included." in output
        assert "Specialty" not in output.replace("Machine Learning Specialty is not included.", "")
        mla_pool = Path(folder) / "aws-exam-practice" / "mla-c01" / "wrong-pool.json"
        assert mla_pool.is_file()
        assert not (Path(folder) / "clf02-practice").exists()


if __name__ == "__main__":
    test_engine()
    test_scripted_session()
    test_modules_stay_separate()
    test_module_menu()
    print("exam terminal tests passed")
