import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from abdominal_overscan.failure_reporting import write_failed_cases


def test_write_failed_cases_writes_and_clears(tmp_path):
    csv_path = tmp_path / "overscanning_results.csv"
    report_path = write_failed_cases(
        csv_path,
        "caudal",
        [{"file_name": "case.nii.gz", "reason": "no valid pubic landmark"}],
    )

    text = report_path.read_text(encoding="utf-8-sig")
    assert "stage,file_name,reason" in text
    assert "caudal,case.nii.gz,no valid pubic landmark" in text

    assert write_failed_cases(csv_path, "caudal", []) == report_path
    assert not report_path.exists()
