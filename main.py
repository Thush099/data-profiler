"""CLI entry point for tabular data profiling, contract validation, and remediation."""

import argparse
from pathlib import Path
from typing import Optional

from config import Config
from src.models import DataContract
from src.pipeline import DataQualityPipeline
from src.utils import format_report_summary, load_data, save_dataframe, save_report


def _resolve_input_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    project_relative = Path(__file__).resolve().parent / raw_path
    if project_relative.exists():
        return project_relative
    return candidate


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Automated tabular data profiler, schema contract validator, and remediation planner."
    )
    parser.add_argument(
        "--input-data",
        default=Config().default_input_path,
        help="Path to input CSV file (default: data/sample_dirty.csv).",
    )
    parser.add_argument(
        "--contract",
        default=None,
        help="Optional path to declarative JSON DataContract file.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target column name to check for class imbalance and target leakage.",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Optional path to write full JSON profile and remediation report.",
    )
    parser.add_argument(
        "--clean-output",
        default=None,
        help="Optional path to write cleaned CSV after executing remediation actions.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run CLI profiling pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config()
    input_path = _resolve_input_path(args.input_data)
    df = load_data(input_path)

    contract: Optional[DataContract] = None
    if args.contract:
        contract = DataContract.from_json_file(_resolve_input_path(args.contract))

    pipeline = DataQualityPipeline(config=config)
    result = pipeline.run(
        df=df,
        contract=contract,
        target=args.target,
        execute_remediation=bool(args.clean_output),
    )
    report_dict = result.to_dict()

    print(
        format_report_summary(
            profile_dict=report_dict["profile"],
            contract_dict=report_dict["contract_validation"],
            remediation_dict=report_dict["remediation_plan"],
        )
    )

    if args.output_report:
        saved_report = save_report(report_dict, args.output_report)
        print(f"\nSaved JSON report to: {saved_report}")

    if args.clean_output and result.remediation_result is not None:
        cleaned_path = save_dataframe(result.remediation_result.cleaned_df, args.clean_output)
        print(
            f"Saved cleaned CSV ({result.remediation_result.rows_after} rows, "
            f"{result.remediation_result.rows_quarantined} quarantined) to: {cleaned_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
