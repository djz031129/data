from __future__ import annotations

from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from figure_utils import (  # noqa: E402
    build_fig1,
    build_fig2,
    build_fig3,
    build_fig4,
    output_dir,
    si_dir_from_script,
)


def main() -> None:
    si_dir = si_dir_from_script(__file__)
    out_dir = output_dir(si_dir, "full_figures")
    outputs = [
        build_fig1(si_dir, out_dir / "Fig1_dataset_overview_workflow_and_correlation.png"),
        build_fig2(si_dir, out_dir / "Fig2_test_set_model_comparison.png"),
        build_fig3(si_dir, out_dir / "Fig3_independent_validation_predicted_vs_actual.png"),
        build_fig4(si_dir, out_dir / "Fig4_shap_analysis_and_hardness_conductivity_screening.png"),
    ]
    print("Generated complete figures:")
    for path in outputs:
        print(f"- {path.relative_to(si_dir)}")


if __name__ == "__main__":
    main()
