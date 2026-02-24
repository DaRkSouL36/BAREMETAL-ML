import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION AND FILE PATHS
# ==============================================================================
CPP_CSV = "RESULT/C_CPP_IMPLEMENTATION_RESULTS.csv"
PYTHON_CSV = "RESULT/PYTHON_SKLEARN_RESULTS.csv"
OUTPUT_CHART = "RESULT/MODEL_COMPARISON_CHART.png"


def load_and_merge_data():
    """LOADS BOTH CSVS AND MERGES THEM ON MODEL AND METRIC."""
    if not os.path.exists(CPP_CSV) or not os.path.exists(PYTHON_CSV):
        print("[ERROR] ONE OR BOTH CSV FILES ARE MISSING IN THE 'RESULT/' DIRECTORY.")
        print("PLEASE RUN BOTH 'Main.exe' AND 'Validation.py' FIRST.")
        sys.exit(1)

    # ==============================================================================
    # LOAD DATA (HANDLE HEADER INCONSISTENCIES)
    # ==============================================================================

    # CHECK IF C/C++ CSV HAS HEADER
    with open(CPP_CSV, "r") as f:
        first_line = f.readline()

    if "MODEL" not in first_line.upper():
        # FILE HAS NO HEADER → ASSIGN MANUALLY
        df_cpp = pd.read_csv(
            CPP_CSV,
            header=None,
            names=["LANGUAGE", "MODEL", "METRIC", "VALUE"],
        )
    else:
        df_cpp = pd.read_csv(CPP_CSV)

    # PYTHON CSV (EXPECTED TO HAVE HEADER)
    df_py = pd.read_csv(PYTHON_CSV)

    # REMOVE DUPLICATE ROWS (IF ANY)
    df_py = df_py.drop_duplicates()

    # ==============================================================================
    # VALIDATE REQUIRED COLUMNS
    # ==============================================================================
    required_cols = {"MODEL", "METRIC", "VALUE"}

    if not required_cols.issubset(df_cpp.columns):
        print("[ERROR] C/C++ CSV FORMAT INVALID. REQUIRED COLUMNS:", required_cols)
        print("FOUND:", df_cpp.columns.tolist())
        sys.exit(1)

    if not required_cols.issubset(df_py.columns):
        print("[ERROR] PYTHON CSV FORMAT INVALID. REQUIRED COLUMNS:", required_cols)
        print("FOUND:", df_py.columns.tolist())
        sys.exit(1)

    # ==============================================================================
    # RENAME VALUE COLUMNS TO REFLECT LANGUAGE
    # ==============================================================================
    df_cpp = df_cpp.rename(columns={"VALUE": "C_CPP_VALUE"})
    df_py = df_py.rename(columns={"VALUE": "PYTHON_VALUE"})

    # DROP LANGUAGE COLUMN IF PRESENT
    df_cpp = df_cpp.drop(columns=["LANGUAGE"], errors="ignore")
    df_py = df_py.drop(columns=["LANGUAGE"], errors="ignore")

    # ==============================================================================
    # MERGE ON MODEL AND METRIC
    # ==============================================================================
    df_merged = pd.merge(df_cpp, df_py, on=["MODEL", "METRIC"], how="inner")

    # CALCULATE ABSOLUTE DIFFERENCE
    df_merged["DIFFERENCE"] = df_merged["C_CPP_VALUE"] - df_merged["PYTHON_VALUE"]

    return df_merged


def print_terminal_report(df):
    """PRINTS A CLEAN, FORMATTED COMPARISON TABLE TO THE CONSOLE."""
    print("\n" + "=" * 100)
    print(f"{'BAREMETAL-ML VS SCIKIT-LEARN: AUTOMATED BENCHMARK REPORT':^100}")
    print("=" * 100)

    print(
        f"{'MODEL':<40} | {'METRIC':<10} | {'C/C++':>12} | {'PYTHON':>12} | {'DIFF':>12}"
    )
    print("-" * 100)

    for _, row in df.iterrows():
        model = row["MODEL"][:38]
        metric = row["METRIC"]
        val_c = row["C_CPP_VALUE"]
        val_p = row["PYTHON_VALUE"]
        diff = row["DIFFERENCE"]

        if metric == "MSE":
            print(
                f"{model:<40} | {metric:<10} | {val_c:>12.6e} | {val_p:>12.6e} | {diff:>+12.6e}"
            )
        else:
            print(
                f"{model:<40} | {metric:<10} | {val_c:>12.6f} | {val_p:>12.6f} | {diff:>+12.6f}"
            )

    print("=" * 100 + "\n")


def generate_comparison_chart(df):
    """GENERATES AND SAVES A GROUPED BAR CHART OF THE RESULTS."""
    # SEPARATE ACCURACY AND MSE (DIFFERENT SCALES)
    df_acc = df[df["METRIC"] == "ACCURACY"]
    df_mse = df[df["METRIC"] == "MSE"]

    # GLOBAL FONT SETTINGS (UPPERCASE + BOLD)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "BAREMETAL-ML C/C++ VS SCIKIT-LEARN PYTHON",
        fontsize=16,
        fontweight="bold",
    )

    # PLOT 1: ACCURACY (CLASSIFICATION)
    if not df_acc.empty:
        x = np.arange(len(df_acc["MODEL"]))
        width = 0.35

        ax1.bar(
            x - width / 2,
            df_acc["C_CPP_VALUE"],
            width,
            label="C/C++ (BAREMETAL)",
        )
        ax1.bar(
            x + width / 2,
            df_acc["PYTHON_VALUE"],
            width,
            label="PYTHON (SCIKIT-LEARN)",
        )

        ax1.set_ylabel("ACCURACY SCORE", fontweight="bold")
        ax1.set_title(
            "CLASSIFICATION PERFORMANCE (HIGHER IS BETTER)", fontweight="bold"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [m.upper() for m in df_acc["MODEL"]],
            rotation=15,
            ha="right",
            fontweight="bold",
        )
        ax1.set_ylim(0, 1.1)
        ax1.legend(prop={"weight": "bold"})
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # PLOT 2: MSE (REGRESSION)
    if not df_mse.empty:
        x = np.arange(len(df_mse["MODEL"]))
        width = 0.35

        ax2.bar(
            x - width / 2,
            df_mse["C_CPP_VALUE"],
            width,
            label="C/C++ (BAREMETAL)",
        )
        ax2.bar(
            x + width / 2,
            df_mse["PYTHON_VALUE"],
            width,
            label="PYTHON (SCIKIT-LEARN)",
        )

        ax2.set_ylabel("MEAN SQUARED ERROR (MSE)", fontweight="bold")
        ax2.set_title("REGRESSION PERFORMANCE (LOWER IS BETTER)", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [m.upper() for m in df_mse["MODEL"]],
            rotation=15,
            ha="right",
            fontweight="bold",
        )
        ax2.set_yscale("log")
        ax2.legend(prop={"weight": "bold"})
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=500)
    print(f"[SYSTEM] COMPARISON CHART SAVED SUCCESSFULLY TO: {OUTPUT_CHART}")


if __name__ == "__main__":
    merged_data = load_and_merge_data()
    print_terminal_report(merged_data)
    generate_comparison_chart(merged_data)
