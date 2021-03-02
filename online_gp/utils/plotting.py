from pathlib import Path
import pandas as pd
from upcycle.plotting.credible_regions import get_gaussian_region, combine_trials


def get_arm(exp_dir, arm_name, table_name, x_col, y_col, window=1):
    arm_path = Path(exp_dir) / arm_name
    arm_dfs = [pd.read_csv(f) for f in arm_path.rglob(f'*{table_name}*')]
    print(f"{len(arm_dfs)} tables found in {arm_path.as_posix()}")

    merged_df = combine_trials(arm_dfs, x_col)
    yval_df = merged_df.filter(regex=f'^{y_col}', axis=1)

    x_range = merged_df[x_col]
    arm_data = yval_df.rolling(window, min_periods=1).mean().values

    mean, lb, ub = get_gaussian_region(arm_data.mean(-1), arm_data.var(-1))
    return x_range, mean, lb, ub
