import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "cm"


def init():
    pass


def std_col():
    return "#44444433"


def filter_cumulative_increase(df, eps, col):
    indices_to_keep = [0]
    last_kept_value = df[col].iloc[0]

    for i in range(1, len(df)):
        current_value = df[col].iloc[i]
        if current_value - last_kept_value >= eps:
            indices_to_keep.append(i)
            last_kept_value = current_value

    return df.iloc[indices_to_keep]
