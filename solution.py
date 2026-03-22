import re
import pandas as pd


def add_virtual_column(df: pd.DataFrame, role: str, new_column: str) -> pd.DataFrame:
    """
    Add a computed column to a pandas DataFrame based on a simple arithmetic rule.

    This function validates inputs and evaluates a left-to-right arithmetic expression
    that may reference existing DataFrame columns by name. Supported operators are
    addition (+), subtraction (-), and multiplication (*). If any validation fails,
    the function returns an empty DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame. Column names must consist only of letters and underscores.
    role : str
        Arithmetic rule, e.g. "a + b", "x - y", "col1 * col2".
        Spaces are allowed around operators and names.
    new_column : str
        Name of the computed column to add. Must consist only of letters and underscores.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the new computed column appended.
        If validation fails (invalid names, operators, or unknown columns), returns an empty DataFrame.
    """
    # ------------------------
    # Helpers
    # ------------------------
    def is_valid_name(name: str) -> bool:
        """Return True if 'name' consists only of letters and underscores."""
        if name is None:
            return False
        return bool(re.fullmatch(r"[a-zA-Z_]+", str(name)))

    def tokenize(expr: str):
        """
        Split expression into tokens: names and operators.
        Returns (cols, ops) or (None, None) if tokenization fails.
        """
        if expr is None:
            return None, None
        expr = expr.strip()
        if not expr:
            return None, None

        # Split preserving operators; then trim and drop empties
        parts = re.split(r"(\s*[\+\-\*]\s*)", expr)
        tokens = [t.strip() for t in parts if t and t.strip()]
        # Must be name (op name)* -> odd number of tokens
        if len(tokens) == 0 or len(tokens) % 2 == 0:
            return None, None

        cols = tokens[::2]
        ops = tokens[1::2]
        return cols, ops

    def validate_columns_exist(df_: pd.DataFrame, names: list[str]) -> bool:
        """Return True if all names exist as columns in df_."""
        return all(n in df_.columns for n in names)

    def validate_operator_set(ops: list[str]) -> bool:
        """Return True if every operator is one of +, -, *."""
        return all(op in {"+", "-", "*"} for op in ops)

    # ------------------------
    # 1) Validate new column name and existing df column names
    # ------------------------
    if not is_valid_name(new_column):
        return pd.DataFrame()
    if any(not is_valid_name(c) for c in df.columns):
        return pd.DataFrame()

    # ------------------------
    # 2) Tokenize and validate the rule
    # ------------------------
    cols, ops = tokenize(role)
    if cols is None or ops is None:
        return pd.DataFrame()
    if not validate_operator_set(ops):
        return pd.DataFrame()
    if any(not is_valid_name(c) for c in cols):
        return pd.DataFrame()
    if not validate_columns_exist(df, cols):
        return pd.DataFrame()

    # ------------------------
    # 3) Evaluate left-to-right
    # ------------------------
    result_series = df[cols[0]].copy()
    for op, col in zip(ops, cols[1:]):
        if op == "+":
            result_series = result_series + df[col]
        elif op == "-":
            result_series = result_series - df[col]
        else:  # "*"
            result_series = result_series * df[col]

    # ------------------------
    # 4) Return a copy with the new column
    # ------------------------
    out = df.copy()
    out[new_column] = result_series
    return out