import pandas as pd
from solution import add_virtual_column
from pandas.testing import assert_frame_equal

def test_sum_of_two_columns():
    df = pd.DataFrame([[1, 1]] * 2, columns = ["label_one", "label_two"])
    df_expected = pd.DataFrame([[1, 1, 2]] * 2, columns = ["label_one", "label_two", "label_three"])
    df_result = add_virtual_column(df, "label_one+label_two", "label_three")
    assert df_result.equals(df_expected), f"The function should sum the columns: label_one and label_two.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}"


def test_multiplication_of_two_columns():
    df = pd.DataFrame([[1, 1]] * 2, columns = ["label_one", "label_two"])
    df_expected = pd.DataFrame([[1, 1, 1]] * 2, columns = ["label_one", "label_two", "label_three"])
    df_result = add_virtual_column(df, "label_one * label_two", "label_three")
    assert df_result.equals(df_expected), f"The function should multiply the columns: label_one and label_two.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}"


def test_subtraction_of_two_columns():
    df = pd.DataFrame([[1, 1]] * 2, columns = ["label_one", "label_two"])
    df_expected = pd.DataFrame([[1, 1, 0]] * 2, columns = ["label_one", "label_two", "label_three"])
    df_result = add_virtual_column(df, "label_one - label_two", "label_three")
    assert df_result.equals(df_expected), f"The function should subtract the columns: label_one and label_two.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}"


def test_empty_result_when_invalid_labels():
    df = pd.DataFrame([[1, 2]] * 3, columns = ["label_one", "label_two"])
    df_result = add_virtual_column(df, "label_one + label_two", "label3")
    assert df_result.empty, f"Should return an empty df when the \"new_column\" is invalid.\n\nResult:\n\n{df_result}\n\nExpected:\n\nEmpty df"


def test_empty_result_when_invalid_rules():
    df = pd.DataFrame([[1, 1]] * 2, columns = ["label_one", "label_two"])
    df_result = add_virtual_column(df, "label&one + label_two", "label_three")
    assert df_result.empty, f"Should return an empty df when the role have invalid character: '&'.\n\nResult:\n\n{df_result}\n\nExpected:\n\nEmpty df"
    df_result = add_virtual_column(df, "label_five + label_two", "label_three")
    assert df_result.empty, f"Should return an empty df when the role have a column which isn't in the df: 'label_five'.\n\nResult:\n\n{df_result}\n\nExpected:\n\nEmpty df"


def test_when_extra_spaces_in_rules():
    df = pd.DataFrame([[1, 1]] * 2, columns = ["label_one", "label_two"])
    df_expected = pd.DataFrame([[1, 1, 2]] * 2, columns = ["label_one", "label_two", "label_three"])
    df_result = add_virtual_column(df, "label_one + label_two ", "label_three")
    assert df_result.equals(df_expected), f"Should work when the role have spaces between the operation and the column.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}"
    df_result = add_virtual_column(df, "  label_one + label_two ", "label_three")
    assert df_result.equals(df_expected), f"Should work when the role have extra spaces in the start/end.\n\nResult:\n\n{df_result}\n\nExpected:\n\n{df_expected}"

def test_empty_role_returns_empty_df():
    """Empty rule string should produce an empty DataFrame."""
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    out = add_virtual_column(df, "", "c")
    assert out.empty

def test_none_role_returns_empty_df():
    """None as rule string should produce an empty DataFrame."""
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    out = add_virtual_column(df, None, "c")
    assert out.empty

def test_invalid_new_column_name_returns_empty_df():
    """new_column with forbidden characters (e.g. dash) -> empty DF."""
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    out = add_virtual_column(df, "a + b", "total-1")
    assert out.empty

def test_invalid_df_column_name_returns_empty_df():
    """Column in source DataFrame with forbidden name (e.g. dash) -> empty DF."""
    df = pd.DataFrame({"a-b":[1,2], "b":[3,4]})
    out = add_virtual_column(df, "a_b + b", "c")
    assert out.empty

def test_missing_column_in_role_returns_empty_df():
    """Expression refers to a column not in DataFrame -> empty DF."""
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    out = add_virtual_column(df, "a + c", "d")
    assert out.empty

def test_even_tokens_returns_empty_df():
    """Too short or incomplete rule (e.g. 'a +') returns empty DF."""
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    out = add_virtual_column(df, "a +", "c")
    assert out.empty

def test_multiple_ops_left_to_right_letters_only():
    """
    Verifies left-to-right evaluation using column names with letters only.
    In this implementation, 'a + b * c - d' means (((a + b) * c) - d).
    """
    df = pd.DataFrame({"a": [2, 3], "b": [5, 2], "c": [4, 1], "d": [1, 1]})
    # Step-by-step (row-wise):
    # row 0: (2 + 5) = 7, 7 * 4 = 28, 28 - 1 = 27
    # row 1: (3 + 2) = 5, 5 * 1 = 5, 5 - 1 = 4
    expected = df.copy()
    tmp = df["a"] + df["b"]
    tmp = tmp * df["c"]
    expected["out"] = tmp - df["d"]

    out = add_virtual_column(df, "a + b * c - d", "out")
    assert_frame_equal(out, expected)

def test_spaces_and_tabs_are_ignored():
    """Rule may have arbitrary spaces/tabs and yield same result."""
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    out1 = add_virtual_column(df, "a+b", "c")
    out2 = add_virtual_column(df, "  a   +   b  ", "c")
    assert_frame_equal(out1, out2)

def test_nan_propagation():
    """If any value in a row is NaN, result should be NaN for that row."""
    df = pd.DataFrame({"a":[1, None, 3], "b":[3,4,5]})
    out = add_virtual_column(df, "a + b", "c")
    assert pd.isna(out.loc[1, "c"]) and out.loc[0, "c"] == 4 and out.loc[2, "c"] == 8

def test_empty_dataframe_structure():
    """Result should have correct columns even for empty input DataFrame."""
    df = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="int64")})
    out = add_virtual_column(df, "a + b", "c")
    assert list(out.columns) == ["a","b","c"]
    assert len(out) == 0

def test_type_mixed_int_float():
    """Mixed int/float input produces float output (Pandas rules)."""
    df = pd.DataFrame({"a":[1,2], "b":[2.5, 4.5]})
    out = add_virtual_column(df, "a + b", "c")
    assert all(isinstance(v, float) for v in out["c"])