from difflib import SequenceMatcher

from utils import get_tool_output

from climb.tool.impl.tool_exploratory_data_analysis import exploratory_data_analysis
from climb.tool.tool_comms import ToolCommunicator

EXPECTED_EDA_OUTPUT = """Dataset Shape: 1000 rows and 7 columns
Column Names and Types:
normal_col        float64
non_normal_col    float64
cat_data            int64
str_data           object
nan_data          float64
corr_data         float64
non_corr_data     float64

Descriptive Statistics for Numerical Features:
        normal_col  non_normal_col     cat_data  nan_data    corr_data  non_corr_data
count  1000.000000     1000.000000  1000.000000       0.0  1000.000000    1000.000000
mean     -0.045257        1.047098     3.825000       NaN     0.054743      -0.033296
std       0.987527        1.022801     0.587128       NaN     0.987527       0.949684
min      -3.046143        0.000074     1.000000       NaN    -2.946143      -4.484641
25%      -0.698420        0.312780     4.000000       NaN    -0.598420      -0.410167
50%      -0.058028        0.751197     4.000000       NaN     0.041972      -0.012606
75%       0.606951        1.465758     4.000000       NaN     0.706951       0.335221
max       2.759355        6.520167     4.000000       NaN     2.859355       7.530807
skew      0.033910        1.743307    -3.661486       NaN     0.033910       0.506095
kurt     -0.040977        3.858616    13.038093       NaN    -0.040977       8.585442

Identified numeric value columns that should most likely be considered categoricals:
['cat_data', 'nan_data'].
This is done by checking whether the column contains only integers and has a low number of unique values (<20 or <5% of total examples).

Detailed Information on Categorical Variables:
cat_data - Unique Values: 4
Top 5 Values:
cat_data
4    900
3     50
1     25
2     25

nan_data - Unique Values: 0
Top 5 Values:
Series([], )

str_data - Unique Values: 5
Top 5 Values:
str_data
a    200
b    200
c    200
d    200
e    200

Missing Values Analysis:
nan_data    1000

Count of columns with all NaN values: 1
Correlation Analysis:

Most Positively Correlated Features:
        Feature 1       Feature 2  Correlation
0      normal_col       corr_data     1.000000
1        cat_data   non_corr_data     0.040232
2      normal_col  non_normal_col     0.023444
3  non_normal_col       corr_data     0.023444

Most Negatively Correlated Features:
        Feature 1      Feature 2  Correlation
0       corr_data  non_corr_data    -0.095358
1      normal_col  non_corr_data    -0.095358
2      normal_col       cat_data    -0.056641
3        cat_data      corr_data    -0.056641
4  non_normal_col       cat_data    -0.038787
5  non_normal_col  non_corr_data    -0.005799

Outlier Identification for Numerical Features:
normal_col - Outliers Count: 1
[Lower Bound: -2.98, Upper Bound: 2.89]
non_normal_col - Outliers Count: 32
[Lower Bound: -1.7, Upper Bound: 3.48]
cat_data - Outliers Count: 100
[Lower Bound: 3, Upper Bound: 3]
nan_data - Outliers Count: 0
[Lower Bound: -1, Upper Bound: -1]
corr_data - Outliers Count: 1
[Lower Bound: -2.88, Upper Bound: 2.99]
non_corr_data - Outliers Count: 74
[Lower Bound: -1.71, Upper Bound: 1.64]

Duplicate Records: 0
"""


def test_exploratory_data_analysis(df_eda_path):
    """High level test to make sure that the output from EDA function remains consistent."""

    mock_tc = ToolCommunicator()

    # Execute function with mock_tc
    exploratory_data_analysis(mock_tc, df_eda_path, "", ".")

    tool_return = get_tool_output(mock_tc).tool_return

    # The tool output is a string. Here, we assert that it's not too different from the expected output
    assert SequenceMatcher(None, tool_return, EXPECTED_EDA_OUTPUT).ratio() > 0.9
