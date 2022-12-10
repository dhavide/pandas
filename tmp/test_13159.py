# Test case based on example code provided in Pandas issue #13159
# See https://github.com/pandas-dev/pandas/issues/13159

import pandas as pd, numpy as np
from io import StringIO

print('Pandas: {}\n'.format(pd.__version__))
x0 = 18292498239.824
df = pd.DataFrame({'data': x0},index=["num"])
print(df, '\n')

# Exporting to CSV
out = df.to_csv()
contents = StringIO(out)

# Use float_precision=None, 'high', or 'round_trip'
# as options when calling pd.read_csv
input_opts = [dict(float_precision=None),
              dict(float_precision='high'),
              dict(float_precision='round_trip')
              ]

# Generate list of dataframes from CSV file when
# reading with various options
dfs = []
for opts in input_opts:
    contents.seek(0)
    dfs.append(pd.read_csv(contents, **opts))

# Compare round-trip values with original input value
for k,df in enumerate(dfs):
    x = df['data'][0]
    print('num: {}\toptions: {}'.format(x, input_opts[k]))
    assert x0==x, "x0<>x\tRel. error: {:12e}".format(abs(x0-x)/abs(x0))

# Finally, compare value parsed from CSV directly
contents.seek(0)
ll = contents.readlines()
x = float(ll[1].split(',')[1].rstrip())
print('num: {}\tvalue parsed directly\n\n'.format(x))
assert x0==x, "x0<>x\tRel. error: {:12e}".format(abs(x0-x)/abs(x0))

print("All assertions passed.")
