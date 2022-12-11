import pandas as pd, numpy as np
from io import StringIO

prec_d = 15
d_free = 2
N = 10**d_free # Number of samples on each interval
ftype = np.float64
info = np.finfo(ftype)
f_min, f_max = info.smallest_normal, info.max
e_min, e_max = int(np.ceil(np.log10(f_min))), int(np.floor(np.log10(f_max)))

print(f'{f_min=:10.2e}\t{e_min=}')
print(f'9.0..0*10**{e_min} ... 9.9..9*10**{e_min}\n\n')
print(f'{f_max=:10.2e}\t{e_max=}')
print(f'9.0..0*10**{e_max-1} ... 9.9..9*10**{e_max-1}')

assert prec_d>d_free, 'Too many digits for {prec_d=}<={d_free}'
k_max = int(10**prec_d)
k_min = k_max - int(10**d_free)
mantissas = [ str(k).replace('9','9.',1) for k in range(k_min, k_max) ]
#d_range = [ f'{k:0{d_free}d}' for k in range(N) ]
#mantissas = [ '9.' + '9'*(prec_d-(d_free+1)) + digits for digits in d_range]

exponents = [ f'e{e:+03d}' for e in range(e_min, e_max) ]
grid = [ m+e for e in exponents for m in mantissas ]

assert ftype(grid[0]) > f_min, f'{ftype(grid[0])=:10.3e} <= {f_min}'
assert ftype(grid[-1]) < f_max, f'{ftype(grid[-1])=:10.3e} >= {f_max}'
s = 'floats\n' + '\n'.join(grid)
csv_io = StringIO(s)

out_opts = dict(index=False, float_format=lambda x:f'{x:.{prec_d-1}e}')
in_opts = dict(index_col=None, dtype={'floats':ftype},
               float_precision='round_trip')

df = pd.read_csv(csv_io, **in_opts)
s2 = df.to_csv(**out_opts)

# Re do now, comparing strings
csv_io.seek(0)
in_opts = dict(index_col=None, dtype={'floats':np.str_})
df1 = pd.read_csv(csv_io, **in_opts)
csv_io = StringIO(s2)
df2 = pd.read_csv(csv_io, **in_opts)

pd.testing.assert_frame_equal(df1, df2)
