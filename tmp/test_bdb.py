import pandas as pd, numpy as np
from io import StringIO

def test_bdb_round_trip(n_bits, prec_d=None, N_samples=None):

    assert n_bits in [16, 32, 64], f"Invalid {n_bits=} (not 16, 32, or 64)"
    params = {16: (np.float16, np.uint16, 5, None),
              32: (np.float32, np.uint32, 9, 1_000),
              64: (np.float64, np.uint64, 17, 500)}
    # See www.exploringbinary.com/number-of-digits-required-for-round-trip-conversions/
    # prec: *minimum* number of decimal digits of precision required for
    # binary-decimal-binary round-trip conversions
    ftype, utype, prec, N = params[n_bits]
    if prec_d:
        prec = prec_d
    info = np.finfo(ftype)
    f_min, f_max = info.smallest_normal, info.max
    e_min, e_max = int(np.ceil(np.log2(f_min))), int(np.floor(np.log2(f_max)))

    if n_bits==16:
        # For half-precision:
        # store all positive, finite binary float16s
        N_pos = 2**15
        grid = np.arange(N_pos, dtype=utype).view(ftype)
        # Filter out 0, subnormals, inf, NaNs
        in_range = (grid>=f_min) & (grid<=f_max)
        grid = grid[in_range]
    else:
        # For single/double precision:
        # store N binary floats larger than & closest to powers of 2 
        if N_samples:
            N = N_samples
        powers_2 = np.array([2**e for e in range(e_min,e_max)], dtype=ftype)
        indexes = powers_2.view(utype)
        grid = (np.kron(indexes, np.ones(N, dtype=utype)) + \
                np.kron(np.ones_like(indexes), np.arange(N, dtype=utype))) \
               .view(ftype)
    # Wrap grid generated in Pandas DataFrame
    df = pd.DataFrame(data=dict(floats=grid), dtype=ftype)
    
    def float_convert(x): return f'{x:.{prec-1}e}'
    out_opts = dict(index=False, float_format=float_convert)
    csv_io = StringIO(df.to_csv(**out_opts))
    in_opts = dict(index_col=None, dtype={'floats':ftype},
                   float_precision='round_trip')
    bdb_df = pd.read_csv(csv_io, **in_opts)
    pd.testing.assert_frame_equal(df, bdb_df, check_exact=True)
