import pandas as pd, numpy as np
from io import StringIO

def test_bdb_round_trip(n_bits, prec_d=None, N=None):

    assert n_bits in [16, 32, 64], \
           f"Invalid {n_bits=} (not 16, 32, or 64)"
    params = {16: (np.float16, np.uint16, 5, 1_024),
              32: (np.float32, np.uint32, 9, 1_000),
              64: (np.float64, np.uint64, 17, 500)}
    # See www.exploringbinary.com/number-of-digits-required-for-round-trip-
    # conversions/
    # prec: *minimum* number of decimal digits of precision required for
    # binary-decimal-binary round-trip conversions
    _ftype, _utype, _prec_d, _N = params[n_bits]
    if prec_d: _prec_d = prec_d
    if N: _N = N
    info = np.finfo(_ftype)
    f_min, f_max = info.smallest_normal, info.max
    e_min, e_max = info.minexp, info.maxexp

    # Store all binary floats 2**e + k*delta where 0 <= k < N,
    # delta is the distance between succesive floats on the interval
    # [2**e, 2**(e+1)), and e_min <= e < e_max.
    # (For float16, this generates *all* positive, finite floats.)
    powers_2 = np.array([2**e for e in range(e_min,e_max)], dtype=_ftype)
    indexes = powers_2.view(_utype)
    grid = (np.kron(indexes, np.ones(_N, dtype=_utype)) + \
            np.kron(np.ones_like(indexes), np.arange(_N, dtype=_utype))) \
           .view(_ftype)
    # Wrap grid generated in Pandas DataFrame
    df = pd.DataFrame(data=dict(floats=grid), dtype=_ftype)

    # keyword args for calls to df.to_csv & pd.read_csv respectively    
    out_opts = dict(index=False, float_format=lambda x:f'{x:.{_prec_d-1}e}')
    in_opts = dict(index_col=None, dtype={'floats':_ftype},
                   float_precision='round_trip')

    csv_io = StringIO(df.to_csv(**out_opts))
    bdb_df = pd.read_csv(csv_io, **in_opts)

    pd.testing.assert_frame_equal(df, bdb_df, check_exact=True)
