"""Welch block-count helpers shared by serial and parallel FFT paths.

Intentionally free of heavy deps (numpy is the ceiling): base.py must stay
importable when optional parallel-stack packages (e.g. threadpoolctl) are absent.
"""


def welch_nblocks(Ns, nfft, novlap):
    """Number of full Welch blocks under floor partitioning.

    Matches ``scipy.signal.welch``: hop = nfft - novlap, drop the remainder.
    Returns 0 when fewer than one full block fits (including hop <= 0).
    """
    hop = nfft - novlap
    if hop <= 0 or Ns < nfft:
        return 0
    return (int(Ns) - int(novlap)) // hop


def _validate_welch_blocks(Ns, nfft, nblocks, novlap):
    """Reject short records and over-subscribed Welch partitions.

    Matches ``scipy.signal.welch``: floor partitioning, drop the remainder.
    Never clamps the final block start — a clamped block is not an independent
    ensemble member and biases SPOD/BSMD eigenvalues.

    ``novlap >= nfft`` (hop <= 0) is rejected: every block would start at the
    same index and the ensemble would be three copies of one periodogram.
    """
    hop = nfft - novlap
    if hop <= 0:
        raise ValueError(f"Invalid Welch hop: nfft={nfft}, novlap={novlap} (hop={hop} <= 0); novlap must be < nfft")
    if Ns < nfft or nblocks < 1:
        raise ValueError(
            f"Cannot form Welch blocks: Ns={Ns}, nfft={nfft} "
            f"(novlap={novlap}) yield fewer than one full block "
            f"(requested nblocks={nblocks})"
        )
    needed = (nblocks - 1) * hop + nfft
    if needed > Ns:
        raise ValueError(
            f"Requested nblocks={nblocks} does not fit in Ns={Ns} with "
            f"nfft={nfft}, novlap={novlap} (need {needed} samples)"
        )
