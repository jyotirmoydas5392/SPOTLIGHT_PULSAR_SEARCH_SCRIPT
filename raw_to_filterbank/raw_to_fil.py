import logging
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import ExitStack

import pytz
import numpy as np
import pandas as pd
import astropy.units as U
from priwo import writehdr
from astropy.time import Time
from joblib import Parallel, delayed
from astropy.coordinates import SkyCoord


logging.basicConfig(level="INFO", datefmt="[%X]", format="%(message)s")
log = logging.getLogger("ver0")


def read_asciihdr(fn: str | Path) -> tuple[dict, pd.DataFrame]:
    hdr = {}
    extras = []
    with open(fn, "r") as lines:
        for line in lines:
            if line.startswith(("#", " ")):
                extras.append(line)
                continue
            key, val = line.split("=")
            key = key.strip()
            val = val.strip()
            try:
                name, conv = {
                    "Header file": ("fname", str),
                    "Beam ID": ("beamid", int),
                    "Host ID": ("hostid", int),
                    "Host name": ("hostname", str),
                    "GTAC code": ("gtaccode", str),
                    "Observer": ("observer", str),
                    "GTAC title": ("gtactitle", str),
                    "Source": ("source", str),
                    "Source RA (Rad)": ("ra", float),
                    "Source DEC (Rad)": ("dec", float),
                    "Channels": ("nf", int),
                    "Bandwidth (MHz)": ("bw", float),
                    "Frequency Ch. 0  (Hz)": ("f0", lambda x: float(x) * 1e-6),
                    "Channel width (Hz)": ("df", lambda x: float(x) * 1e-6),
                    "Sampling time  (uSec)": ("dt", lambda x: float(x) * 1e-6),
                    "Antenna mask pol1": (
                        "maskX",
                        lambda x: [int(_) == 1 for _ in np.binary_repr(int(x, 0))],
                    ),
                    "Antennas pol1": ("antX", lambda x: list(x.split())),
                    "Antenna mask pol2": (
                        "maskY",
                        lambda x: [int(_) == 1 for _ in np.binary_repr(int(x, 0))],
                    ),
                    "Antennas pol2": ("antY", lambda x: list(x.split())),
                    "Beam mode": ("beammode", str),
                    "No. of stokes": ("npol", int),
                    "Num bits/sample": ("nbits", int),
                    "De-Disperion DM": (
                        "dm",
                        lambda x: None if x == "NA" else float(x),
                    ),
                    "PFB": ("pfb", bool),
                    "PFB number of taps": (
                        "pfbtaps",
                        lambda x: None if x == "NA" else float(x),
                    ),
                    "WALSH": ("walsh", lambda x: False if x == "OFF" else True),
                    "Broad-band RFI Filter": (
                        "rfifilter",
                        lambda x: False if x == "OFF" else True,
                    ),
                    "Date": ("istdate", str),
                    "IST Time": ("isttime", str),
                }[key]
                hdr[name] = conv(val)
            except KeyError:
                pass

    fields = []
    header = extras[1].split()[2:]
    extras = extras[2:]
    for extra in extras:
        fields.append([float(_) for _ in extra.split()])
    radecs = pd.DataFrame(fields, columns=header)

    hdr["istdatetime"] = datetime.strptime(
        " ".join(
            [
                hdr["istdate"],
                hdr["isttime"][:-3],
            ]
        ),
        "%d/%m/%Y %H:%M:%S.%f",
    )

    return hdr, radecs


def getmjd(DT: datetime):
    localtz = pytz.timezone("Asia/Kolkata")
    localdt = localtz.localize(DT, is_dst=None)
    utcdt = localdt.astimezone(pytz.utc)
    mjd = Time(utcdt).mjd
    return mjd


def inchunks(fx, N: int):
    while True:
        data = fx.read(N)
        if (not data) or len(data) < N:
            break
        yield data


def xtract2fil(fn: str | Path, nbeams: int, outdir: str | Path):
    fn = Path(fn)
    outdir = Path(outdir)

    hdr, radecs = read_asciihdr(str(fn) + ".ahdr")

    df = hdr["df"]
    bw = hdr["bw"]
    dt = hdr["dt"]
    nf = hdr["nf"]
    nbits = hdr["nbits"]
    fname = str(fn.name)
    source = hdr["source"]
    mjd = getmjd(hdr["istdatetime"])

    nbits = 8 if nbits != 8 else nbits  # HACK: Need this for now.
    dt = dt if dt > 0.0 else 1.31072e-3  # HACK: Need this for now.

    if df > 0.0:
        fh = hdr["f0"] + bw - (0.5 * df)
        df = -df
        toflip = True
    else:
        fh = hdr["f0"]
        toflip = False

    nt = 800
    nblocks = 32
    sbsize = nf * nt * nblocks

    outpaths = [
        outdir / f"BM{ix}.fil"
        for ix in (radecs["BM-Idx"].to_numpy(dtype=int))
    ]

    for ix, outpath in enumerate(outpaths):
        coords = SkyCoord(radecs.iloc[ix]["RA"] * U.rad, radecs.iloc[ix]["DEC"] * U.rad)

        rad = int(coords.ra.hms.h)
        ram = int(coords.ra.hms.m)
        ras = float(coords.ra.hms.s)
        if rad < 0.0:
            ram = -ram
            ras = -ras
        ra_sigproc = float("".join([str(rad), str(ram), str(ras)]))

        decd = int(coords.dec.dms.d)
        decm = int(coords.dec.dms.m)
        decs = float(coords.dec.dms.s)
        if decd < 0.0:
            decm = -decm
            decs = -decs
        dec_sigproc = float("".join([str(decd), str(decm), str(decs)]))

        writehdr(
            {
                "rawdatafile": fname,
                "source_name": source,
                "nifs": 1,
                "nbits": nbits,
                "data_type": 1,
                "machine_id": 7,
                "telescope_id": 7,
                "barycentric": 0,
                "pulsarcentric": 0,
                "tstart": mjd,
                "foff": df,
                "fch1": fh,
                "tsamp": dt,
                "nchans": nf,
                "src_raj": ra_sigproc,
                "src_dej": dec_sigproc,
                "size": 0,
            },
            str(outpath),
        )

    with ExitStack() as stack:
        outfiles = [stack.enter_context(open(outpath, "ab")) for outpath in outpaths]
        with open(fn, "rb") as fx:
            for ix, data in enumerate(inchunks(fx, sbsize)):
                outfile = outfiles[ix % nbeams]
                array = np.frombuffer(data, dtype=np.uint8).reshape(-1, nf)
                array = np.fliplr(array) if toflip else array
                array.tofile(outfile)
    log.info(f"Done with file: {fn}")


def main():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("files", nargs="*")
    parser.add_argument("-n", "--njobs", type=int)
    parser.add_argument("-b", "--nbeams", type=int, default=10)
    parser.add_argument("-o", "--output", type=Path, default=Path.cwd())
    args = parser.parse_args()

    log.info(f"Xtracting {args.nbeams*len(args.files)} beams from {len(args.files)} raw files...")
    Parallel(n_jobs=args.njobs if args.njobs is not None else len(args.files))(
        delayed(xtract2fil)(
            fn=f,
            nbeams=args.nbeams,
            outdir=args.output,
        )
        for f in args.files
    )


if __name__ == "__main__":
    main()