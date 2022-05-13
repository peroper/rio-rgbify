import os

import click
from click.testing import CliRunner

import numpy as np

import rasterio as rio
from rio_rgbify.scripts.cli import rgbify

from raster_tester.compare import affaux, upsample_array


in_elev_src = os.path.join(os.path.dirname(__file__), "fixtures", "elev.tif")
expected_src = os.path.join(os.path.dirname(__file__), "expected", "elev-rgb.tif")


def flex_compare(r1, r2, thresh=10):
    upsample = 4
    r1 = r1[::upsample]
    r2 = r2[::upsample]
    toAff, frAff = affaux(upsample)
    r1 = upsample_array(r1, upsample, frAff, toAff)
    r2 = upsample_array(r2, upsample, frAff, toAff)
    tdiff = np.abs(r1.astype(np.float64) - r2.astype(np.float64))

    click.echo(
        "{0} values exceed the threshold difference with a max variance of {1}".format(
            np.sum(tdiff > thresh), tdiff.max()
        ),
        err=True,
    )

    return not np.any(tdiff > thresh)


def test_cli_good_elev():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            rgbify,
            [in_elev_src, "rgb.tif", "--interval", 0.001, "--base-val", -100, "-j", 1],
        )

        assert result.exit_code == 0

        with rio.open("rgb.tif") as created:
            with rio.open(expected_src) as expected:
                carr = created.read()
                earr = expected.read()
                for a, b in zip(carr, earr):
                    assert flex_compare(a, b)


def test_cli_fail_elev():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                "rgb.tif",
                "--interval",
                0.00000001,
                "--base-val",
                -100,
                "-j",
                1,
            ],
        )
        assert result.exit_code == 1
        assert result.exception


def test_bad_input_format():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles = "output.lol"
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles,
                "--min-z",
                10,
                "--max-z",
                9,
                "--format",
                "webp",
                "-j",
                1,
            ],
        )
        assert result.exit_code == 1
        assert result.exception
