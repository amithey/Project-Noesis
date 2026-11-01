import pytest

from noesis_gpu import chain_run


def test_driver_requires_explicit_pins_and_never_creates_weights(tmp_path):
    with pytest.raises(SystemExit):
        chain_run.parser().parse_args([])
    args = chain_run.parser().parse_args(["--seed", "10006", "--noesis-source", "x", "--data", "d", "--checkpoint", "c",
                                          "--output", str(tmp_path / "o")])
    assert args.seed == 10006 and args.exploration_mv_ms == 1.5 and args.seconds == 5.0


def test_driver_refuses_a_wrong_source_before_creating_the_output(tmp_path):
    args = chain_run.parser().parse_args(["--seed", "10006", "--noesis-source", str(tmp_path), "--data", str(tmp_path),
                                          "--checkpoint", str(tmp_path / "c.npz"), "--output", str(tmp_path / "o")])
    with pytest.raises(ValueError):
        chain_run.run(args)
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("seed", [10, 11, 200, 201, 202])
def test_driver_refuses_held_out_seeds_before_creating_anything(tmp_path, seed):
    args = chain_run.parser().parse_args(["--seed", str(seed), "--noesis-source", str(tmp_path), "--data", str(tmp_path),
                                          "--checkpoint", str(tmp_path / "c.npz"), "--output", str(tmp_path / "o")])
    with pytest.raises(ValueError, match="held-out"):
        chain_run.run(args)
    assert not (tmp_path / "o").exists()
