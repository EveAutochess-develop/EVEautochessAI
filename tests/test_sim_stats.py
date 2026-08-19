from eveac_ai.sim_stats import (
    add_summaries,
    bucket_finishes,
    format_gen_stats,
    format_round_stats,
    summarize_finishes,
    summary_from_timing,
)


def test_bucket_ten_second_windows():
    finishes = [(3.0, 100.0), (12.0, 900.0), (19.5, 400.0), (20.0, 50.0)]
    bins = bucket_finishes(finishes, 10.0)
    assert [b["n"] for b in bins] == [1, 2, 1]
    assert bins[1]["n_cap"] == 1


def test_summarize_rate_not_sim_over_wall():
    finishes = [(5.0, 900.0), (15.0, 900.0), (25.0, 100.0)]
    s = summarize_finishes(finishes, wall_s=30.0, slots=2, occupy_s=58.0, interval_s=10.0)
    assert s["n_jobs"] == 3
    assert abs(s["rate"] - 0.1) < 1e-9
    assert s["bins"] == [1, 1, 1]
    assert s["n_cap"] == 2
    assert abs(s["util"] - 58.0 / 60.0) < 1e-9
    line = format_round_stats(s)
    assert "wall=30.0s" in line
    assert "rate=0.10/s" in line
    assert "bin10s=1,1,1" in line


def test_gen_aggregate():
    a = summarize_finishes([(8.0, 900.0)], wall_s=10.0, slots=8, occupy_s=80.0)
    b = summarize_finishes([(4.0, 100.0), (9.0, 200.0)], wall_s=10.0, slots=8, occupy_s=70.0)
    agg: dict = {}
    add_summaries(agg, a)
    add_summaries(agg, b)
    line = format_gen_stats(agg, gen=69)
    assert line.startswith("sim.gen=69")
    assert "cap=1/3" in line
    assert "slots=8" in line


def test_summary_from_kernel_timing():
    timing = {
        "wall_s": 20.0,
        "slots": 8,
        "occupy_s": 150.0,
        "finishes": [(5.1, 80.0), (18.0, 900.0)],
    }
    s = summary_from_timing(timing, interval_s=10)
    assert s["n_jobs"] == 2
    assert s["bins"] == [1, 1]
    assert s["n_cap"] == 1
