"""评测体系 2.0 的单元测试：解析、计分、统计方法。不调用大模型。"""
import pytest

from eval2 import mqm, stats


def test_mqm_score_weights():
    r = mqm.parse('{"errors": [{"severity": "major", "category": "accuracy/omission"},'
                  ' {"severity": "minor", "category": "style"}, {"severity": "critical", "category": "accuracy"}]}')
    assert r.score == -(5 + 1 + 25) and r.worst == "critical"
    assert r.errors[0]["category"] == "accuracy"   # 子类被规范化到大类


def test_mqm_no_error_and_noise():
    assert mqm.parse('{"errors": []}').score == 0
    r = mqm.parse('{"errors": [{"severity": "no-error"}, {"severity": "MINOR", "category": "weird"}]}')
    assert r.score == -1 and r.errors[0]["category"] == "other" and r.worst == "minor"


@pytest.mark.parametrize("bad", ["not json", '{"foo": 1}', "[]"])
def test_mqm_parse_failure_is_recorded_not_raised(bad):
    r = mqm.parse(bad)
    assert r.parse_error and r.score == 0


def test_prompt_includes_context_only_when_given():
    assert "上文" not in mqm.build_prompt("你好", "Hi")
    assert "我妈" in mqm.build_prompt("说不回来了", "She said", ["我妈来电话了"])


def test_bootstrap_ci_contains_mean_and_narrows_with_data():
    m, lo, hi = stats.bootstrap_ci([0, 1] * 10)
    assert lo <= m <= hi
    _, lo2, hi2 = stats.bootstrap_ci([0, 1] * 200)
    assert hi2 - lo2 < hi - lo


def test_paired_bootstrap_detects_clear_and_no_difference():
    a = [-5.0] * 50
    assert stats.paired_bootstrap(a, [0.0] * 50) < 0.05          # B 明显更好
    assert stats.paired_bootstrap(a, a) == 1.0                    # 完全相同：不显著
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1], [1, 2])


def test_weighted_kappa():
    assert stats.weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3], 4) == pytest.approx(1.0)
    assert stats.weighted_kappa([0, 1, 2, 3], [3, 2, 1, 0], 4) < 0
    near = stats.weighted_kappa([0, 1, 2, 3, 0, 1], [0, 1, 2, 2, 0, 1], 4)
    far = stats.weighted_kappa([0, 1, 2, 3, 0, 1], [0, 1, 2, 0, 0, 1], 4)
    assert near > far                                              # 差一级比差三级惩罚轻


def test_spearman_with_ties():
    assert stats.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert stats.spearman([0, 0, -1, -5], [0, 0, -1, -6]) == pytest.approx(1.0)
