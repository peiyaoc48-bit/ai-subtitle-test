"""预处理单元测试：不调用大模型，CI 中也会运行。"""
import pytest

from app.translator import collapse_repeats


@pytest.mark.parametrize(
    "text, expected",
    [
        ("呜" * 62, "呜呜呜"),
        ("呜" * 62 + "李娜真坏", "呜呜呜李娜真坏"),
        ("不去了！！！！", "不去了！！！！"),      # 4 个不压缩
        ("不去了！！！！！！", "不去了！！！"),    # 6 个压缩成 3 个
        ("哈哈哈", "哈哈哈"),                      # 正常重复不受影响
        ("林晓，你今天怎么来得这么早？", "林晓，你今天怎么来得这么早？"),
    ],
)
def test_collapse_repeats(text, expected):
    assert collapse_repeats(text) == expected
