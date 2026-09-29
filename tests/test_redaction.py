"""遮盖导出核心逻辑测试（纯标准库，无需启动服务）。"""

import pytest

from app.main import RedactionRequest, redact as redact_endpoint
from app.redaction import MASK_TOKEN, RedactionError, redact_export


def _assert_fragments_on_text(result):
    """不变量：每个片段的新范围内文字是原文可见部分，不含遮盖标记。"""
    for f in result["annotations"]:
        piece = result["text"][f["start"]:f["end"]]
        assert f["start"] < f["end"]
        assert MASK_TOKEN not in piece


def test_single_mask_hit_annotation_not_exported():
    text = "联系张三办理"
    anns = [{"id": "p", "start": 2, "end": 4, "label": "人名"},
            {"id": "a", "start": 0, "end": 2, "label": "普通"}]
    res = redact_export(text, anns, ["人名"])
    assert res["text"] == f"联系{MASK_TOKEN}办理"
    assert res["mask_count"] == 1
    assert [f["source_id"] for f in res["annotations"]] == ["a"]
    assert res["annotations"][0] == {
        "source_id": "a", "label": "普通", "start": 0, "end": 2,
        "fragment_index": 1}
    _assert_fragments_on_text(res)


def test_touching_regions_merge_into_one_token():
    # 两条命中批注端点相接：[2,4) 与 [4,6) 合并为一个遮盖区，只放一个标记
    text = "联系张三本人办理"
    anns = [{"id": "p1", "start": 2, "end": 4, "label": "人名"},
            {"id": "p2", "start": 4, "end": 6, "label": "代号"}]
    res = redact_export(text, anns, ["人名", "代号"])
    assert res["mask_count"] == 1
    assert res["text"] == f"联系{MASK_TOKEN}办理"
    assert res["annotations"] == []


def test_overlapping_regions_merge():
    # 重叠的命中区间 [2,6) 与 [4,8) 合并为 [2,8)
    text = "联系张三本人办理"
    anns = [{"id": "p1", "start": 2, "end": 6, "label": "人名"},
            {"id": "p2", "start": 4, "end": 8, "label": "代号"}]
    res = redact_export(text, anns, ["人名", "代号"])
    assert res["mask_count"] == 1
    assert res["text"] == f"联系{MASK_TOKEN}"


def test_cross_region_annotation_split_in_order():
    # 遮盖区 [2,4)「张三」与 [7,9)「北京」分离；普通批注 [1,12) 跨越两个
    # 遮盖区，切成 3 个可见片段
    text = "联系张三，现居北京海淀。"
    anns = [
        {"id": "p", "start": 2, "end": 4, "label": "人名"},
        {"id": "c", "start": 7, "end": 9, "label": "地名"},
        {"id": "x", "start": 1, "end": 12, "label": "说明"},
    ]
    res = redact_export(text, anns, ["人名", "地名"])
    assert res["mask_count"] == 2
    assert res["text"] == f"联系{MASK_TOKEN}，现居{MASK_TOKEN}海淀。"
    frags = res["annotations"]
    # 只有 x 导出；片段按原文位置排列、序号从 1 开始
    assert [f["source_id"] for f in frags] == ["x", "x", "x"]
    assert [f["fragment_index"] for f in frags] == [1, 2, 3]
    assert [f["label"] for f in frags] == ["说明", "说明", "说明"]
    # x 原文可见部分依次为：[1,2)「系」、[4,7)「，现居」、[9,12)「海淀。」
    pieces = [res["text"][f["start"]:f["end"]] for f in frags]
    assert pieces == ["系", "，现居", "海淀。"]
    _assert_fragments_on_text(res)


def test_fragments_follow_input_order_then_position():
    # 输入顺序故意与位置相反：输出仍按原批注输入顺序，不做全局位置重排
    text = "联系张三，现居北京。"  # 10 码点
    anns = [
        {"id": "late", "start": 9, "end": 10, "label": "标点"},
        {"id": "p", "start": 2, "end": 4, "label": "人名"},
        {"id": "early", "start": 0, "end": 2, "label": "动词"},
    ]
    res = redact_export(text, anns, ["人名"])
    assert [f["source_id"] for f in res["annotations"]] == ["late", "early"]
    _assert_fragments_on_text(res)


def test_fully_covered_annotation_dropped():
    text = "联系张三办理"
    anns = [{"id": "p", "start": 2, "end": 4, "label": "人名"},
            {"id": "inside", "start": 2, "end": 4, "label": "别名"},
            {"id": "strict", "start": 3, "end": 4, "label": "单字"}]
    # inside/strict 虽不在遮盖标签中，但完全落在遮盖区内 → 舍弃
    res = redact_export(text, anns, ["人名"])
    assert res["annotations"] == []


def test_partial_overlap_keeps_visible_edge():
    text = "联系张三办理手续"
    anns = [{"id": "p", "start": 2, "end": 4, "label": "人名"},
            {"id": "a", "start": 0, "end": 3, "label": "词组"},
            {"id": "b", "start": 3, "end": 6, "label": "词组"}]
    res = redact_export(text, anns, ["人名"])
    frags = res["annotations"]
    # a 留 [0,2)「联系」；b 留 [4,6)「办理」
    assert [(f["source_id"], res["text"][f["start"]:f["end"]]) for f in frags] == [
        ("a", "联系"), ("b", "办理")]
    assert [f["fragment_index"] for f in frags] == [1, 1]
    _assert_fragments_on_text(res)


def test_annotation_ending_at_region_start_boundary():
    text = "联系张三办理"
    anns = [{"id": "p", "start": 2, "end": 4, "label": "人名"},
            {"id": "a", "start": 0, "end": 2, "label": "词组"},
            {"id": "b", "start": 4, "end": 6, "label": "词组"}]
    res = redact_export(text, anns, ["人名"])
    assert res["text"] == f"联系{MASK_TOKEN}办理"
    assert [res["text"][f["start"]:f["end"]] for f in res["annotations"]] == [
        "联系", "办理"]
    _assert_fragments_on_text(res)


def test_no_hit_returns_text_and_ranges_unchanged():
    text = "联系张三办理"
    anns = [{"id": "p", "start": 2, "end": 4, "label": "人名"},
            {"id": "a", "start": 0, "end": 2, "label": "普通"}]
    res = redact_export(text, anns, ["地名"])
    assert res["text"] == text
    assert res["mask_count"] == 0
    assert res["annotations"] == [
        {"source_id": "p", "label": "人名", "start": 2, "end": 4,
         "fragment_index": 1},
        {"source_id": "a", "label": "普通", "start": 0, "end": 2,
         "fragment_index": 1},
    ]
    _assert_fragments_on_text(res)


def test_empty_mask_labels_no_hit():
    res = redact_export("abc", [{"id": "a", "start": 0, "end": 1,
                                 "label": "x"}], [])
    assert res["text"] == "abc"
    assert res["annotations"][0]["start"] == 0


def test_codepoint_offsets_with_astral_chars():
    # 🎉 占单码点；遮盖与片段坐标都按码点
    text = "🎉🎉张三🎉🎉"
    anns = [{"id": "p", "start": 2, "end": 4, "label": "人名"},
            {"id": "a", "start": 1, "end": 5, "label": "氛围"}]
    res = redact_export(text, anns, ["人名"])
    assert res["text"] == f"🎉🎉{MASK_TOKEN}🎉🎉"
    frags = res["annotations"]
    # a 切成 [1,2)「🎉」与 [4,5)「🎉」两段
    assert [res["text"][f["start"]:f["end"]] for f in frags] == ["🎉", "🎉"]
    assert [(f["start"], f["end"]) for f in frags] == [(1, 2), (6, 7)]
    assert [f["fragment_index"] for f in frags] == [1, 2]
    _assert_fragments_on_text(res)


def test_mask_token_is_four_codepoints():
    assert len(MASK_TOKEN) == 4


def test_invalid_mask_labels():
    with pytest.raises(RedactionError):
        redact_export("abc", [], [""])
    with pytest.raises(RedactionError):
        redact_export("abc", [], ["人名", "人名"])


# ---------- 接口接线：非法输入 422，不返回部分结果 ----------

def test_endpoint_invalid_interval_returns_422():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        redact_endpoint(RedactionRequest(
            text="ab",
            annotations=[{"id": "a", "start": 0, "end": 9, "label": "人名"}],
            mask_labels=["人名"],
        ))
    assert exc_info.value.status_code == 422


def test_endpoint_duplicate_id_returns_422():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        redact_endpoint(RedactionRequest(
            text="abc",
            annotations=[{"id": "a", "start": 0, "end": 1, "label": "x"},
                         {"id": "a", "start": 1, "end": 2, "label": "y"}],
            mask_labels=[],
        ))
    assert exc_info.value.status_code == 422


def test_endpoint_empty_label_returns_422():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        redact_endpoint(RedactionRequest(
            text="abc",
            annotations=[{"id": "a", "start": 0, "end": 1, "label": ""}],
            mask_labels=[],
        ))
    assert exc_info.value.status_code == 422


def test_endpoint_invalid_mask_labels_422():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        redact_endpoint(RedactionRequest(
            text="abc",
            annotations=[{"id": "a", "start": 0, "end": 1, "label": "x"}],
            mask_labels=[""],
        ))
    assert exc_info.value.status_code == 422
    with pytest.raises(HTTPException) as exc_info:
        redact_endpoint(RedactionRequest(
            text="abc", mask_labels=["x", "x"]))
    assert exc_info.value.status_code == 422


def test_endpoint_ok_shape():
    res = redact_endpoint(RedactionRequest(
        text="联系张三办理",
        annotations=[{"id": "p", "start": 2, "end": 4, "label": "人名"}],
        mask_labels=["人名"],
    ))
    assert set(res) == {"text", "annotations", "mask_count"}
    assert res["text"] == f"联系{MASK_TOKEN}办理"
    assert res["annotations"] == []
    assert res["mask_count"] == 1
