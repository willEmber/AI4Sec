"""
测试期刊等级查询工具
使用 ieee_result.json 中的 publication_title 字段作为测试数据
"""

import json
import os
import sys

# 将当前目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from publication_rank import EasyScholarClient, _validate_publication_name, _normalize_publication_name


def test_input_validation():
    """测试输入校验和格式化"""
    print("=" * 60)
    print("测试1: 输入校验和格式化")
    print("=" * 60)

    # 正常输入
    assert _normalize_publication_name("  IEEE Transactions  ") == "IEEE Transactions"
    print("  ✓ 首尾空白去除正常")

    assert _normalize_publication_name("IEEE  Transactions   on  AI") == "IEEE Transactions on AI"
    print("  ✓ 多余空格合并正常")

    assert _normalize_publication_name("IEEE\t\nTransactions") == "IEEE Transactions"
    print("  ✓ 特殊空白字符处理正常")

    # 空输入
    try:
        _validate_publication_name("")
        assert False, "应当抛出 ValueError"
    except ValueError as e:
        print(f"  ✓ 空字符串校验正常: {e}")

    try:
        _validate_publication_name("   ")
        assert False, "应当抛出 ValueError"
    except ValueError as e:
        print(f"  ✓ 纯空白校验正常: {e}")

    # 超长输入
    try:
        _validate_publication_name("A" * 501)
        assert False, "应当抛出 ValueError"
    except ValueError as e:
        print(f"  ✓ 超长字符串校验正常: {e}")

    # 非字符串输入
    try:
        _validate_publication_name(123)
        assert False, "应当抛出 ValueError"
    except ValueError as e:
        print(f"  ✓ 非字符串类型校验正常: {e}")

    print("\n输入校验测试全部通过！\n")


def test_query_with_ieee_data():
    """使用 ieee_result.json 的数据进行实际 API 查询测试"""
    print("=" * 60)
    print("测试2: 使用 ieee_result.json 数据查询期刊等级")
    print("=" * 60)

    # 加载测试数据
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ieee_result.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取所有不重复的 publication_title
    titles = list(set(
        article.get("publication_title", "")
        for article in data["articles"]
        if article.get("publication_title")
    ))
    titles.sort()

    print(f"\n共找到 {len(titles)} 个不重复期刊名\n")

    # 使用客户端批量查询
    with EasyScholarClient() as client:
        results = client.query_batch(titles)

    # 汇总结果
    print("\n" + "=" * 60)
    print("查询结果汇总")
    print("=" * 60)
    print(f"{'期刊名':<60} {'SCI':<10} {'CCF':<10}")
    print("-" * 80)

    success_count = 0
    sci_count = 0
    ccf_count = 0

    for r in results:
        if r.success:
            success_count += 1
            sci_display = r.sci or "-"
            ccf_display = r.ccf or "-"
            if r.sci:
                sci_count += 1
            if r.ccf:
                ccf_count += 1
            # 截断过长的名称
            name_display = r.name[:57] + "..." if len(r.name) > 60 else r.name
            print(f"{name_display:<60} {sci_display:<10} {ccf_display:<10}")
        else:
            name_display = r.name[:57] + "..." if len(r.name) > 60 else r.name
            print(f"{name_display:<60} {'错误: ' + r.error}")

    print("-" * 80)
    print(f"查询成功: {success_count}/{len(results)}")
    print(f"有 SCI 等级: {sci_count}")
    print(f"有 CCF 等级: {ccf_count}")

    return results


def test_known_result():
    """
    使用已知结果验证查询逻辑。
    test_easyscholar.json 是 "IEEE Transactions on Medical Imaging" 的预期返回，
    其中 sci=Q1, ccf=B
    """
    print("\n" + "=" * 60)
    print("测试3: 已知结果验证 (IEEE Transactions on Medical Imaging)")
    print("=" * 60)

    with EasyScholarClient() as client:
        result = client.query("IEEE Transactions on Medical Imaging")

    print(f"  期刊: {result.name}")
    print(f"  成功: {result.success}")
    print(f"  SCI:  {result.sci}")
    print(f"  CCF:  {result.ccf}")

    # 根据 test_easyscholar.json 的数据，预期 sci=Q1, ccf=B
    if result.success:
        assert result.sci == "Q1", f"预期 SCI=Q1，实际 SCI={result.sci}"
        assert result.ccf == "B", f"预期 CCF=B，实际 CCF={result.ccf}"
        print("\n  ✓ 已知结果验证通过！SCI=Q1, CCF=B 与预期一致")
    else:
        print(f"\n  ✗ 查询失败: {result.error}")


if __name__ == "__main__":
    test_input_validation()
    test_known_result()
    test_query_with_ieee_data()
    print("\n所有测试完成！")
