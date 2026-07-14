# @Function  : 字符串工具——CJK 感知的对齐 / 表格格式化
def get_display_width(text: str) -> int:
    """
    计算字符串的实际显示宽度。

    Examples:
        >>> get_display_width("hello")
        5
        >>> get_display_width("你好")
        4
        >>> get_display_width("hi你好")
        6
    """
    return sum(2 if _is_wide_char(c) else 1 for c in text)


def _is_wide_char(char: str) -> bool:
    """判断单字符是否为宽字符(占 2 个显示宽度)"""
    if '\u4e00' <= char <= '\u9fff':  # 中日韩统一表意文字
        return True
    if '\u3000' <= char <= '\u303f':  # 中日韩标点符号
        return True
    if '\uff00' <= char <= '\uffef':  # 全角 ASCII
        return True
    if '\u3400' <= char <= '\u4dbf':  # 中日韩扩展 A
        return True
    if '\u3040' <= char <= '\u309f':  # 平假名
        return True
    if '\u30a0' <= char <= '\u30ff':  # 片假名
        return True
    if '\uac00' <= char <= '\ud7af':  # 韩文音节
        return True
    return False


def pad_to_width(text: str, width: int, align: str = 'left') -> str:
    """
    将字符串填充到指定显示宽度。

    Args:
        text: 输入字符串
        width: 目标显示宽度
        align: 'left' / 'right' / 'center'

    Examples:
        >>> pad_to_width("hello", 10)
        'hello     '
        >>> pad_to_width("你好", 10)
        '你好      '
    """
    current = get_display_width(text)
    padding = width - current
    if padding <= 0:
        return text

    if align == 'right':
        return ' ' * padding + text
    elif align == 'center':
        left = padding // 2
        right = padding - left
        return ' ' * left + text + ' ' * right
    else:
        return text + ' ' * padding


def format_table_row(columns: list, widths: list, aligns: list = None) -> str:
    """格式化表格一行"""
    if aligns is None:
        aligns = ['left'] * len(columns)
    assert len(columns) == len(widths) == len(aligns), "列数 / 宽度 / 对齐数必须一致"

    parts = [
        pad_to_width(str(col), w, a)
        for col, w, a in zip(columns, widths, aligns)
    ]
    return ' '.join(parts)


def format_table_separator(widths: list, char: str = '-') -> str:
    """生成与表格列宽匹配的分隔线"""
    total = sum(widths) + len(widths) - 1
    return char * total


if __name__ == "__main__":
    print("=== string_utils 自测 ===\n")

    cases = [("hello", 5), ("你好", 4), ("hi你好", 6), ("类别名称", 8)]
    for text, expected in cases:
        actual = get_display_width(text)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} '{text}' → {actual} (期望 {expected})")

    print("\n=== 表格对齐示例 ===")
    widths = [12, 20, 6]
    aligns = ['left', 'left', 'left']
    print(format_table_row(['ID', '类别名称', '数量'], widths, aligns))
    print(format_table_separator(widths))
    print(format_table_row(['0', 'head', '60'], widths, aligns))
    print(format_table_row(['1', 'ordinary_clothes', '59'], widths, aligns))
    print(format_table_row(['2', '反光衣', '116'], widths, aligns))