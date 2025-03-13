from openpyxl import Workbook
from openpyxl.styles import Alignment


def write_to_excel(metrics: dict,
                   save_path: str) -> None:
    # 创建工作簿和工作表
    wb = Workbook()
    ws = wb.active

    # 写入表头（字典的键作为列标题）
    headers = list(metrics.keys())
    ws.append(headers)

    # 假设所有列表的长度相同，写入每一行数据
    n = len(metrics[headers[0]])
    for i in range(n):
        row_data = [metrics[header][i] for header in headers]
        ws.append(row_data)

    # 创建一个居中对齐的格式
    center_alignment = Alignment(horizontal='center', vertical='center')

    # 为所有单元格设置居中
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = center_alignment

    # 保存为 Excel 文件
    wb.save(save_path)
