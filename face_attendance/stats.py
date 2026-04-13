import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("attendance.csv")

# Lấy danh sách tên (không trùng)
names = df["Name"].unique()

# Hiển thị menu
print("===== DANH SÁCH NHÂN VIÊN =====")
for i, name in enumerate(names, start=1):
    print(f"{i}. {name}")

# Nhập lựa chọn
try:
    choice = int(input("Chọn số: "))

    if 1 <= choice <= len(names):
        selected_name = names[choice - 1]

        # Đếm số ngày
        count = len(df[df["Name"] == selected_name])

        print(f"\n✅ {selected_name} đã đi làm {count} ngày")
    else:
        print("❌ Số không hợp lệ!")

except:
    print("❌ Vui lòng nhập số!")