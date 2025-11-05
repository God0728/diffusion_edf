import numpy as np
from scipy.spatial.transform import Rotation

# 输入的变换矩阵
T = np.array([
    [0.8031, -0.5953, -0.0242, -0.0937],
    [0.5945,  0.8034, -0.0350, -0.0752],
    [0.0402,  0.0137,  0.9991,  0.0400],
    [0,       0,       0,       1.0000]
])

# 提取旋转矩阵和平移向量
R = T[:3, :3]  # 旋转部分
t = T[:3, 3]   # 平移部分

# 转换旋转矩阵为四元数
rotation = Rotation.from_matrix(R)
quaternion = rotation.as_quat()  # [x, y, z, w]

print("Position:")
print(f"    {t.tolist()}")
print()
print("Quaternion [x, y, z, w]:")
print(f"    {quaternion.tolist()}")
print()

# 格式化为JSON格式
print("JSON格式:")
print('{')
print('    "position": [')
print(f'        {t[0]:.6f},')
print(f'        {t[1]:.6f},')
print(f'        {t[2]:.6f}')
print('    ],')
print('    "quaternion": [')
print(f'        {quaternion[0]:.6f},')
print(f'        {quaternion[1]:.6f},')
print(f'        {quaternion[2]:.6f},')
print(f'        {quaternion[3]:.6f}')
print('    ]')
print('}')
