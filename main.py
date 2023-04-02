import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import pandas as pd

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.style.use('dark_background')

class_colors = {
    "痛苦术": (128/255, 0/255, 128/255),
    "奥法": (173/255, 216/255, 230/255),
    "武器战": (139/255, 69/255, 19/255),
    "刺杀贼": (255/255, 255/255, 0/255),
    "鸟德": (255/255, 165/255, 0/255),
    "野兽猎": (0/255, 128/255, 0/255),
    "战斗贼": (255/255, 255/255, 0/255),
    "恶魔术": (128/255, 0/255, 128/255),
    "毁灭术": (128/255, 0/255, 128/255),
    "元素萨": (0/255, 0/255, 139/255),
    "增强萨": (0/255, 0/255, 139/255),
    "猫德(实战)": (255/255, 165/255, 0/255),
    "火法": (173/255, 216/255, 230/255),
    "冰迪凯": (139/255, 0/255, 0/255),
    "冰法": (173/255, 216/255, 230/255),
    "狂暴战": (139/255, 69/255, 19/255),
    "射击猎": (0/255, 128/255, 0/255),
    "惩戒骑": (255/255, 192/255, 203/255),
    "暗牧": (128/255, 128/255, 128/255),
    "生存猎": (0/255, 128/255, 0/255),
    "邪迪凯": (139/255, 0/255, 0/255),
    "猫德(模拟)": (255/255, 165/255, 0/255),
}


def read_excel_data(file_path):
    df = pd.read_excel(file_path, index_col=0)
    data = {}
    for class_name in df.index:
        item_levels = df.columns.values
        dps_values = df.loc[class_name].values
        data[class_name] = np.array([item_levels, dps_values])
    return data

data = read_excel_data('./cat_vs_s_tier.xlsx')

# Prediction function (Modify this according to the actual prediction model)
def predict_dps(item_levels, class_data):
    # Here, we use a simple linear regression for demonstration purposes
    coefficients = np.polyfit(class_data[0], class_data[1], 1)
    return np.polyval(coefficients, item_levels)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(right=0.78)
ax.set_title("猫德DPS成长曲线", fontsize=16, fontweight='bold')
ax.set_xlabel("装备等级", fontsize=14)
ax.set_ylabel("DPS", fontsize=14)
ax.set_xlim(200, 260)  # Adjust the x-axis range according to your data
ax.set_ylim(5000, 13500)  # Adjust the y-axis range according to your data
ax.grid(True)  # Enable grid


lines = {class_name: ax.plot([], [], label=class_name, marker='o', markersize=8, color=class_colors[class_name])[0] for class_name in data}
predictions = {class_name: ax.plot([], [], linestyle="--", linewidth=2, color=class_colors[class_name])[0] for class_name in data}

ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fancybox=True, ncol=1, fontsize=10) # Move legend to right

def update(frame):
    class_name = list(data.keys())[frame % len(data)]
    class_data = data[class_name]

    # Update the original data plot
    lines[class_name].set_data(class_data[0], class_data[1])

    # Update the prediction curve
    extended_item_levels = np.linspace(50, 250, 100)  # Adjust the range according to your data
    predicted_dps = predict_dps(extended_item_levels, class_data)
    predictions[class_name].set_data(extended_item_levels, predicted_dps)

    return lines.values(), predictions.values()

ani = FuncAnimation(fig, update, frames=len(data), interval=10000)
ani.save('dps_animation.gif', writer='imagemagick', fps=1, dpi=150)
