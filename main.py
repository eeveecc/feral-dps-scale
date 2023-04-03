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
    "猫德": (255/255, 165/255, 0/255),
    "火法": (173/255, 216/255, 230/255),
    "冰迪凯": (139/255, 0/255, 0/255),
    "冰法": (173/255, 216/255, 230/255),
    "狂暴战": (139/255, 69/255, 19/255),
    "射击猎": (0/255, 128/255, 0/255),
    "惩戒骑": (255/255, 192/255, 203/255),
    "暗牧": (128/255, 128/255, 128/255),
    "生存猎": (0/255, 128/255, 0/255),
    "邪迪凯": (139/255, 0/255, 0/255)
}


def read_excel_data(file_path):
    df = pd.read_excel(file_path, index_col=0)
    data = {}
    for class_name in df.index:
        item_levels = df.columns.values
        dps_values = df.loc[class_name].values
        data[class_name] = np.array([item_levels, dps_values])
    return data

target_data = "cat_vs_melee"
data = read_excel_data('./' + target_data + '.xlsx')

# Prediction function (Modify this according to the actual prediction model)
from sklearn.linear_model import LinearRegression

def predict_dps(item_levels, class_data):
    X = class_data[0].reshape(-1, 1)
    y = class_data[1]
    model = LinearRegression().fit(X, y)
    predicted_dps = model.predict(item_levels.reshape(-1, 1))
    return predicted_dps

# Set up the plot
fig, ax = plt.subplots(figsize=(24, 10))
fig.subplots_adjust(right=0.78)
ax.set_title("基于P1和P2数据的DPS成长曲线", fontsize=16, fontweight='bold', pad=40)
ax.text(0.5, 1.05, "90-120秒战斗,基于P1帕奇维克和P2的托里姆", ha='center', va='bottom', transform=ax.transAxes, fontsize=10)
ax.set_xlabel("装备等级", fontsize=14, labelpad=10)
ax.set_ylabel("DPS", fontsize=14, labelpad=10)
ax.set_xlim(200, 260)  # Adjust the x-axis range according to your data
ax.set_ylim(5000, 16000)  # Adjust the y-axis range according to your data
ax.grid(True)  # Enable grid


lines = {class_name: ax.plot([], [], label=class_name, marker='o', markersize=4, color=class_colors[class_name])[0] for class_name in data}
predictions = {class_name: ax.plot([], [], linestyle="--", linewidth=1, color=class_colors[class_name])[0] for class_name in data}

ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fancybox=True, ncol=1, fontsize=10) # Move legend to right

def update(frame):
    class_name = list(data.keys())[frame % len(data)]
    class_data = data[class_name]

    if frame <= len(data):
        # Update the original data plot
        lines[class_name].set_data(class_data[0], class_data[1])

    elif frame <= len(data)*2:  
        # Update the prediction curve
        extended_item_levels = np.linspace(200, 277, 2000)  # Adjust the range according to your data
        predicted_dps = predict_dps(extended_item_levels, class_data)
        predictions[class_name].set_data(extended_item_levels, predicted_dps)
    
    else:
        ax.scatter([217, 241, 255, 274], [8011, 10353, 11759, 14582], marker='*', s=350, color='orange', label='Extra Points')
        ax.annotate('P1模拟 - 8011', xy=(217, 8011), xytext=(212, 6070), fontsize=14, color='white', arrowprops=dict(arrowstyle="->", color='white'))
        ax.annotate('P2模拟 - 10353', xy=(241, 10353), xytext=(234, 7353), fontsize=14, color='white', arrowprops=dict(arrowstyle="->", color='white'))
        ax.annotate('P3模拟 - 11759', xy=(255, 11759), xytext=(252, 8759), fontsize=14, color='white', arrowprops=dict(arrowstyle="->", color='white'))
        ax.annotate('P4模拟(TBD) - 14582', xy=(274, 14582), xytext=(265, 11582), fontsize=14, color='white', arrowprops=dict(arrowstyle="->", color='white'))

    return lines.values(), predictions.values()

ani = FuncAnimation(fig, update, frames=len(data)*2+15)
ani.save(target_data + '.gif', writer='imagemagick', fps=1, dpi=150)
