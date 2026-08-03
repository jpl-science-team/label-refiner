import matplotlib.pyplot as plt

# 1. Define the data extracted from image_2b1f22.png
metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
values = [86.0, 72.0, 78.0, 58.5]

# 2. Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white') # Ensure white background

# 3. Create bars with a professional, muted blue color
bar_color = '#2c7fb8'
bars = ax.bar(metrics, values, color=bar_color, width=0.55, zorder=3)

# 4. Add data labels directly on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#333333')

# 5. Customize axes, labels, and title
ax.set_ylim(0, 100) # Set Y-axis from 0 to 100%
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold', color='#444444')
ax.set_title('YOLOv8n-OBB Model Baseline Metrics', 
             fontsize=14, fontweight='bold', color='#222222', pad=20)

# Format tick marks
ax.tick_params(axis='x', labelsize=11, colors='#333333')
ax.tick_params(axis='y', labelsize=11, colors='#333333')

# 6. Add subtle horizontal grid lines for readability
ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#aaaaaa', zorder=0)

# 7. Remove top and right spines (borders) for a modern, clean look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')

# 8. Render the plot
plt.tight_layout()
plt.show()