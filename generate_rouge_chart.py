import matplotlib.pyplot as plt
import numpy as np

# ROUGE scores from evaluation
rouge_scores = {
    'ROUGE-1': 0.3279,
    'ROUGE-2': 0.1035,
    'ROUGE-L': 0.1714
}

# Extract labels and values
labels = list(rouge_scores.keys())
values = list(rouge_scores.values())
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

# Create figure with appropriate size for report
plt.figure(figsize=(10, 6))

# Create bar chart
bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add title
plt.title('ROUGE Score Comparison - MARL-MDS Model\nMulti-News Test Set Evaluation', 
          fontsize=16, fontweight='bold', pad=20)

# Add labels
plt.xlabel('ROUGE Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=14, fontweight='bold')

# Set y-axis range to show scores clearly
plt.ylim(0, max(values) * 1.2)

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add subtitle with sample information
plt.figtext(0.5, 0.02, 
            'Evaluation: 47 test samples | Time: 397.06 seconds | Dataset: Multi-News Test Set', 
            ha='center', fontsize=11, style='italic')

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the figure
plt.savefig('rouge_scores_bar_chart.png', dpi=300, bbox_inches='tight')
print("Bar chart saved as 'rouge_scores_bar_chart.png'")

# Display the chart
plt.show()
