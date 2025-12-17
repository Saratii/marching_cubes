import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import sys
plt.style.use('dark_background')
data_folder = sys.argv[1] if len(sys.argv) > 1 else 'plots/latest'
if not data_folder.startswith('plots/'):
    data_folder = f'plots/{data_folder}'
throughput_files = sorted(glob.glob(f'{data_folder}/throughput_thread_*.csv'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
queue_size_files = sorted(glob.glob(f'{data_folder}/queue_size_thread_*.csv'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
if not throughput_files or not queue_size_files:
    print(f"Error: No data files found in {data_folder}")
    sys.exit(1)
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(14, 16))
fig.patch.set_facecolor('#1a1a1a')
ax0.set_facecolor('#2a2a2a')
ax1.set_facecolor('#2a2a2a')
ax2.set_facecolor('#2a2a2a')
ax3.set_facecolor('#2a2a2a')
colors = plt.cm.plasma([i / len(throughput_files) for i in range(len(throughput_files))])
all_chunks_per_second = []
all_times = []
all_dfs = []
for i, file in enumerate(throughput_files):
    df = pd.read_csv(file)
    all_dfs.append(df)
if all_dfs:
    time_sum_dict = {}
    for df in all_dfs:
        for _, row in df.iterrows():
            t_rounded = round(row['time_seconds'])
            if t_rounded not in time_sum_dict:
                time_sum_dict[t_rounded] = []
            time_sum_dict[t_rounded].append(row['chunks_per_second'])
    times = sorted(time_sum_dict.keys())
    totals = [sum(time_sum_dict[t]) for t in times]
    ax0.plot(times, totals, color='#00d4ff', linestyle='-', linewidth=2, label='Total Chunks/sec', alpha=0.9)
    overall_avg = np.mean(totals)
    ax0.axhline(y=overall_avg, color='#00ff00', linestyle='--', linewidth=2, label=f'Average: {overall_avg:.2f}', alpha=0.7)
    x_min, x_max = ax0.get_xlim()
    ax0.text(x_min, overall_avg, f'{overall_avg:.1f}', color='#00ff00', fontsize=10, ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#00ff00', alpha=0.8))
    ax0.text(x_max, overall_avg, f'{overall_avg:.1f}', color='#00ff00', fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#00ff00', alpha=0.8))
ax0.set_xlabel('Time (seconds)', fontsize=12, color='white')
ax0.set_ylabel('Total Chunks per Second', fontsize=12, color='white')
ax0.set_title('Total Chunk Generation Throughput (All Threads Combined)', fontsize=14, color='white', pad=20)
ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, facecolor='#2a2a2a', edgecolor='#444444')
ax0.grid(True, alpha=0.2, color='white')
ax0.tick_params(colors='white')
for i, file in enumerate(throughput_files):
    thread_num = int(file.split('_')[-1].split('.')[0])
    df = pd.read_csv(file)
    ax1.plot(df['time_seconds'], df['chunks_per_second'], label=f'Thread {thread_num}', color=colors[i], linewidth=1.5, alpha=0.8)
    all_chunks_per_second.extend(df['chunks_per_second'].values)
if all_chunks_per_second:
    overall_avg = np.mean(all_chunks_per_second)
    ax1.axhline(y=overall_avg, color='#00ff00', linestyle='--', linewidth=2, label=f'Overall Average: {overall_avg:.2f}', alpha=0.7)
    x_min, x_max = ax1.get_xlim()
    ax1.text(x_min, overall_avg, f'{overall_avg:.1f}', color='#00ff00', fontsize=10, ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#00ff00', alpha=0.8))
    ax1.text(x_max, overall_avg, f'{overall_avg:.1f}', color='#00ff00', fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#00ff00', alpha=0.8))
ax1.set_xlabel('Time (seconds)', fontsize=12, color='white')
ax1.set_ylabel('Chunks per Second', fontsize=12, color='white')
ax1.set_title('Chunk Generation Throughput per Thread', fontsize=14, color='white', pad=20)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, facecolor='#2a2a2a', edgecolor='#444444')
ax1.grid(True, alpha=0.2, color='white')
ax1.tick_params(colors='white')
all_queue_sizes = []
queue_dfs = []
for i, file in enumerate(queue_size_files):
    thread_num = int(file.split('_')[-1].split('.')[0])
    df = pd.read_csv(file)
    ax2.plot(df['time_seconds'], df['queue_size'], label=f'Thread {thread_num}', color=colors[i], linewidth=1.5, alpha=0.8)
    all_queue_sizes.extend(df['queue_size'].values)
    queue_dfs.append(df)
if all_queue_sizes:
    overall_avg = np.mean(all_queue_sizes)
    ax2.axhline(y=overall_avg, color='#00ff00', linestyle='--', linewidth=2, label=f'Overall Average: {overall_avg:.2f}', alpha=0.7)
    x_min, x_max = ax2.get_xlim()
    ax2.text(x_min, overall_avg, f'{overall_avg:.1f}', color='#00ff00', fontsize=10, ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#00ff00', alpha=0.8))
    ax2.text(x_max, overall_avg, f'{overall_avg:.1f}', color='#00ff00', fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#00ff00', alpha=0.8))
ax2.set_xlabel('Time (seconds)', fontsize=12, color='white')
ax2.set_ylabel('Priority Queue Size', fontsize=12, color='white')
ax2.set_title('Priority Queue Size per Thread', fontsize=14, color='white', pad=20)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, facecolor='#2a2a2a', edgecolor='#444444')
ax2.grid(True, alpha=0.2, color='white')
ax2.tick_params(colors='white')
all_entity_chunks = []
entity_dfs = []
for i, file in enumerate(throughput_files):
    thread_num = int(file.split('_')[-1].split('.')[0])
    df = pd.read_csv(file)
    ax3.plot(df['time_seconds'], df['entity_chunks_per_second'], label=f'Thread {thread_num}', color=colors[i], linewidth=1.5, alpha=0.8)
    all_entity_chunks.extend(df['entity_chunks_per_second'].values)
    entity_dfs.append(df)
if all_entity_chunks:
    overall_avg = np.mean(all_entity_chunks)
    ax3.axhline(y=overall_avg, color='#00ff00', linestyle='--', linewidth=2, label=f'Overall Average: {overall_avg:.2f}', alpha=0.7)
    x_min, x_max = ax3.get_xlim()
    ax3.text(x_min, overall_avg, f'{overall_avg:.1f}', color='#00ff00', fontsize=10, ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#00ff00', alpha=0.8))
    ax3.text(x_max, overall_avg, f'{overall_avg:.1f}', color='#00ff00', fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#00ff00', alpha=0.8))
ax3.set_xlabel('Time (seconds)', fontsize=12, color='white')
ax3.set_ylabel('Entity Chunks per Second', fontsize=12, color='white')
ax3.set_title('Chunks with Entities (Mesh) Generation Rate per Thread', fontsize=14, color='white', pad=20)
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, facecolor='#2a2a2a', edgecolor='#444444')
ax3.grid(True, alpha=0.2, color='white')
ax3.tick_params(colors='white')
plt.tight_layout()
os.makedirs(data_folder, exist_ok=True)
output_file = f'{data_folder}/chunk_metrics.png'
plt.savefig(output_file, dpi=300, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
print(f"Generated {output_file}")
plt.show()