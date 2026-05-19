import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

def parse_units(value):
    """Converts strings like '302.7MiB', '100.36%', '1.2GiB' to float."""
    if pd.isna(value) or value == 'N/A' or value == '':
        return 0.0
    
    # Remove percentage sign
    value = str(value).replace('%', '').strip()
    
    # Check for memory units
    match = re.match(r"([0-9\.]+)\s*([a-zA-Z]*)", value)
    if not match:
        try:
            return float(value)
        except:
            return 0.0
            
    num, unit = match.groups()
    num = float(num)
    unit = unit.lower()
    
    if 'g' in unit:
        return num * 1024
    if 'k' in unit:
        return num / 1024
    if 'b' in unit and 'm' not in unit:
        return num / (1024 * 1024)
    
    return num # Default is MiB

def main():
    print("Loading data...")

    # Load combined CSV and split into docker_df + gpu_df
    combined_df = pd.read_csv('stats_log.csv')

    # --- docker_df: original docker columns ---
    docker_cols = ['Timestamp', 'Container', 'CPU_Perc', 'Mem_Usage', 'Mem_Limit',
                   'Mem_Perc', 'Net_Input', 'Net_Output', 'Block_Input', 'Block_Output']
    docker_df = combined_df[docker_cols].copy()

    gpu_df = combined_df[combined_df['Container'] == 'vllm-server'][
        ['Timestamp', 'GPU_Util_Max', 'GPU_Mem_Used']
    ].copy()

    gpu_df['GPU_Util_Perc'] = gpu_df['GPU_Util_Max'].apply(parse_units)
    gpu_df['GPU_Mem_Used_MiB'] = gpu_df['GPU_Mem_Used'].apply(parse_units)
    gpu_df = gpu_df.drop(columns=['GPU_Util_Max', 'GPU_Mem_Used'])

    gpu_df['Timestamp'] = pd.to_datetime(gpu_df['Timestamp'])

    # Convert timestamps
    docker_df['Timestamp'] = pd.to_datetime(docker_df['Timestamp'])

    # Calculate elapsed time in seconds
    start_time = min(docker_df['Timestamp'].min(), gpu_df['Timestamp'].min())
    docker_df['Elapsed_Time'] = (docker_df['Timestamp'] - start_time).dt.total_seconds()
    gpu_df['Elapsed_Time'] = (gpu_df['Timestamp'] - start_time).dt.total_seconds()

    # Load LangSmith stats to determine initialization and LLM inferences
    import glob
    import datetime
    langsmith_files = glob.glob('./outputs/*_langsmith_stats.csv')
    init_end_time_sec = None
    workload_end_time_sec = None
    llm_intervals = []

    if langsmith_files:
        try:
            ls_df = pd.read_csv(langsmith_files[0])
            ls_df['Start Time'] = pd.to_datetime(ls_df['Start Time'])
            ls_df['End Time'] = pd.to_datetime(ls_df['End Time'])
            
            local_tz = datetime.datetime.now().astimezone().tzinfo
            ls_df['Start Time'] = ls_df['Start Time'].dt.tz_convert(local_tz).dt.tz_localize(None)
            ls_df['End Time'] = ls_df['End Time'].dt.tz_convert(local_tz).dt.tz_localize(None)
            
            # Filter to current run
            ls_df = ls_df[ls_df['Start Time'] >= start_time]
            
            if not ls_df.empty:
                init_end_time_sec_orig = (ls_df['Start Time'].min() - start_time).total_seconds()
                
                # Rebase elapsed times so that 0 is the end of initialization
                docker_df['Elapsed_Time'] -= init_end_time_sec_orig
                gpu_df['Elapsed_Time'] -= init_end_time_sec_orig
                
                new_start_time = ls_df['Start Time'].min()
                init_end_time_sec = 0
                workload_end_time_sec = (ls_df['End Time'].max() - new_start_time).total_seconds()
                
                # Extract LLM inference intervals and compute concurrency
                events = []
                llm_df = ls_df[ls_df['Run Type'] == 'llm']
                for _, row in llm_df.iterrows():
                    if pd.notna(row['End Time']):
                        start_sec = (row['Start Time'] - new_start_time).total_seconds()
                        end_sec = (row['End Time'] - new_start_time).total_seconds()
                        events.append((start_sec, 'start'))
                        events.append((end_sec, 'end'))
                
                events.sort(key=lambda x: (x[0], 1 if x[1] == 'end' else 0))
                
                concurrent_calls = 0
                last_time = None
                for time, event_type in events:
                    if last_time is not None and time > last_time and concurrent_calls > 0:
                        llm_intervals.append((last_time, time, concurrent_calls))
                    
                    if event_type == 'start':
                        concurrent_calls += 1
                    else:
                        concurrent_calls -= 1
                    
                    last_time = time
        except Exception as e:
            print(f"Error processing langsmith stats: {e}")

    # Clean Docker and GPU data
    docker_df['Container'] = docker_df['Container'].replace('gpt-researcher-cli', 'gpt-researcher')
    docker_df['Container'] = docker_df['Container'].replace('mswa-bench', 'mini-swe-agent')
    docker_df['CPU_Perc_Val'] = docker_df['CPU_Perc'].apply(parse_units)
    docker_df['Mem_Usage_GiB'] = docker_df['Mem_Usage'].apply(parse_units) / 1024
    docker_df['Net_Input_GiB'] = docker_df['Net_Input'].apply(parse_units) / 1024
    docker_df['Net_Output_GiB'] = docker_df['Net_Output'].apply(parse_units) / 1024
    docker_df['Block_Input_GiB'] = docker_df['Block_Input'].apply(parse_units) / 1024
    docker_df['Block_Output_GiB'] = docker_df['Block_Output'].apply(parse_units) / 1024

    gpu_df['GPU_Mem_Used_GiB'] = pd.to_numeric(gpu_df['GPU_Mem_Used_MiB'], errors='coerce').fillna(0) / 1024
    gpu_df['GPU_Util_Perc'] = pd.to_numeric(gpu_df['GPU_Util_Perc'], errors='coerce').fillna(0)
    
    docker_df = docker_df.sort_values(['Container', 'Timestamp']).reset_index(drop=True)

    # Per-container deltas
    MIB = 1024 ** 2  # 1 MiB = 1,048,576 bytes
    docker_df['dt_s'] = docker_df.groupby('Container')['Timestamp'].diff().dt.total_seconds()

    for col in ['Net_Input', 'Net_Output']:
        delta_bytes = docker_df.groupby('Container')[col + '_GiB'].diff()
        docker_df[col + '_MiB_s'] = (delta_bytes * 1024) / docker_df['dt_s']
        # If counter resets (container restart), delta goes negative — clamp to NaN
        docker_df.loc[docker_df[col + '_MiB_s'] < 0, col + '_MiB_s'] = pd.NA


    import matplotlib.cm as cm
    containers = docker_df['Container'].unique()
    num_entities = len(containers) + 4
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / max(1, num_entities - 1)) for i in range(num_entities)]
    
    init_line_color = colors[0]
    llm_bg_color = colors[-1]
    gpu_util_color = colors[-3]
    gpu_mem_color = colors[-2]
    
    container_colors = {c: colors[i+1] for i, c in enumerate(containers)}

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(4, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    def add_overlays(ax):
        if init_end_time_sec is not None:
            ax.axvline(x=init_end_time_sec, color=init_line_color, linestyle='--', linewidth=1.5, label='_nolegend_')
        
        added_llm_label = False
        for start_sec, end_sec, concurrency in llm_intervals:
            label = '_nolegend_'
            if not added_llm_label:
                label = 'LLM Inference'
                added_llm_label = True
            
            calc_alpha = min(0.1 + 0.15 * concurrency, 0.8)
            ax.axvspan(start_sec, end_sec, color=llm_bg_color, alpha=calc_alpha, label=label)

    # CPU Utilization
    for container in containers:
        data = docker_df[docker_df['Container'] == container]
        axes[0].plot(data['Elapsed_Time'], data['CPU_Perc_Val'], label=container, color=container_colors[container])
    axes[0].set_ylabel('CPU Utilization (%)')
    axes[0].set_title('CPU Utilization vs Time')
    add_overlays(axes[0])
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # System Memory Usage
    for container in containers:
        data = docker_df[docker_df['Container'] == container]
        axes[1].plot(data['Elapsed_Time'], data['Mem_Usage_GiB'], label=container, color=container_colors[container])
    axes[1].set_ylabel('Memory Usage (GiB)')
    axes[1].set_title('System Memory Usage vs Time')
    add_overlays(axes[1])
    #axes[1].legend(loc='center right')
    axes[1].grid(True, alpha=0.3)

    # GPU Utilization
    if not gpu_df.empty:
        axes[2].plot(gpu_df['Elapsed_Time'], gpu_df['GPU_Util_Perc'], label='vllm-server', color=container_colors['vllm-server'])
    axes[2].set_ylabel('GPU Utilization (%)')
    axes[2].set_title('GPU Utilization vs Time')
    add_overlays(axes[2])
    #axes[2].legend(loc='center right')
    axes[2].grid(True, alpha=0.3)

    # GPU Memory Usage
#    if not gpu_df.empty:
#        axes[3].plot(gpu_df['Elapsed_Time'], gpu_df['GPU_Mem_Used_GiB'], label='Total GPU Memory Used', color=gpu_mem_color)
#    axes[3].set_ylabel('GPU Memory Used (GiB)')
#    axes[3].set_title('GPU Memory Usage vs Time')
#    add_overlays(axes[3])
#    axes[3].legend(loc='center right')
#    axes[3].grid(True, alpha=0.3)

    # Network I/O
    for i, container in enumerate(containers):
        if 'vllm' in container:
            continue
        data = docker_df[docker_df['Container'] == container]
        color_in = container_colors[container]
        color_out = colors[(i + 5) % len(colors)]
        axes[3].plot(data['Elapsed_Time'], data['Net_Input_MiB_s']+data['Net_Output_MiB_s'], label=container, color=container_colors[container])
        #axes[3].plot(data['Elapsed_Time'], data['Net_Output_GiB'], label=f"{container} Out", linestyle='-', color=color_out)
    axes[3].set_ylabel('Network I/O (MiB/s)')
    axes[3].set_title('Network I/O vs Time')
    #axes[3].set_yscale('log')
    add_overlays(axes[3])
    #axes[3].legend(loc='center right')
    axes[3].grid(True, alpha=0.3)

    # Block I/O
#    for i, container in enumerate(containers):
#        if 'gpt-researcher' not in container:
#            continue
#        data = docker_df[docker_df['Container'] == container]
#        color_in = container_colors[container]
#        color_out = colors[(i + 5) % len(colors)]
#        axes[5].plot(data['Elapsed_Time'], data['Block_Input_GiB'], label=f"{container} Read", color=color_in)
#        axes[5].plot(data['Elapsed_Time'], data['Block_Output_GiB'], label=f"{container} Write", linestyle='-', color=color_out)
#    axes[5].set_ylabel('Block I/O (GiB)')
#    axes[5].set_title('Block I/O vs Time')
#    add_overlays(axes[5])
#    axes[5].legend(loc='center right')
#    axes[5].grid(True, alpha=0.3)

    # Add x-axis label to all subplots
    for ax in axes:
        ax.set_xlabel('Time (seconds)')
        ax.tick_params(labelbottom=True)
        if init_end_time_sec is not None:
            if workload_end_time_sec is not None and not pd.isna(workload_end_time_sec):
                ax.set_xlim(left=init_end_time_sec, right=workload_end_time_sec)
            else:
                ax.set_xlim(left=init_end_time_sec)


    #lines_1, labels_1 = axes[0].get_legend_handles_labels()
    #fig.legend(lines_1, labels_1, loc='lower right', ncol=2)

    plt.tight_layout()
    output_file = './outputs/resource_timeline_cropped.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
