import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_audio_duration(file_path):
    """Get duration of audio file in seconds"""
    try:
        # Get duration without loading the full audio (faster)
        duration = librosa.get_duration(filename=file_path)
        return duration
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_dataset_duration(dataset_path, output_dir="audio_analysis"):
    """Analyze audio duration distribution in the dataset"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define dataset splits
    splits = ['train', 'dev', 'eval']
    all_durations = {}
    
    print("🎵 Analyzing Audio Duration Distribution")
    print("=" * 50)
    
    for split in splits:
        split_path = os.path.join(dataset_path, f"ASVspoof2019_LA_{split}")
        
        if not os.path.exists(split_path):
            print(f"⚠️ Warning: {split_path} not found, skipping {split}")
            continue
            
        print(f"\n📁 Analyzing {split.upper()} set...")
        
        # Get all audio files
        audio_files = [f for f in os.listdir(split_path) if f.endswith('.wav')]
        durations = []
        
        # Analyze each audio file
        for audio_file in tqdm(audio_files, desc=f"Processing {split}"):
            file_path = os.path.join(split_path, audio_file)
            duration = get_audio_duration(file_path)
            
            if duration is not None:
                durations.append(duration)
        
        all_durations[split] = durations
        
        # Print basic statistics
        durations_array = np.array(durations)
        print(f"📊 {split.upper()} Statistics:")
        print(f"   Total files: {len(durations)}")
        print(f"   Min duration: {durations_array.min():.2f}s")
        print(f"   Max duration: {durations_array.max():.2f}s")
        print(f"   Mean duration: {durations_array.mean():.2f}s")
        print(f"   Median duration: {np.median(durations_array):.2f}s")
        print(f"   Std duration: {durations_array.std():.2f}s")
    
    return all_durations

def create_duration_distribution_analysis(all_durations, output_dir="audio_analysis"):
    """Create detailed distribution analysis and visualizations"""
    
    print("\n📈 Creating Distribution Analysis...")
    
    # Combine all durations for overall analysis
    all_combined = []
    for split, durations in all_durations.items():
        all_combined.extend(durations)
    
    all_combined = np.array(all_combined)
    
    # Create duration bins (0.5s intervals)
    max_duration = int(np.ceil(all_combined.max()))
    bins = np.arange(0, max_duration + 0.5, 0.5)
    
    # Analyze distribution by bins
    distribution_data = []
    
    for split, durations in all_durations.items():
        durations_array = np.array(durations)
        
        # Count files in each duration range
        for i in range(len(bins)-1):
            start_time = bins[i]
            end_time = bins[i+1]
            
            count = np.sum((durations_array >= start_time) & (durations_array < end_time))
            percentage = (count / len(durations_array)) * 100
            
            distribution_data.append({
                'Split': split,
                'Duration_Range': f"{start_time:.1f}-{end_time:.1f}s",
                'Start_Time': start_time,
                'End_Time': end_time,
                'Count': count,
                'Percentage': percentage
            })
    
    # Create DataFrame
    df = pd.DataFrame(distribution_data)
    
    # Save detailed distribution to Excel
    excel_path = os.path.join(output_dir, "audio_duration_distribution.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Overall distribution
        df.to_excel(writer, sheet_name='Duration_Distribution', index=False)
        
        # Summary by split
        summary_data = []
        for split, durations in all_durations.items():
            durations_array = np.array(durations)
            summary_data.append({
                'Split': split,
                'Total_Files': len(durations),
                'Min_Duration': durations_array.min(),
                'Max_Duration': durations_array.max(),
                'Mean_Duration': durations_array.mean(),
                'Median_Duration': np.median(durations_array),
                'Std_Duration': durations_array.std(),
                'Total_Hours': durations_array.sum() / 3600
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Integer second distribution (what you asked for)
        integer_distribution = []
        for split, durations in all_durations.items():
            durations_array = np.array(durations)
            
            for second in range(1, int(np.ceil(durations_array.max())) + 1):
                count = np.sum((durations_array >= second-1) & (durations_array < second))
                percentage = (count / len(durations_array)) * 100
                
                integer_distribution.append({
                    'Split': split,
                    'Duration_Second': f"{second}s",
                    'Count': count,
                    'Percentage': percentage
                })
        
        integer_df = pd.DataFrame(integer_distribution)
        integer_df.to_excel(writer, sheet_name='Integer_Second_Distribution', index=False)
    
    print(f"📋 Detailed analysis saved to: {excel_path}")
    
    return df, summary_df, integer_df

def create_visualizations(all_durations, output_dir="audio_analysis"):
    """Create 4 separate visualization plots"""
    
    print("\n📊 Creating Individual Visualizations...")
    
    # Set up the plotting style (compatible with older matplotlib)
    try:
        plt.style.use('seaborn')
    except:
        try:
            plt.style.use('ggplot')
        except:
            plt.style.use('default')
    
    # Set seaborn style if available
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    except ImportError:
        pass
    
    colors = ['blue', 'orange', 'green', 'red']
    
    # ========== PLOT 1: Histogram Distribution ==========
    plt.figure(figsize=(10, 6))
    for i, (split, durations) in enumerate(all_durations.items()):
        plt.hist(durations, bins=50, alpha=0.7, label=split.upper(), 
                density=True, color=colors[i % len(colors)])
    
    plt.xlabel('Duration (seconds)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Audio Duration Distribution by Split', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot 1
    plot1_path = os.path.join(output_dir, "1_duration_histogram.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"📈 Plot 1 saved: {plot1_path}")
    plt.close()
    
    # ========== PLOT 2: Box Plot Comparison ==========
    plt.figure(figsize=(10, 6))
    data_for_box = [durations for durations in all_durations.values()]
    labels_for_box = [split.upper() for split in all_durations.keys()]
    
    # Create box plot
    bp = plt.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Duration (seconds)', fontsize=12)
    plt.title('Duration Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Save plot 2
    plot2_path = os.path.join(output_dir, "2_duration_boxplot.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"📈 Plot 2 saved: {plot2_path}")
    plt.close()
    
    # ! Boxplot by split with additional statistics
    # ========== PLOT 3: Integer Second Distribution BY SPLIT ==========
    plt.figure(figsize=(14, 8))
    
    # Colors for each split
    colors = ['blue', 'orange', 'green', 'red']
    
    # Get all durations for consistent x-axis range
    all_combined = []
    for durations in all_durations.values():
        all_combined.extend(durations)
    max_duration = int(np.ceil(max(all_combined)))
    
    # Create integer second labels (1s, 2s, 3s, etc.)
    second_labels = [f"{second}s" for second in range(1, min(max_duration + 1, 11))]
    x_positions = range(len(second_labels))
    
    # Width of bars (divide by number of splits for side-by-side bars)
    bar_width = 0.3
    
    # Plot histogram for each split
    for i, (split, durations) in enumerate(all_durations.items()):
        second_counts = []
        
        # Count files in each integer second range for this split
        for second in range(1, min(max_duration + 1, 11)):
            count = sum(1 for d in durations if second-1 <= d < second)
            second_counts.append(count)
        
        # Calculate x positions for this split's bars
        x_pos = [x + (i * bar_width) for x in x_positions]
        
        # Create bars for this split
        bars = plt.bar(x_pos, second_counts, bar_width, 
                      label=f'{split.upper()} ({len(durations):,} files)', 
                      color=colors[i % len(colors)], 
                      alpha=0.8, 
                      edgecolor='black', 
                      linewidth=0.5)
        
        # Add value labels on bars
        for bar, count in zip(bars, second_counts):
            if count > 0:  # Only show labels for non-zero bars
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(second_counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold', rotation=0)
    
    # Customize the plot
    plt.xlabel('Duration Range (Integer Seconds)', fontsize=12)
    plt.ylabel('Number of Audio Files', fontsize=12)
    plt.title('Audio Files Distribution by Duration - Split Comparison', fontsize=14, fontweight='bold')
    
    # Set x-axis ticks to be centered between the grouped bars
    center_positions = [x + bar_width for x in x_positions]
    plt.xticks(center_positions, second_labels, rotation=0)
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add total files annotation
    total_files = len(all_combined)
    plt.text(0.02, 0.95, f'Total Files Across All Splits: {total_files:,}', 
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot 3
    plot3_path = os.path.join(output_dir, "3_integer_second_distribution_by_split.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"📈 Plot 3 saved: {plot3_path}")
    plt.close()

    # Alternative PLOT 3: Stacked Integer Second Distribution

    # ! Stacked bar plot for integer second distribution across splits
    # # ========== PLOT 3: Stacked Integer Second Distribution ==========
    # plt.figure(figsize=(12, 8))
    
    # # Get data for all splits
    # all_combined = []
    # for durations in all_durations.values():
    #     all_combined.extend(durations)
    # max_duration = int(np.ceil(max(all_combined)))
    
    # # Prepare data for stacked bars
    # second_labels = [f"{second}s" for second in range(1, min(max_duration + 1, 11))]
    
    # # Collect counts for each split
    # split_data = {}
    # colors = ['blue', 'orange', 'green', 'red']
    
    # for split, durations in all_durations.items():
    #     second_counts = []
    #     for second in range(1, min(max_duration + 1, 11)):
    #         count = sum(1 for d in durations if second-1 <= d < second)
    #         second_counts.append(count)
    #     split_data[split] = second_counts
    
    # # Create stacked bars
    # bottom = [0] * len(second_labels)
    
    # for i, (split, counts) in enumerate(split_data.items()):
    #     bars = plt.bar(second_labels, counts, 
    #                   label=f'{split.upper()} ({len(all_durations[split]):,} files)',
    #                   color=colors[i % len(colors)], 
    #                   alpha=0.8,
    #                   bottom=bottom,
    #                   edgecolor='black',
    #                   linewidth=0.5)
        
    #     # Add count labels in the middle of each segment
    #     for j, (bar, count) in enumerate(zip(bars, counts)):
    #         if count > 0:
    #             label_y = bottom[j] + count/2
    #             plt.text(bar.get_x() + bar.get_width()/2., label_y,
    #                     f'{count:,}', ha='center', va='center', 
    #                     fontsize=9, fontweight='bold', color='white')
        
    #     # Update bottom for next stack
    #     bottom = [b + c for b, c in zip(bottom, counts)]
    
    # plt.xlabel('Duration Range (Integer Seconds)', fontsize=12)
    # plt.ylabel('Number of Audio Files', fontsize=12)
    # plt.title('Audio Files Distribution by Duration - Stacked by Split', fontsize=14, fontweight='bold')
    # plt.legend(loc='upper right')
    # plt.grid(True, alpha=0.3, axis='y')
    # plt.xticks(rotation=0)
    
    # # Add total annotation
    # total_files = len(all_combined)
    # plt.text(0.02, 0.95, f'Total Files: {total_files:,}', 
    #          transform=plt.gca().transAxes, fontsize=11,
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # plt.tight_layout()
    
    # # Save plot 3
    # plot3_path = os.path.join(output_dir, "3_integer_second_stacked_distribution.png")
    # plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    # print(f"📈 Plot 3 (stacked) saved: {plot3_path}")
    # plt.close()
    
    # ========== PLOT 4: Cumulative Distribution ==========
    plt.figure(figsize=(10, 6))
    for i, (split, durations) in enumerate(all_durations.items()):
        sorted_durations = np.sort(durations)
        cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        plt.plot(sorted_durations, cumulative, label=split.upper(), 
                linewidth=3, color=colors[i % len(colors)], marker='o', markersize=3)
    
    plt.xlabel('Duration (seconds)', fontsize=12)
    plt.ylabel('Cumulative Proportion', fontsize=12)
    plt.title('Cumulative Duration Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot 4
    plot4_path = os.path.join(output_dir, "4_cumulative_distribution.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    print(f"📈 Plot 4 saved: {plot4_path}")
    plt.close()
    
    # ========== BONUS: Combined Overview (Optional) ==========
    plt.figure(figsize=(15, 4))
    
    # Mini histogram for overview
    for i, (split, durations) in enumerate(all_durations.items()):
        plt.hist(durations, bins=30, alpha=0.6, label=split.upper(), 
                color=colors[i % len(colors)])
    
    plt.xlabel('Duration (seconds)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Audio Duration Overview - All Splits', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save overview
    overview_path = os.path.join(output_dir, "0_overview_all_splits.png")
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    print(f"📈 Overview saved: {overview_path}")
    plt.close()
    
    print("\n✅ All individual plots created successfully!")
    print(f"📁 Check the '{output_dir}' folder for 5 separate images:")
    print("   • 0_overview_all_splits.png")
    print("   • 1_duration_histogram.png") 
    print("   • 2_duration_boxplot.png")
    print("   • 3_integer_second_distribution.png")
    print("   • 4_cumulative_distribution.png")

# Alternative: Create a function for just the integer second plot
def create_integer_second_plot_only(all_durations, output_dir="audio_analysis"):
    """Create only the integer second distribution plot"""
    
    plt.figure(figsize=(14, 8))
    
    # Combine all durations
    all_combined = []
    for durations in all_durations.values():
        all_combined.extend(durations)
    
    # Count by integer seconds (show more range)
    max_duration = int(np.ceil(max(all_combined)))
    second_counts = []
    second_labels = []
    
    for second in range(1, min(max_duration + 1, 16)):  # Show up to 15 seconds
        count = sum(1 for d in all_combined if second-1 <= d < second)
        second_counts.append(count)
        second_labels.append(f"{second}s")
    
    # Create bars with gradient colors
    bars = plt.bar(second_labels, second_counts, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(second_counts))),
                   edgecolor='black', alpha=0.8, width=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, second_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(second_counts)*0.02,
                f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Duration Range', fontsize=14)
    plt.ylabel('Number of Audio Files', fontsize=14)
    plt.title('Audio Files Distribution by Duration (Integer Seconds)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add total count annotation
    total_files = len(all_combined)
    plt.text(0.98, 0.95, f'Total Files: {total_files:,}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the focused plot
    focused_path = os.path.join(output_dir, "FOCUSED_integer_seconds.png")
    plt.savefig(focused_path, dpi=300, bbox_inches='tight')
    print(f"🎯 Focused integer second plot saved: {focused_path}")
    plt.close()

# Also add this function to check available styles (optional debugging)
def check_available_styles():
    """Check what matplotlib styles are available"""
    print("Available matplotlib styles:")
    print(plt.style.available)

# If you want to check styles, uncomment this line in main:
# check_available_styles()
def print_integer_second_summary(all_durations):
    """Print a nice summary of integer second distribution"""
    
    print("\n" + "="*60)
    print("🕒 INTEGER SECOND DURATION SUMMARY")
    print("="*60)
    
    # Combine all durations
    all_combined = []
    for durations in all_durations.values():
        all_combined.extend(durations)
    
    total_files = len(all_combined)
    max_duration = int(np.ceil(max(all_combined)))
    
    print(f"Total audio files analyzed: {total_files:,}")
    print(f"Duration range: 0 - {max_duration} seconds")
    print("\n📊 Distribution by duration:")
    print("-" * 40)
    
    for second in range(1, min(max_duration + 1, 11)):  # Show up to 10 seconds
        count = sum(1 for d in all_combined if second-1 <= d < second)
        percentage = (count / total_files) * 100
        
        # Create a simple bar visualization
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length + "░" * (25 - bar_length)
        
        print(f"{second:2d}s: {count:5,} files ({percentage:5.1f}%) |{bar}|")
    
    # Show longer durations summary
    long_count = sum(1 for d in all_combined if d >= 10)
    if long_count > 0:
        long_percentage = (long_count / total_files) * 100
        print(f"10s+: {long_count:5,} files ({long_percentage:5.1f}%)")
    
    print("-" * 40)

# Add this new function after the existing create_visualizations function:

def create_combined_visualization(all_durations, output_dir="audio_analysis"):
    """Create one combined figure with all 4 plots in a 2x2 grid"""
    
    print("\n📊 Creating Combined Visualization...")
    
    # Set up the plotting style
    try:
        plt.style.use('seaborn')
    except:
        try:
            plt.style.use('ggplot')
        except:
            plt.style.use('default')
    
    # Set seaborn style if available
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    except ImportError:
        pass
    
    colors = ['blue', 'orange', 'green', 'red']
    
    # Create the combined figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Audio Duration Distribution Analysis - Complete Overview', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # ========== SUBPLOT 1: Histogram Distribution ==========
    ax1 = axes[0, 0]
    for i, (split, durations) in enumerate(all_durations.items()):
        ax1.hist(durations, bins=50, alpha=0.7, label=split.upper(), 
                density=True, color=colors[i % len(colors)])
    
    ax1.set_xlabel('Duration (seconds)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Duration Distribution by Split', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== SUBPLOT 2: Box Plot Comparison ==========
    ax2 = axes[0, 1]
    data_for_box = [durations for durations in all_durations.values()]
    labels_for_box = [split.upper() for split in all_durations.keys()]
    
    # Create box plot
    bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Duration (seconds)', fontsize=11)
    ax2.set_title('Duration Comparison (Box Plot)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ========== SUBPLOT 3: Integer Second Distribution ==========
    ax3 = axes[1, 0]
    all_combined = []
    for durations in all_durations.values():
        all_combined.extend(durations)
    
    # Count by integer seconds
    max_duration = int(np.ceil(max(all_combined)))
    second_counts = []
    second_labels = []
    
    for second in range(1, min(max_duration + 1, 11)):  # Show up to 10 seconds
        count = sum(1 for d in all_combined if second-1 <= d < second)
        second_counts.append(count)
        second_labels.append(f"{second}s")
    
    bars = ax3.bar(second_labels, second_counts, color='skyblue', 
                   edgecolor='navy', alpha=0.7, width=0.6)
    
    # Add value labels on bars (smaller font for combined view)
    for bar, count in zip(bars, second_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(second_counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Duration Range', fontsize=11)
    ax3.set_ylabel('Number of Files', fontsize=11)
    ax3.set_title('Files by Integer Second Duration', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # ========== SUBPLOT 4: Cumulative Distribution ==========
    ax4 = axes[1, 1]
    for i, (split, durations) in enumerate(all_durations.items()):
        sorted_durations = np.sort(durations)
        cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        ax4.plot(sorted_durations, cumulative, label=split.upper(), 
                linewidth=2.5, color=colors[i % len(colors)], marker='o', markersize=2)
    
    ax4.set_xlabel('Duration (seconds)', fontsize=11)
    ax4.set_ylabel('Cumulative Proportion', fontsize=11)
    ax4.set_title('Cumulative Duration Distribution', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    
    # Save the combined plot
    combined_path = os.path.join(output_dir, "COMBINED_all_4_plots.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"📈 Combined visualization saved: {combined_path}")
    plt.close()
    
    return combined_path

# Update the main section to call both functions:
if __name__ == "__main__":
    # Set your dataset path
    dataset_path = "/home/rothi/ssl-antispoofing-bahasa-indonesia/dataset/LA"
    output_dir = "audio_duration_analysis"
    
    print("🎵 AUDIO DURATION DISTRIBUTION ANALYZER")
    print("=" * 50)
    
    # Analyze duration distribution
    all_durations = analyze_dataset_duration(dataset_path, output_dir)
    
    if all_durations:
        # Create detailed analysis
        df, summary_df, integer_df = create_duration_distribution_analysis(all_durations, output_dir)
        
        # Create individual visualizations (4 separate images)
        create_visualizations(all_durations, output_dir)
        
        # Create combined visualization (1 image with all 4 plots)
        create_combined_visualization(all_durations, output_dir)
        
        # Print integer second summary
        print_integer_second_summary(all_durations)
        
        print(f"\n✅ Analysis complete! Check the '{output_dir}' folder for detailed results.")
        print("\n📁 Generated Files:")
        print("   INDIVIDUAL PLOTS:")
        print("   • 1_duration_histogram.png") 
        print("   • 2_duration_boxplot.png")
        print("   • 3_integer_second_distribution.png")
        print("   • 4_cumulative_distribution.png")
        print("   • 0_overview_all_splits.png")
        print("   COMBINED PLOT:")
        print("   • COMBINED_all_4_plots.png  ⭐ (All 4 in one image)")
        print("   EXCEL FILES:")
        print("   • audio_duration_distribution.xlsx")
    else:
        print("❌ No audio files found for analysis.")