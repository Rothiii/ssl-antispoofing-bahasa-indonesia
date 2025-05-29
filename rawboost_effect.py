import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import argparse
from RawBoost import (
    ISD_additive_noise,
    LnL_convolutive_noise,
    SSI_additive_noise,
    normWav,
)

class MockArgs:
    """Mock arguments class to simulate RawBoost parameters"""
    def __init__(self):
        # Parameters for LnL_convolutive_noise (algo 1)
        self.N_f = 5
        self.nBands = 5
        self.minF = 20
        self.maxF = 8000
        self.minBW = 100
        self.maxBW = 1000
        self.minCoeff = 10
        self.maxCoeff = 100
        self.minG = 0
        self.maxG = 0
        self.minBiasLinNonLin = 5
        self.maxBiasLinNonLin = 20
        
        # Parameters for ISD_additive_noise (algo 2)
        self.P = 10
        self.g_sd = 2
        
        # Parameters for SSI_additive_noise (algo 3)
        self.SNRmin = 10
        self.SNRmax = 40

def process_Rawboost_feature(feature, sr, args, algo):
    """
    Apply RawBoost augmentation based on algorithm number
    """
    print(f"Applying RawBoost algorithm {algo}...")
    
    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
    
    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    
    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )
    
    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    
    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )
    
    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )
    
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)
    
    # original data without Rawboost processing
    else:
        feature = feature
    
    return feature

def plot_comparison(original, augmented_list, sample_name, output_dir):
    """
    Create comparison plots for original vs augmented audio
    """
    algo_names = [
        "Original",
        "LnL Convolutive",
        "ISD Impulsive", 
        "SSI Additive",
        "All (1+2+3)",
        "LnL+ISD (1+2)",
        "LnL+SSI (1+3)",
        "ISD+SSI (2+3)",
        "LnL||ISD (1||2)"
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'RawBoost Augmentation Effects - {sample_name}', fontsize=16)
    
    # Plot original
    axes[0, 0].plot(original)
    axes[0, 0].set_title('Original')
    axes[0, 0].set_ylabel('Amplitude')
    
    # Plot augmented versions
    for i, (aug_audio, algo_name) in enumerate(zip(augmented_list, algo_names[1:])):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        axes[row, col].plot(aug_audio)
        axes[row, col].set_title(f'Algo {i+1}: {algo_name}')
        axes[row, col].set_ylabel('Amplitude')
        
        if row == 2:  # Last row
            axes[row, col].set_xlabel('Sample')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample_name}_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_spectrograms(original, augmented_list, sample_name, output_dir, sr=16000):
    """
    Create spectrogram comparison
    """
    algo_names = [
        "Original",
        "LnL Convolutive",
        "ISD Impulsive", 
        "SSI Additive",
        "All (1+2+3)",
        "LnL+ISD (1+2)",
        "LnL+SSI (1+3)",
        "ISD+SSI (2+3)",
        "LnL||ISD (1||2)"
    ]
    
    # Create figure with subplots for spectrograms
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Spectrogram Comparison - {sample_name}', fontsize=16)
    
    # Plot original spectrogram
    f, t, Sxx = spectrogram(original, sr)
    im0 = axes[0, 0].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    axes[0, 0].set_title('Original')
    axes[0, 0].set_ylabel('Frequency [Hz]')
    
    # Plot augmented spectrograms
    for i, (aug_audio, algo_name) in enumerate(zip(augmented_list, algo_names[1:])):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        f, t, Sxx = spectrogram(aug_audio, sr)
        im = axes[row, col].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        axes[row, col].set_title(f'Algo {i+1}: {algo_name}')
        axes[row, col].set_ylabel('Frequency [Hz]')
        
        if row == 2:  # Last row
            axes[row, col].set_xlabel('Time [s]')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sample_name}_spectrograms.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_augmentation_effects(input_files, output_dir):
    """
    Generate augmented audio files and visualizations
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize mock arguments
    args = MockArgs()
    
    # Algorithm descriptions
    algo_descriptions = {
        0: "original",
        1: "lnl_convolutive",
        2: "isd_impulsive", 
        3: "ssi_additive",
        4: "all_series_123",
        5: "lnl_isd_series_12",
        6: "lnl_ssi_series_13",
        7: "isd_ssi_series_23",
        8: "lnl_isd_parallel"
    }
    
    for file_path in input_files:
        print(f"\nProcessing: {file_path}")
        
        # Load audio
        audio, sr = librosa.load(file_path, sr=16000)
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"Loaded audio: {len(audio)} samples, {sr} Hz")
        
        # Store original and augmented versions
        augmented_audios = []
        
        # Generate augmented versions for each algorithm
        for algo in range(1, 9):  # Algorithms 1-8
            print(f"Generating algorithm {algo}...")
            
            # Apply augmentation
            augmented = process_Rawboost_feature(audio.copy(), sr, args, algo)
            augmented_audios.append(augmented)
            
            # Save augmented audio
            output_filename = f"{sample_name}_algo{algo}_{algo_descriptions[algo]}.wav"
            output_path = os.path.join(audio_dir, output_filename)
            sf.write(output_path, augmented, sr)
            print(f"Saved: {output_filename}")
        
        # Save original for comparison
        original_path = os.path.join(audio_dir, f"{sample_name}_original.wav")
        sf.write(original_path, audio, sr)
        
        # Create comparison plots
        print("Generating comparison plots...")
        plot_comparison(audio, augmented_audios, sample_name, plots_dir)
        plot_spectrograms(audio, augmented_audios, sample_name, plots_dir, sr)
        
        # Generate summary statistics
        create_summary_stats(audio, augmented_audios, sample_name, output_dir)
        
        print(f"✅ Completed processing for {sample_name}")

def create_summary_stats(original, augmented_list, sample_name, output_dir):
    """
    Create summary statistics for augmented audio
    """
    algo_names = [
        "LnL Convolutive",
        "ISD Impulsive", 
        "SSI Additive",
        "All (1+2+3)",
        "LnL+ISD (1+2)",
        "LnL+SSI (1+3)",
        "ISD+SSI (2+3)",
        "LnL||ISD (1||2)"
    ]
    
    stats_file = os.path.join(output_dir, f"{sample_name}_stats.txt")
    
    with open(stats_file, 'w') as f:
        f.write(f"RawBoost Augmentation Statistics - {sample_name}\n")
        f.write("="*60 + "\n\n")
        
        # Original statistics
        f.write(f"ORIGINAL AUDIO:\n")
        f.write(f"  Mean: {np.mean(original):.6f}\n")
        f.write(f"  Std:  {np.std(original):.6f}\n")
        f.write(f"  Min:  {np.min(original):.6f}\n")
        f.write(f"  Max:  {np.max(original):.6f}\n")
        f.write(f"  RMS:  {np.sqrt(np.mean(original**2)):.6f}\n\n")
        
        # Augmented statistics
        for i, (aug_audio, algo_name) in enumerate(zip(augmented_list, algo_names)):
            f.write(f"ALGORITHM {i+1} - {algo_name}:\n")
            f.write(f"  Mean: {np.mean(aug_audio):.6f}\n")
            f.write(f"  Std:  {np.std(aug_audio):.6f}\n")
            f.write(f"  Min:  {np.min(aug_audio):.6f}\n")
            f.write(f"  Max:  {np.max(aug_audio):.6f}\n")
            f.write(f"  RMS:  {np.sqrt(np.mean(aug_audio**2)):.6f}\n")
            
            # Calculate difference from original
            mse = np.mean((original - aug_audio) ** 2)
            f.write(f"  MSE from original: {mse:.6f}\n\n")

def main():
    """
    Main function to run the augmentation effect generator
    """
    parser = argparse.ArgumentParser(description='Generate RawBoost augmentation effects')
    parser.add_argument('--input_files', nargs='+', required=True,
                       help='Input audio files to process')
    parser.add_argument('--output_dir', default='data_augmentation_effect',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
    
    print("🎵 RawBoost Augmentation Effect Generator")
    print("="*50)
    print(f"Input files: {len(args.input_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Algorithms: 8 RawBoost variants")
    print("="*50)
    
    # Generate augmentation effects
    generate_augmentation_effects(args.input_files, args.output_dir)
    
    print("\n" + "="*50)
    print("🎉 Augmentation effect generation completed!")
    print(f"📁 Results saved in: {args.output_dir}")
    print("📁 Structure:")
    print("  ├── audio/          (augmented .wav files)")
    print("  ├── plots/          (comparison plots)")
    print("  └── *_stats.txt     (statistical summaries)")
    print("="*50)

if __name__ == "__main__":
    # Example usage (if running directly)
    if len(os.sys.argv) == 1:
        # Demo mode - you can specify your sample files here
        sample_files = [
            "dataset/LA/ASVspoof2019_LA_train/LA_T_0101BFD00001.wav",  # Replace with your actual file paths
            "dataset/LA/ASVspoof2019_LA_train/LA_T_0101BFD00002.wav"
        ]
        
        print("Demo mode - please specify actual file paths in the script")
        print("Or run with: python data_augmentation_effect.py --input_files file1.wav file2.wav")
    else:
        main()