#!/usr/bin/env python3
"""Training script for BinConv model on Monash Tourism Monthly dataset.

This script demonstrates how to train and evaluate a BinConv forecasting model
on the tourism monthly dataset from the Monash Time Series Forecasting Archive.
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from binconvfm.models.binconv import BinConvForecaster
from binconvfm.datamodules import MonashDataModule


def main():
    parser = argparse.ArgumentParser(description="Train BinConv on Tourism Monthly dataset")
    
    # Data parameters
    parser.add_argument("--data-path", type=str, 
                       default="../TSDatasets/monash/data/tourism_monthly_dataset.tsf",
                       help="Path to the tourism dataset TSF file")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--context-length", type=int, default=72,
                       help="Length of input context window (months)")
    parser.add_argument("--prediction-depth-train", type=int, default=1,
                       help="Length of output for training")
    parser.add_argument("--prediction-depth-test", type=int, default=24,
                       help="Forecast horizon for evaluation (months)")
    
    # Model parameters
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--n-samples", type=int, default=100,
                       help="Number of samples for prediction")
    parser.add_argument("--num-filters-2d", type=int, default=72,
                       help="Number of 2D filters")
    parser.add_argument("--num-filters-1d", type=int, default=72,
                       help="Number of 1D filters")
    parser.add_argument("--num-bins", type=int, default=1000,
                       help="Number of bins for discretization")
    parser.add_argument("--min-bin-value", type=float, default=-5.0,
                       help="Minimum bin value")
    parser.add_argument("--max-bin-value", type=float, default=5.0,
                       help="Maximum bin value")
    parser.add_argument("--num-blocks", type=int, default=3,
                       help="Number of convolutional blocks")
    parser.add_argument("--dropout", type=float, default=0.35,
                       help="Dropout rate")
    
    # Other parameters
    parser.add_argument("--verbose", action="store_true",
                       help="Print verbose dataset information")
    parser.add_argument("--plot-predictions", action="store_true",
                       help="Plot sample predictions")
    parser.add_argument("--save-results", type=str, default=None,
                       help="Path to save results (optional)")
    
    args = parser.parse_args()
    
    print("=== BinConv Training on Tourism Monthly Dataset ===")
    print(f"Data path: {args.data_path}")
    print(f"Context length: {args.context_length} months")
    print(f"Prediction depth test: {args.prediction_depth_test} months")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Create data module
    print("Loading dataset...")
    datamodule = MonashDataModule(
        batch_size=args.batch_size,
        horizon=args.prediction_depth_test,
        input_len=args.context_length,
        output_len=args.prediction_depth_train,
        filename=args.data_path,
        verbose=args.verbose
    )
    
    # Create model
    print("Creating BinConv model...")
    model = BinConvForecaster(
        num_epochs=args.num_epochs,
        n_samples=args.n_samples,
        context_length=args.context_length,
        num_filters_2d=args.num_filters_2d,
        num_filters_1d=args.num_filters_1d,
        num_bins=args.num_bins,
        min_bin_value=args.min_bin_value,
        max_bin_value=args.max_bin_value,
        num_blocks=args.num_blocks,
        dropout=args.dropout
    )
    print("Model created successfully!")
    print()
    
    # Train model
    print("Training model...")
    model.fit(datamodule)
    print("Training completed!")
    print()
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(datamodule)
    print("Evaluation results:")
    for metric, value in results.items():
        print(f"  {metric.upper()}: {value:.4f}")
    print()
    
    # Generate predictions for visualization
    if args.plot_predictions:
        print("Generating predictions for visualization...")
        predictions = model.predict(datamodule, horizon=args.prediction_depth_test)
        
        # Set up prediction data
        datamodule.setup("predict")
        
        # Plot a few sample predictions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        predict_loader = datamodule.predict_dataloader()
        batch_data = next(iter(predict_loader))
        input_seqs, target_seqs = batch_data
        
        # Select quantiles for prediction intervals
        quantiles = torch.tensor([0.1, 0.5, 0.9])
        
        for i in range(min(4, len(input_seqs))):
            ax = axes[i]
            
            # Get input and target sequences
            input_seq = input_seqs[i].squeeze()
            target_seq = target_seqs[i].squeeze()
            
            # Get prediction quantiles
            pred_quantiles = torch.quantile(
                predictions[0][i, :, :, -1], 
                q=quantiles, 
                dim=0
            )
            
            # Time indices
            input_time = range(len(input_seq))
            target_time = range(len(input_seq), len(input_seq) + len(target_seq))
            
            # Plot historical data
            ax.plot(input_time, input_seq, 'b-', label='Historical', linewidth=2)
            
            # Plot actual future values
            ax.plot(target_time, target_seq, 'g-', label='Actual', linewidth=2)
            
            # Plot prediction median
            ax.plot(target_time, pred_quantiles[1], 'r-', label='Prediction (median)', linewidth=2)
            
            # Plot prediction interval
            ax.fill_between(
                target_time, 
                pred_quantiles[0], 
                pred_quantiles[2],
                alpha=0.3, 
                color='red', 
                label='80% Prediction Interval'
            )
            
            ax.set_title(f'Time Series {i+1}')
            ax.set_xlabel('Time (months)')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if args.save_results:
            plot_path = Path(args.save_results).with_suffix('.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to: {plot_path}")
        else:
            plt.show()
    
    # Save results if requested
    if args.save_results:
        results_path = Path(args.save_results).with_suffix('.txt')
        with open(results_path, 'w') as f:
            f.write("BinConv Training Results on Tourism Monthly Dataset\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Data path: {args.data_path}\n")
            f.write(f"Context length: {args.context_length} months\n")
            f.write(f"Prediction depth test: {args.prediction_depth_test} months\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Number of epochs: {args.num_epochs}\n\n")
            f.write("Evaluation Results:\n")
            for metric, value in results.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
        
        print(f"Results saved to: {results_path}")
    
    print("Script completed successfully!")


if __name__ == "__main__":
    main()