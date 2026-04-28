# main.py
import argparse
import os
import logging
import random
import torch
import numpy as np
from train import train_model, test_model
from utils.data_processor import process_and_save_data
from utils.graph_builder import build_adjacency_matrix


def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(43)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_args():
    parser = argparse.ArgumentParser(description="ST-GRAD Pipeline - Optimized for Efficiency")

    # Execution Mode
    parser.add_argument('--mode', type=str, default='all', choices=['data', 'graph', 'train', 'predict', 'all'],
                        help="all: Smart execution; data: Force pre-process; graph: Force graph; train: Force train")

    # Paths
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--cluster_file', type=str, default='data/raw/cluster_results.csv')
    parser.add_argument('--save_dir', type=str, default='models/saved')
    parser.add_argument('--load_path', type=str, default='models/saved/st_grad_best_model.pth')

    # Model/Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--diffusion_steps', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=10)

    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs('data/processed', exist_ok=True)

    # Define file paths to check
    dataset_path = 'data/processed/stock_dataset.npz'
    adj_matrix_path = 'data/processed/normalized_adj_matrix.npy'

    # --- Stage 1: Graph Construction ---
    # Logic: Run if mode is 'graph' OR (mode is 'all' AND file doesn't exist)
    should_run_graph = (args.mode == 'graph') or (args.mode == 'all' and not os.path.exists(adj_matrix_path))

    if should_run_graph:
        logging.info(">>> Stage 1/3: Adjacency Matrix Construction started...")
        try:
            build_adjacency_matrix(args.cluster_file, adj_matrix_path, 'data/raw/stock_order.npy')
            logging.info(">>> Stage 1/3: Graph Construction completed successfully.\n")
        except Exception as e:
            logging.error(f"Graph Construction Failed: {e}")
            return
    elif args.mode == 'all':
        logging.info(f">>> Stage 1/3: Skipping Graph Construction (Found existing: {adj_matrix_path}).")



    # --- Stage 2: Data Preprocessing ---
    # Logic: Run if mode is 'data' OR (mode is 'all' AND file doesn't exist)
    should_run_data = (args.mode == 'data') or (args.mode == 'all' and not os.path.exists(dataset_path))

    if should_run_data:
        logging.info(">>> Stage 2/3: Data Preprocessing started...")
        try:
            process_and_save_data(args.data_dir, 'data/raw/stock_order.npy', dataset_path)
            logging.info(">>> Stage 2/3: Data Preprocessing completed successfully.\n")
        except Exception as e:
            logging.error(f"Data Preprocessing Failed: {e}")
            return
    elif args.mode == 'all':
        logging.info(f">>> Stage 2/3: Skipping Preprocessing (Found existing: {dataset_path}).")

    # --- Stage 3: Model Training ---
    if args.mode in ['train', 'all']:
        logging.info(">>> Stage 3/3: Model Training started...")
        try:
            train_model(args)
            logging.info(">>> Stage 3/3: Model Training pipeline finished.\n")
        except Exception as e:
            logging.error(f"Model Training Failed: {e}")
            return

    # --- Independent: Prediction/Evaluation ---
    if args.mode == 'predict':
        logging.info(f">>> Predict Mode: Evaluating {args.load_path}...")
        test_model(args)


if __name__ == "__main__":
    main()