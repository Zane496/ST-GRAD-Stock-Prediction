import os
import warnings
import numpy as np
import pandas as pd

np.random.seed(43)

WINDOW_SIZE = 30
PRED_HORIZON = 1
BASE_THRESHOLD = 0.0003
MIN_VOLUME_CHANGE = 0.003
K_FACTOR = 3

SECTOR_ORDER = ['energy', 'healthcare', 'technology', 'materials', 'industrial']
SECTOR_FOLDERS = {
    'energy': 'Energy', 'healthcare': 'Healthcare',
    'technology': 'Technology', 'materials': 'BasicMaterials', 'industrial': 'Industrials'
}

FEATURES = [
    'Relative_Close', 'OC_Ratio', 'VWAP', 'RSI', 'RSI_Relative',
    'MACD_Signal', 'Volume_Ratio', 'ROC', 'Momentum_5', 'Momentum_10',
    'MA5_vs_MA20', 'BB_Width', 'BB_Position', 'ATR', 'Volume_Change_Pct',
    'Close_vs_High', 'Volatility', 'Price_Volume_Corr', "Volume_Momentum",
    "Price_Acceleration", "High_Low_Ratio", "Close_MA_Ratio",
    "Sector_Relative_Strength", "Volatility_Ratio"
]

def calculate_indicators(df):
    """Calculate technical indicators."""
    df = df.fillna(method='ffill').fillna(0)
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = df[price_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

    df['OC_Ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
    df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(30).mean()
    df['Relative_Close'] = df['close'] / df['close'].rolling(30).mean()
    df['Volatility'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (typical_price.rolling(30).sum() + df['volume'] * df['close']) / (df['volume'].rolling(30).sum() + df['volume'] + 1e-8)

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(21).mean()
    loss = (-delta).where(delta < 0, 0).rolling(21).mean()
    df['RSI'] = (100 - 100 / (1 + gain / (loss + 1e-8))).clip(0, 100)
    df['RSI_Relative'] = df['RSI'] / 50 - 1

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    df['MACD_Signal'] = df['MACD_Hist'] / (df['close'] + 1e-8)
    df['ROC'] = df['close'].pct_change(periods=12)
    df['Momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['Momentum_10'] = df['close'] / df['close'].shift(10) - 1

    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA5_vs_MA20'] = (df['MA5'] - df['MA20']) / (df['MA20'] + 1e-8)

    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['BB_Width'] = (4 * bb_std) / (bb_middle + 1e-8)
    df['BB_Position'] = (df['close'] - (bb_middle - 2 * bb_std)) / (4 * bb_std + 1e-8)

    true_range = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['ATR'] = true_range.rolling(14).mean()
    df['Volume_Change_Pct'] = df['volume'].pct_change()
    df['Close_vs_High'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['Price_Volume_Corr'] = df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
    df['Volume_Momentum'] = df['volume'].pct_change(5)
    df['Price_Acceleration'] = df['close'].pct_change().diff()
    df['High_Low_Ratio'] = df['high'] / (df['low'] + 1e-8)
    df['Close_MA_Ratio'] = df['close'] / df['close'].rolling(20).mean()
    df['Sector_Relative_Strength'] = df['close'].pct_change(5) - df['close'].pct_change(5).rolling(10).mean()
    df['Volatility_Ratio'] = df['close'].rolling(5).std() / df['close'].rolling(20).std()

    return df.fillna(0)

def create_labels(data, window_size=WINDOW_SIZE):
    """Generate dynamic trading labels."""
    labels_length = len(data) - window_size - PRED_HORIZON
    labels = np.zeros((labels_length, data.shape[1], 1), dtype=np.float32)

    for i in range(labels_length):
        current = data[i + window_size - 1, :, 0]
        next_day = data[i + window_size, :, 0]
        avg_price = np.mean(data[i:i + window_size, :, 0], axis=0)

        volume_pct_idx = FEATURES.index('Volume_Change_Pct') if 'Volume_Change_Pct' in FEATURES else 6
        volume_pct = data[i + window_size - 1, :, volume_pct_idx]

        valid_mask = (
            ~np.isnan(current) & ~np.isnan(next_day) & ~np.isnan(volume_pct) &
            (current > 1e-6) & (avg_price > 1e-6) & (np.abs(volume_pct) >= MIN_VOLUME_CHANGE)
        )

        volatility = np.tanh(np.abs(volume_pct) * 8)
        thresholds = BASE_THRESHOLD * (1 + volatility * K_FACTOR)
        returns = (next_day - avg_price) / (avg_price + 1e-8)

        labels[i, valid_mask, 0] = (
            (returns[valid_mask] > thresholds[valid_mask]) &
            (np.random.random(returns[valid_mask].shape) > 0.10)
        ).astype(np.float32)

    return labels

def load_and_align(data_dir, stock_order):
    """Load and align timeline for all assets."""
    file_dict = {}
    for sector in SECTOR_ORDER:
        sector_dir = os.path.join(data_dir, SECTOR_FOLDERS[sector])
        if os.path.exists(sector_dir):
            for f in os.listdir(sector_dir):
                if f.endswith('.csv'):
                    file_dict[f.replace('.csv', '')] = os.path.join(sector_dir, f)

    dfs = [calculate_indicators(pd.read_csv(file_dict[code])) for code in stock_order]

    common_dates = dfs[0]['date']
    for df in dfs[1:]:
        common_dates = pd.merge(pd.DataFrame({'date': common_dates}), pd.DataFrame({'date': df['date']}), on='date')['date']

    feature_data = np.stack([df[df['date'].isin(common_dates)][FEATURES].values for df in dfs], axis=1)
    feature_data = np.nan_to_num(np.where(np.isinf(feature_data), 0, feature_data), nan=0.0)
    aligned_dates = dfs[0][dfs[0]['date'].isin(common_dates)]['date'].values

    return feature_data, aligned_dates

def process_and_save_data(data_dir, stock_order_path, output_npz_path):
    """Execute data pipeline."""
    stock_order = np.load(stock_order_path, allow_pickle=True)
    raw_data, all_dates = load_and_align(data_dir, stock_order)

    y = create_labels(raw_data, WINDOW_SIZE)
    X = np.stack([raw_data[i:i + WINDOW_SIZE] for i in range(len(raw_data) - WINDOW_SIZE - PRED_HORIZON + 1)], axis=0)
    X = X[:len(y)]

    split_idx = int(0.7 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_mean = np.nanmean(X_train, axis=(0, 1, 3), keepdims=True)
        train_std = np.nanstd(X_train, axis=(0, 1, 3), keepdims=True)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)

    X_train = np.nan_to_num((X_train - train_mean) / train_std, nan=0.0)
    X_val = np.nan_to_num((X_val - train_mean) / train_std, nan=0.0)

    train_valid = ~(np.isnan(X_train).any(axis=(1, 2, 3)) | np.isinf(X_train).any(axis=(1, 2, 3)))
    val_valid = ~(np.isnan(X_val).any(axis=(1, 2, 3)) | np.isinf(X_val).any(axis=(1, 2, 3)))

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        X_train=X_train[train_valid].astype(np.float32), y_train=y_train[train_valid].astype(np.float32),
        X_val=X_val[val_valid].astype(np.float32), y_val=y_val[val_valid].astype(np.float32),
        train_mean=train_mean.astype(np.float32), train_std=train_std.astype(np.float32),
        stock_order=stock_order, features=FEATURES, all_dates=all_dates
    )