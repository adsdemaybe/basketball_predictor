# 🏀 March Madness Basketball Game Predictor

A deep learning project that predicts NCAA basketball tournament game outcomes using PyTorch. This model analyzes historical game statistics to forecast match results for the March Madness tournament.

## 📋 Project Overview

This project implements a Multi-Layer Perceptron (MLP) neural network to predict the outcome of NCAA Men's Basketball Tournament games. The model is trained on detailed historical game data from the Kaggle March Machine Learning Mania 2025 competition and generates probability predictions for tournament matchups. This is my first project every using PyTorch and actually implementing any DL techniques to a data set. It was quite fun, as I got to experiment with the things that I actually learnt about. This project still remains incomplete, as a PoC. 

### Key Features

- **Data Augmentation**: Flips game data to create balanced training examples from both winning and losing perspectives
- **Feature Engineering**: Calculates percentage-based statistics (FG%, 3P%, FT%) and derived metrics
- **Temporal Weighting**: Applies exponential decay to prioritize more recent games in training
- **Deep Learning Architecture**: Configurable MLP with batch normalization, LeakyReLU activation, and dropout regularization
- **Tournament Prediction**: Generates Kaggle-compatible submission files with win probabilities

## 🏗️ Model Architecture

### Neural Network Design

The model uses a customizable Multi-Layer Perceptron with the following components:

- **Input Layer**: 20 features (team statistics and game context)
- **Hidden Layers**: Configurable (default: [20, 10, 8] neurons)
- **Activation**: LeakyReLU with configurable negative slope
- **Regularization**: 
  - Batch Normalization after each hidden layer
  - Dropout (default: 0.2) for generalization
  - L2 weight decay (default: 1e-5)
- **Output Layer**: Single neuron with sigmoid activation for binary classification
- **Loss Function**: Binary Cross-Entropy with Logits Loss
- **Optimizer**: Adam (configurable: SGD, RMSprop)

### Training Features

- **Weighted Loss**: Games are weighted by recency using exponential decay (λ = 0.5)
- **Data Normalization**: Z-score normalization using training set statistics
- **Train/Test Split**: 80/20 split with stratified sampling
- **Batch Processing**: Configurable batch size (default: 64)
- **GPU Support**: Automatic CUDA detection for accelerated training

## 📊 Dataset

The model uses data from the Kaggle March Machine Learning Mania 2025 competition:

### Primary Data Files

- `MNCAATourneyDetailedResults.csv` - Historical NCAA tournament game results with detailed statistics
- `MRegularSeasonDetailedResults.csv` - Regular season detailed game statistics
- `SampleSubmissionStage1.csv` / `SampleSubmissionStage2.csv` - Tournament prediction templates

### Data Not Included in Repository

⚠️ **Important**: The `march-machine-learning-mania-2025/` folder containing all CSV data files is **NOT** tracked in this repository due to its large size (>160MB).

**To use this project, you need to:**
1. Download the dataset from [Kaggle March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
2. Extract the data files into a folder named `march-machine-learning-mania-2025/` in the project root
3. Update file paths in the notebook cells to match your local setup

## 🔧 Installation

### Prerequisites

- Python 3.13.7
- Virtual environment (`.venv` included in project)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/adsdemaybe/basketball_predictor.git
   cd basketball_predictor
   ```

2. **Activate the virtual environment**
   ```bash
   # On macOS/Linux
   source .venv/bin/activate
   
   # On Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset** from Kaggle (see Dataset section above)

## 🚀 Usage

### Training the Model

Open `MarchMadness.ipynb` in Jupyter Notebook or VS Code and run the cells sequentially:

1. **Import Libraries and Define Dataset Class**
   ```python
   from MarchMadness import CSVDataset, MLP, train_model, prepare_data, predict
   ```

2. **Prepare Data**
   ```python
   train_loader, test_loader, n_inputs = prepare_data(
       "/path/to/MNCAATourneyDetailedResults.csv",
       test_size=0.2,
       batch_size=64
   )
   ```

3. **Train the Model**
   ```python
   model, loss_history = train_model(
       train_loader, test_loader,
       n_inputs,
       dropout_value=0.2,
       hidden_units=[20, 10, 8, 1],
       relu_slope=0.1,
       learning_rate=0.001,
       epochs=120,
       optimizer_type='adam'
   )
   ```

4. **Visualize Training Progress**
   ```python
   plt.plot(loss_history['train_loss'], label='Train Loss')
   plt.plot(loss_history['test_loss'], label='Test Loss')
   plt.legend()
   plt.show()
   ```

5. **Generate Predictions**
   ```python
   dataset = CSVDataset("/path/to/MNCAATourneyDetailedResults.csv")
   predict("/path/to/SampleSubmissionStage2.csv", model, dataset)
   ```

## 📁 Project Structure

```
basketball_predictor/
├── .venv/                          # Python virtual environment (not tracked)
├── march-machine-learning-mania-2025/  # Dataset folder (not tracked)
│   ├── MNCAATourneyDetailedResults.csv
│   ├── MRegularSeasonDetailedResults.csv
│   ├── SampleSubmissionStage1.csv
│   ├── SampleSubmissionStage2.csv
│   └── ... (other data files)
├── MarchMadness.ipynb              # Main training notebook
├── DLTest.ipynb                    # Experimental testing notebook
├── MarchMadnessTestBook.py         # Standalone training script
├── Test.py                         # Data preprocessing utilities
├── input_data.csv                  # Generated aggregated statistics
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🧮 Feature Engineering

The model transforms raw game statistics into meaningful features:

### Input Features (20 total)

**Current Team Statistics:**
- Field Goal Percentage (FG%)
- 3-Point Percentage (3P%)
- Free Throw Percentage (FT%)
- Total Rebounds (Offensive + Defensive)
- Assists (Ast)
- Turnovers (TO)
- Steals (Stl)
- Blocks (Blk)
- Personal Fouls (PF)
- Location (Home: 1, Away: -1, Neutral: 0)

**Opponent Team Statistics:**
- Same 9 statistics as above (excluding location)

**Game Context:**
- Number of Overtime Periods (NumOT)

### Data Augmentation Strategy

Each game is represented twice in the dataset:
1. **Winner's perspective**: Current Team = Winner, Result = 1
2. **Loser's perspective**: Current Team = Loser, Result = 0

This creates a balanced dataset and allows the model to learn from both winning and losing patterns.

## 📈 Model Performance

The training process tracks:
- **Train Loss**: BCE loss on training set
- **Test Loss**: BCE loss on validation set
- **Epoch Time**: Training duration per epoch
- **Weight Norms**: Model parameter magnitudes for convergence monitoring

Typical training results (120 epochs):
- Final train loss: ~0.45-0.50
- Final test loss: ~0.48-0.52
- Training time: ~5-10 seconds per epoch on CPU

## 🎯 Prediction Output

The `predict()` function generates a Kaggle-compatible CSV file with columns:
- **ID**: Match identifier in format `year_teamA_teamB`
- **Pred**: Probability that team A wins (range: 0.0 to 1.0)

## 🔬 Experimental Files

- **`DLTest.ipynb`**: Alternative implementation with different preprocessing
- **`MarchMadnessTestBook.py`**: Simplified standalone training script
- **`Test.py`**: Data flipping and preprocessing utilities

## 🛠️ Customization

### Hyperparameter Tuning

Modify these parameters in `train_model()`:

```python
model, loss_history = train_model(
    train_loader, test_loader,
    n_inputs,
    dropout_value=0.3,           # Increase for more regularization
    hidden_units=[32, 16, 8],    # Deeper/wider network
    relu_slope=0.1,              # LeakyReLU negative slope
    learning_rate=0.0005,        # Smaller for finer updates
    epochs=200,                  # More training iterations
    optimizer_type='rmsprop',    # Try different optimizers
    weight_decay=1e-4            # Stronger L2 regularization
)
```

### Sample Weighting

Adjust temporal decay in dataset initialization:

```python
dataset = CSVDataset(path, lambda_decay=0.3)  # More weight on recent games
```

## 📝 Requirements

Key dependencies (see `requirements.txt` for full list):

- PyTorch >= 2.0
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Implement cross-validation for hyperparameter tuning
- [ ] Add team-specific embeddings or historical ELO ratings
- [ ] Experiment with advanced architectures (ResNet, Transformer)
- [ ] Incorporate additional features (coaching stats, conference strength)
- [ ] Add model interpretability (feature importance, SHAP values)

## 📄 License

This project is provided as-is for educational purposes. Dataset usage subject to Kaggle competition rules.

## 🙏 Acknowledgments

- Data provided by Kaggle's March Machine Learning Mania 2025 competition
- Built with PyTorch and the Python data science ecosystem

## 📧 Contact

For questions or collaboration opportunities, please open an issue in the GitHub repository.

---

**Note**: Remember to download the dataset separately and update file paths before running the notebooks!
