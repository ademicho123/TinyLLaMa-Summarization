import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(data):
    """Plot distributions of numerical features"""
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, len(numerical_cols)*3))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(len(numerical_cols), 1, i)
        sns.histplot(data=data, x=col, hue='Class', multiple="stack")
        plt.title(f'Distribution of {col}')
    plt.tight_layout()

def plot_correlations(data):
    """Plot correlation matrix"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')

def plot_learning_curves(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'])
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    
    plt.tight_layout()
