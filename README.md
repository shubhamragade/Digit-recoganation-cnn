# Digit-recoganation-cnn
#  CNN on Digits Dataset (PyTorch)

This project demonstrates how to build, train, and evaluate
Convolutional Neural Networks (CNNs) on the **sklearn Digits dataset**
using **PyTorch**.\
The notebook is structured for step-by-step learning --- starting from a
simple baseline CNN and then improving it with modern deep learning
techniques.

------------------------------------------------------------------------

##  Project Structure

-   **Data Preparation**:
    -   Loads the `digits` dataset from `sklearn.datasets`.\
    -   Normalizes the images to \[0,1\].\
    -   Prepares training & testing splits using `DataLoader`.
-   **Models**:
    1.  **BaseCNN** -- A simple CNN with one convolution + pooling +
        fully connected layers.\
    2.  **ImprovedCNN** -- A deeper CNN with:
        -   More convolution filters\
        -   Batch Normalization\
        -   Dropout regularization
-   **Training Loop**:
    -   Implemented from scratch with `CrossEntropyLoss` and `Adam`
        optimizer.\
    -   Both models trained for a few epochs.
-   **Evaluation**:
    -   Accuracy is computed on the test dataset.\
    -   Results are compared in a `pandas.DataFrame`.

------------------------------------------------------------------------

##  Results (Sample)

  Model         Test Accuracy (%)
  ------------- -------------------
  BaseCNN       \~92%
  ImprovedCNN   \~97%

------------------------------------------------------------------------

##  How to Run

1.  Clone the repository:

    ``` bash
    git clone https://github.com/<your-username>/cnn-digits-pytorch.git
    cd cnn-digits-pytorch
    ```

2.  Install dependencies:

    ``` bash
    pip install torch torchvision scikit-learn pandas matplotlib
    ```

3.  Run the notebook:

    ``` bash
    jupyter notebook
    ```

    Open the provided notebook and execute cells step by step.

------------------------------------------------------------------------

##  Tech Stack

-   **PyTorch** -- deep learning framework\
-   **Scikit-learn** -- dataset (digits)\
-   **Pandas** -- results comparison\
-   **Matplotlib** (optional) -- visualization

------------------------------------------------------------------------

##  Next Steps

-   Add more experiments with optimizers & learning rates.\
-   Try deeper architectures (ResNet, VGG).\
-   Visualize feature maps & misclassified images.

------------------------------------------------------------------------

 This notebook is designed for **educational purposes** --- helping
beginners understand how CNNs work on small datasets before moving to
real-world problems.
