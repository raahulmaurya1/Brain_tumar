# Brain Tumor Detection/Analysis using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg" alt="Python 3.x">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow 2.x">
  <img src="https://img.shields.io/badge/Keras-2.x-red.svg" alt="Keras 2.x">
  <img src="https://img.shields.io/badge/scikit--learn-green.svg" alt="Scikit-learn">
</p>

## 🌟 Table of Contents
- [Description](#-description)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data](#-data)
- [Methodology](#-methodology)
- [Model Architecture (Detailed)](#-model-architecture-detailed)
- [Performance & Results](#-performance--results)
- [Comparison with Existing Models](#-comparison-with-existing-models)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)
- [License](#-license)

## 📝 Description

Introducing a groundbreaking advancement in medical imaging, our customized 3D Convolutional Neural Network (CNN) stands out as a leading solution for brain tumor detection. This innovative model has achieved an exceptional classification accuracy of **98.03%**, demonstrating its superior capability in precisely identifying brain tumors from MRI scans. Designed to be lightweight yet robust, our model surpasses existing methods by offering high performance with optimized computational complexity, making it an ideal tool for rapid and accurate diagnosis.

## ✨ Features

* **📊 Data Loading and Preprocessing**: Efficiently reads and prepares a CSV dataset, which likely contains calculated image features and corresponding tumor presence labels. MRI scans are translated into numerical data and then reconstructed into three-channel color representations, enhancing the network's ability to interpret spatial structure and depth-related features.
* **🔍 Feature Analysis**: Computes a rich set of image features, including Mean, Variance, Standard Deviation, Entropy, Skewness, Kurtosis, Contrast, Energy, ASM (Angular Second Moment), Homogeneity, Dissimilarity, and Correlation.
* **🧠 Lightweight Customized 3D CNN**: Implements a customized 3D Convolutional Neural Network tailored for brain tumor detection. This model significantly reduces the number of tunable parameters while maintaining robust feature extraction capabilities.
* **📈 Model Training & Evaluation**: The notebook includes dedicated sections for training the model on the prepared dataset and rigorously evaluating its performance using appropriate metrics.

## 🚀 Installation

To get started with this project and run the Jupyter Notebook, please follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install dependencies:**
    It is highly recommended to use `pip` for installing the required libraries. The core libraries used in this notebook are:

    * `tensorflow`
    * `opencv-python` (`cv2`)
    * `keras` (integrated with TensorFlow)
    * `scikit-learn` (`sklearn`)
    * `pandas`
    * `numpy`
    * `matplotlib`
    * `Pillow` (`PIL.Image`)
    * `plotly` (for interactive visualizations, if applicable)

    You can install them all at once using the following command:

    ```bash
    pip install tensorflow opencv-python scikit-learn pandas numpy matplotlib Pillow plotly
    ```
    * **(Optional)**: If you encounter any dependency issues or wish to ensure you have the absolute latest versions, you may uncomment and execute the following commands within your notebook:
        ```python
        #pip install --upgrade anaconda
        #pip install --upgrade tensorflow
        ```

## 💻 Usage

1.  **Download the dataset:** Ensure that the `Brain Tumor.csv` file, along with any associated image data, is placed in the correct directory relative to the notebook. The notebook specifically looks for 'Brain Tumor.csv'.
2.  **Open the Jupyter Notebook:** Navigate to the project directory in your terminal and launch Jupyter Notebook:

    ```bash
    jupyter notebook Finnal_File.ipynb
    ```
3.  **Run all cells:** Once the notebook is open in your browser, execute all cells sequentially from top to bottom. This will perform all steps from data loading and preprocessing to model training and evaluation.

## 💾 Data

The project utilizes the publicly accessible "**braintumorv7**" dataset sourced from Kaggle. This dataset contains a total of **3,762 cases** for brain tumor detection. Data records are categorized into two target classes: **tumor** and **non-tumor**.
* **Non-tumor**: 2,079 entries
* **Tumor**: 1,683 entries

The dataset includes twelve distinct features extracted from images, with the first five representing first-order metrics. Additionally, it includes image data that complements the numerical features for classification.

Here's an illustrative snapshot of the dataset structure:

| Image | Class | Mean | Variance | Standard Deviation | Entropy | Skewness | Kurtosis | Contrast | Energy | ASM | Homogeneity | Dissimilarity | Correlation | Coarseness |
| :---- | :---- | :--- | :------- | :----------------- | :------ | :------- | :------- | :------- | :----- | :-- | :---------- | :------------ | :---------- | :--------- |
| Image1 | 0 | 6.535 | 619.58 | 24.89 | 0.109 | 4.276 | 18.90 | 98.61 | 0.293 | 0.086 | 0.530 | 4.473 | 0.981 | 7.45E-155 |
| Image2 | 0 | 8.749 | 805.95 | 28.38 | 0.266 | 3.718 | 14.46 | 63.85 | 0.475 | 0.225 | 0.651 | 3.220 | 0.988 | 7.45E-155 |
| Image3 | 1 | 7.341 | 1143.80 | 33.82 | 0.001 | 5.061 | 26.47 | 81.86 | 0.031 | 0.001 | 0.268 | 5.981 | 0.978 | 7.45E-155 |
| Image4 | 1 | 5.958 | 959.71 | 30.97 | 0.001 | 5.677 | 33.42 | 151.22 | 0.032 | 0.001 | 0.243 | 7.700 | 0.964 | 7.45E-155 |
| Image5 | 0 | 7.315 | 729.54 | 27.01 | 0.146 | 4.283 | 19.07 | 174.98 | 0.343 | 0.118 | 0.501 | 6.834 | 0.972 | 7.45E-155 |

## ⚙️ Methodology

This study employs a sophisticated approach involving a **customized 3D Convolutional Neural Network** to detect brain tumors by analyzing MRI imaging data.

The methodology involves several key phases:
1.  **Data Loading and Resizing**: MRI images are resized to a standardized dimension of `224x224` pixels to ensure uniformity for network input.
2.  **3D Color Channel Incorporation**: A 3D color channel is incorporated to enhance the network's ability to capture intricate spatial features within MRI scans. This translates MRI images into numerical arrays, preserving spatial and intensity information crucial for detection.
3.  **Input Reshaping**: Numerical arrays are reshaped to conform to the specific input requirements of the customized 3D CNN model.
4.  **Customized CNN Architecture**: The model is meticulously designed with multiple convolutional layers for high-dimensional feature extraction, interspersed with max pooling layers to reduce dimensionality while retaining essential features. Fully connected layers integrate learned features, leading to a softmax activation function for multi-class classification.

The hyperparameters for the proposed customized CNN model are as follows:

| Hyperparameter | Value |
| :------------- | :---- |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 50 |
| Optimizer | Adam |
| Dropout Rate | 0.3 |
| Activation Function | ReLU |
| Output Activation | Softmax |
| Loss Function | Binary Crossentropy |

## 🏗️ Model Architecture (Detailed)

The proposed CNN model is a customized 3D Convolutional Neural Network (as depicted in Figure 1 of the paper). Its architecture is built to focus on picking out important features, which plays a significant role in accurately spotting tumors.

<img src="https://github.com/raahulmaurya1/Brain_tumar/blob/cadd16d51514baa5b773762a95ee671c8d04bf20/CNN.drawio%20(1).png" width="800" height="600"/>


The detailed architecture of the proposed tailored CNN model is as follows:

| Layer (Type) | Filter Size | Kernel Size | Strides | Output Shape | Number Of Parameters |
| :----------- | :---------- | :---------- | :------ | :----------- | :------------------- |
| Conv2D | 32 | $3\times3$ | | $112\times112\times16$ | 448 |
| Conv2D(1) | 32 | $3\times3$ | | $56\times56\times16$ | 2320 |
| MaxPooling2D | | | $2\times2$ | $28\times28\times16$ | 0 |
| Conv2D (2) | 64 | $3\times3$ | | $14\times14\times32$ | 4640 |
| Conv2D (3) | 64 | $3\times3$ | | $7\times7\times32$ | 9248 |
| MaxPooling2D (1) | | | $2\times2$ | $4\times4\times32$ | 0 |
| Conv2D (4) | 128 | $3\times3$ | | $2\times2\times64$ | 18496 |
| Conv2D (5) | 128 | $3\times3$ | | $1\times1\times64$ | 36928 |
| MaxPooling2D (2) | | | $2\times2$ | $1\times1\times64$ | 0 |
| Flatten | | | | 64 | 0 |
| Dense | | | | 256 | 16640 |
| Dense (1) | | | | 128 | 32896 |
| Dense (2) | | | | 1 | 129 |
| **Total Parameters** | | | | | **121744** |
| **Trainable Parameters** | | | | | **693505** |
| **Non-trainable Parameters** | | | | | **0** |

The input size is reshaped to $(512\times1024\times3)$. The network starts with a convolutional layer of 32 filters, each $(3\times3)$ kernel size, outputting $(112\times112\times16)$. Subsequent layers include more convolutional layers with increasing filter sizes and max-pooling layers for dimensionality reduction. The final layers consist of flattened output passed through fully connected (Dense) layers, with the last layer employing a softmax activation function for precise predictions.

## 📊 Performance & Results

The proposed customized CNN model achieved an impressive **classification accuracy of 98.03%**.
<img src="https://github.com/raahulmaurya1/Brain_tumar/blob/a38662c71028796162d31cf8591269ef3d865c62/brain.png" alt="Brain" width="1000"/>

The model's performance is further evidenced by its **confusion matrix** (as seen in Figure 3 of the paper), which highlights important metrics such as True Positives and False Positives.
* **True Positives**: Instances of brain tumor correctly identified by the model.
* **False Positives**: Incorrectly classified tumors that are actually healthy cases.

**Confusion Matrix Values**:

## 📉 Confusion Matrix

<img src="https://github.com/raahulmaurya1/Brain_tumar/blob/6d1436a3a39571743df276b890f70ace73fe074b/confusion%20matrix.png" alt="Confusion Matrix" width="400"/>

## 📈 Loss & Accuracy

<img src="https://github.com/raahulmaurya1/Brain_tumar/blob/a38662c71028796162d31cf8591269ef3d865c62/loss.png" alt="Training Performance" width="400"/>
<img src="https://github.com/raahulmaurya1/Brain_tumar/blob/a38662c71028796162d31cf8591269ef3d865c62/accuracy.png" alt="Training Performance" width="400"/>


The training and testing graphical representations (Figure 4 and Figure 5 in the paper) show consistent learning with minimal overfitting, indicating robust generalization. The accuracy generally improves over epochs, and loss values decrease, reflecting improved performance.

## 📈 Comparison with Existing Models

The proposed customized CNN model demonstrates superior performance compared to several existing methods:

| Existed Method | Classification Model | Accuracy |
| :------------- | :------------------- | :------- |
| Shamshad et al. [19] | VGG-16 | 97% |
| Suryawanshi et al. [23] | CNN-SVM | 95.16% |
| Hossain et al. [6] | IVX-16 | 96.94% |
| Ghosh et al. [4] | XGBOOST | 90% |
| Hammad et al. [5] | CNN model with 8 layers | 96.86% |
| **Proposed Model** | **CNN** | **98.03%** |

* **VGG-16 (Shamshad et al.)**: Achieved 97% accuracy, which is 1.03% lower than our model's 98.03%.
* **CNN-SVM Hybrid (Suryawanshi et al.)**: Achieved approximately 95.16%, falling short by 2.87% compared to our model.
* **Transfer Learning with IVX-16 (Hossain et al.)**: Achieved about 96.94%, with our model outperforming it by 1.09%.
* **XGBoost (Ghosh et al.)**: Achieved 90%, which is 8.03% lower than our model. This highlights the advantage of deep learning frameworks over conventional machine learning for intricate patterns in medical imaging.
* **CNN (Hammad et al.)**: Attained approximately 96.86%, remaining 1.17% lower than our model's accuracy. This small improvement demonstrates the importance of our model's specific architectural choices.

## 📦 Dependencies

The notebook explicitly imports the following Python libraries:

* `tensorflow`
* `cv2` (OpenCV)
* `keras`
* `numpy`
* `pandas`
* `random`
* `os`
* `matplotlib.image`
* `matplotlib.pyplot`
* `plotly.offline`
* `sklearn.utils`
* `sklearn.model_selection`
* `PIL.Image`
* `sklearn.preprocessing`

## 🤝 Contributing

We ❤️ contributions!  
Fork the repo → make your changes → submit a PR.

---

## 📜 License

Licensed under the [MIT License](LICENSE)

---

## 📬 Contact

Created by Rahul Maurya 
📧 Email: raahulmaurya2@gmail.com  
🔗 GitHub: @raahulmaurya1
