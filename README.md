# LLM-Augmented-Document-Retrieval-System


## Overview
This project is to enhance information retrieval by classifying the relevancy of document-query pairs and generating textual justifications. It leverages the Longformer and GPT-2 models within a seq-to-seq framework, using the MS MARCO dataset for training and validation.

## Installation

```bash
# Clone the repository
git clone https://github.com/satyanshu404/LLM-Augmented-Document-Retrieval-System.git

# Navigate to the project directory
cd LLM-Augmented-Document-Retrieval-System

# Install required dependencies
pip install -r requirements.txt
```

## Usage

To run the classification and justification generation, use the `main.py` file. Provide the query and document:

```bash
python main.py
```

Ensure you provide `query` and `document`

## Experimental Setup

### For this project I have utilized this Environment
- **Operating System**: Ubuntu 22.04.2 LT
- **CPU**: Intel(R) Xeon(R) Silver 4309Y CPU @ 2.80GHz
- **RAM**: 128GB
- **GPU**: NVIDIA L40
- **Cuda Version**: 12.3
- **GPU Memory**: 46 GB

### Configuration
- **Model Configurations**: Longformer and GPT-2 integrated in a sequence-to-sequence architecture
- **Hyperparameters**:
  - Learning Rate: 2 × 10^(-5)
  - Training Batch Size: 2
  - Evaluation Batch Size: 2
  - Optimizer: Adam (β1 = 0.9, β2 = 0.999, ε = 1 × 10^(-8))
  - LR Scheduler: Linear
  - Number of Epochs: 50 for Longformer, 200 for Seq-to-Seq Model

## Results

The system's performance in terms of classification and justification generation is documented here, including confusion matrices and ROUGE scores for the test datasets.


## Limitations
The model does not provide strong justifications, potentially due to its smaller number of parameters. Additionally, its performance could diminish with inputs exceeding 16,000 tokens. Since it was trained on only 1,000 instances, its generalization ability may be compromised.
