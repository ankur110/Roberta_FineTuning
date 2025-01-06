# Fine-Tuning RoBERTa: Full Training vs Parameter-Efficient Fine-Tuning (PEFT)

## Introduction

Fine-tuning pre-trained language models like **RoBERTa** can significantly enhance performance on various natural language processing (NLP) tasks. However, full fine-tuning of large models is resource-intensive. **Parameter-Efficient Fine-Tuning (PEFT)** offers an alternative by updating only a small subset of parameters, reducing computational costs while maintaining performance.

This project compares the **full fine-tuning** of RoBERTa with PEFT methods using **LoRA (Low-Rank Adaptation)**. The trade-offs between these approaches are demonstrated on the **AG News** classification dataset.

---

## Project Structure

The repository contains the following files:

1. `full-finetuning.ipynb`: Full fine-tuning of the RoBERTa-large model.
2. `peft-roberta.ipynb`: Parameter-Efficient Fine-Tuning (PEFT) using LoRA.
3. `peft_vs_fulltraining.ipynb`: Performance comparison of the two approaches.

---

## Dataset and Libraries

### Dataset
- **AG News**: A popular dataset for news topic classification, containing four categories: World, Sports, Business, and Sci/Tech.

### Libraries
The following Python libraries are used in this project:
- `transformers`: Hugging Face's library for NLP models.
- `datasets`: Dataset handling and preprocessing.
- `peft`: Implementation of Parameter-Efficient Fine-Tuning techniques.
- `trl`: Tools for reinforcement learning with transformers.
- `evaluate`: Metrics computation.
- `scikit-learn`: Additional evaluation metrics.
- `accelerate`: Optimized training for distributed setups.
- `bitsandbytes`: 8-bit optimizers for efficient training.
- `huggingface_hub`: Access to Hugging Face's model hub.
- `scipy`: Scientific computing.
- `tensorboard`: Visualizing training metrics.
- `matplotlib`: Plotting and visualization.
- `sacrebleu`: BLEU score evaluation for translation tasks.

---

## Notebooks Overview

### 1. Full Fine-Tuning (`full-finetuning.ipynb`)
This notebook performs full fine-tuning of the RoBERTa-large model on the AG News classification task. All model parameters are updated during training.

#### Training Metrics (Sample):
| Step  | Training Loss | Validation Loss | Accuracy | F1     | Precision | Recall  |
|-------|---------------|-----------------|----------|--------|-----------|---------|
| 500   | 0.315800      | 0.337210       | 0.901316 | 0.900811 | 0.905600  | 0.901316 |
| 5000  | 0.189300      | 0.194171       | 0.943158 | 0.943018 | 0.943037  | 0.943158 |
| 7000  | 0.139800      | 0.196201       | 0.949737 | 0.949613 | 0.949593  | 0.949737 |

### 2. Parameter-Efficient Fine-Tuning (`peft-roberta.ipynb`)
This notebook demonstrates PEFT using **LoRA** to fine-tune the RoBERTa-large model. Only a small subset of parameters is updated.

#### Training Metrics (Sample):
| Step  | Training Loss | Validation Loss | Accuracy | F1     | Precision | Recall  |
|-------|---------------|-----------------|----------|--------|-----------|---------|
| 1000  | 0.267400      | 0.321892       | 0.901842 | 0.901505 | 0.901446  | 0.901842 |
| 5000  | 0.236400      | 0.259376       | 0.914474 | 0.914257 | 0.914906  | 0.914474 |
| 7000  | 0.190400      | 0.248881       | 0.916053 | 0.915737 | 0.915774  | 0.916053 |

### 3. Comparison (`peft_vs_fulltraining.ipynb`)
This notebook evaluates and compares the performance of the fully fine-tuned model and the PEFT model on the test dataset.

#### Test Accuracy:
| Model                 | Accuracy |
|-----------------------|----------|
| **Full Training**     | 95.16%   |
| **PEFT (LoRA)**       | 93.16%   |

---

## Performance Comparison

### Training Time and Resources
| Aspect                | Full Fine-Tuning     | PEFT                   |
|-----------------------|----------------------|------------------------|
| **Training Time**     | Longer              | Faster                 |
| **GPU Memory Usage**  | High                | Low                    |
| **Resource Demand**   | High                | Suitable for limited resources |

### Test Accuracy
- **Full Training Model**: 95.16%
- **PEFT Model**: 93.16%

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ankur110/Roberta_FineTuning.git
   ```

2. Install required libraries:
   ```bash
   pip install transformers datasets peft trl evaluate scikit-learn accelerate bitsandbytes huggingface_hub scipy tensorboard matplotlib sacrebleu
   ```

3. Run the Jupyter notebooks in the following order:
   - `full-finetuning.ipynb`
   - `peft-roberta.ipynb`
   - `peft_vs_fulltraining.ipynb`

4. Customize the training parameters in the notebooks as needed.

---

## Results and Insights
- **Full fine-tuning** achieves higher accuracy but requires significantly more computational resources.
- **PEFT (LoRA)** offers a cost-efficient alternative with slightly lower accuracy, making it ideal for resource-constrained environments.

---

## References
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

