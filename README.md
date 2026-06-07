# Secure Cross-Hospital Logistic Regression with MPC

This repository contains the code used for a bachelor thesis project on privacy-preserving logistic regression for cross-hospital medical ordering anomaly detection.

The project uses secure multi-party computation to train a logistic regression model on horizontally partitioned hospital data. The secure implementation is written for MP-SPDZ and uses the `semi2k` protocol in a semi-honest two-party setting.

## Repository description

Suggested GitHub description:

MP-SPDZ implementation of secure logistic regression for cross-hospital detection of medical ordering anomalies.

## Repository contents

* `preprocess.py`
  Preprocesses the dataset, constructs anomaly labels, splits the data between two parties, writes MP-SPDZ input files, and trains a plaintext baseline.

* `horizontal_lr_anomaly.mpc`
  MP-SPDZ source file for secure logistic regression training and evaluation.

* `README.md`
  Documentation for running the project.

* `LICENSE`
  MIT License for this repository.

## Project overview

The project uses the UCI Myocardial Infarction Complications dataset as a healthcare case study. The dataset does not contain direct labels for guideline noncompliance, so anomaly labels are constructed during preprocessing.

The preprocessing step selects 34 features in total:

* 9 clinical covariates
* 25 procedure-related features

Anomaly labels are created using a distance-based method over the procedure-related features. The top 20% most unusual records are labeled as anomalous.

The resulting dataset is split horizontally between two parties:

* Party 0 represents Hospital A
* Party 1 represents Hospital B

Each party contributes 800 patient records. The secure logistic regression model is then trained in MP-SPDZ without either party revealing its raw input data to the other party.

## Requirements

### Python dependencies

Install the required Python packages:


pip install pandas numpy scikit-learn


### MP-SPDZ

Clone MP-SPDZ from GitHub:

git clone https://github.com/data61/MP-SPDZ.git
cd MP-SPDZ

Follow the official MP-SPDZ installation instructions for your operating system.

On Linux, the quick setup can usually be started with:

Scripts/tldr.sh

## Dataset setup

Download the UCI Myocardial Infarction Complications dataset and place the CSV file at:

dataset_mpc/myocardial_infarction.csv


The expected structure is:


MP-SPDZ/
├── dataset_mpc/
│   └── myocardial_infarction.csv
├── Programs/
│   └── Source/
│       └── horizontal_lr_anomaly.mpc
├── Player-Data/
└── preprocess.py


## How to run

### 1. Add the files to MP-SPDZ

Place the MP-SPDZ program here:


MP-SPDZ/Programs/Source/horizontal_lr_anomaly.mpc


Place the preprocessing script in the root of the MP-SPDZ folder:


MP-SPDZ/preprocess.py


### 2. Run preprocessing

From the root of the MP-SPDZ folder, run:


python3 preprocess.py


This creates the MP-SPDZ input files:


Player-Data/Input-P0-0
Player-Data/Input-P1-0


The input order is important. For each sample, the preprocessing script writes all feature values first, followed by the label. This matches the input order used in `horizontal_lr_anomaly.mpc`.

### 3. Compile and run the secure program

From the root of the MP-SPDZ folder, run:


Scripts/compile-run.py -E semi2k -R 192 horizontal_lr_anomaly


This compiles and runs the secure logistic regression program using the `semi2k` protocol with a 192-bit ring.

## Main configuration

The main configuration used in the thesis is:

| Setting               | Value                        |
| --------------------- | ---------------------------- |
| MPC framework         | MP-SPDZ                      |
| Protocol              | `semi2k`                     |
| Security model        | Semi-honest                  |
| Number of parties     | 2                            |
| Total samples         | 1600                         |
| Samples per party     | 800                          |
| Number of features    | 34                           |
| Positive class        | Top 20% anomaly labels       |
| Learning rate         | 0.005                        |
| Epochs                | 150                          |
| Decision threshold    | 0.456                        |
| Fixed-point precision | `sfix.set_precision(32, 63)` |
| Ring size             | 192 bits                     |

## Output

The secure MP-SPDZ program prints:

* configuration details
* total positive and negative labels
* learned model bias
* learned model weights
* sample predicted probabilities
* confusion matrix
* accuracy
* precision
* recall
* F1 score

MP-SPDZ also reports runtime and communication statistics in the program output.

## Plaintext baseline

The preprocessing script also trains a plaintext logistic regression model using scikit-learn. This baseline is included as a reference for comparing predictive performance and overhead against the secure MP-SPDZ implementation.

## Notes

This repository contains the code needed to reproduce the preprocessing and secure logistic regression experiment. The dataset itself is not included and must be downloaded separately from the UCI Machine Learning Repository.

The secure implementation uses full-batch gradient descent. All feature values and labels are read as private inputs from the two parties. The feature matrix is represented using secret-shared fixed-point values, while the labels are represented as secret-shared integers.

## Citation

This project uses MP-SPDZ. If you use MP-SPDZ, please cite the official MP-SPDZ paper:

    @inproceedings{mp-spdz,
        author = {Marcel Keller},
        title = {{MP-SPDZ}: A Versatile Framework for Multi-Party Computation},
        booktitle = {Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security},
        year = {2020},
        doi = {10.1145/3372297.3417872},
        url = {https://doi.org/10.1145/3372297.3417872}
    }
    
## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Author

Michael Siswowijoto
