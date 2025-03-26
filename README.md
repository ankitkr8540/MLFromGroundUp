# MLFromGroundUp

A comprehensive collection of machine learning algorithms implemented from scratch in Python. This repository serves as both a learning resource and a reference for understanding the mathematical foundations behind popular machine learning techniques.

## Overview

This project aims to demystify machine learning by implementing core algorithms without relying on high-level libraries. By building these algorithms from their mathematical foundations, this repository provides insights into how machine learning methods actually work under the hood.

Each algorithm is implemented in Python with detailed documentation of the underlying mathematics, clear code structure, and example applications.

## Algorithms

### K-Nearest Neighbors (KNN)

A non-parametric classification algorithm that makes predictions based on the majority class of nearest neighbors in the feature space.

**Key features:**

- Distance-weighted prediction using 1/d² weighting
- Custom cross-validation implementation
- Feature scaling and dimensionality reduction

**Mathematical foundation:**

- Euclidean distance calculation
- Weighted voting mechanism
- Cross-validation accuracy metrics

[View KNN implementation →](./K-Nearest Neighbors/)

### K-Means Clustering

An unsupervised learning algorithm that partitions data into K distinct clusters based on distance to the nearest centroid.

**Key features:**

- Multiple initialization strategies (random, k-means++)
- Support for different distance metrics (Euclidean, cosine)
- Mini-batch processing for scalability
- Multiple runs to avoid local minima

**Mathematical foundation:**

- Centroid initialization techniques
- Distance calculation methods
- Assignment and update steps
- Convergence criteria and SSE objective function

[View K-Means implementation →](./kmeans/)

## Project Structure

```
MLFromGroundUp/
├── knn-classifier/
│   ├── knn_implementation.py
│   ├── README.md
│   └── knn-diagram.png
├── kmeans-clustering/
│   ├── image_clustering.py
│   ├── README.md
│   └── kmeans-diagram.png
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries (install via `pip install -r requirements.txt`):
  - numpy>=1.24.3
  - pandas>=2.2.0
  - matplotlib>=3.8.2
  - scikit-learn>=1.3.0 (used only for data preprocessing and comparison)

### Installation

```bash
# Clone the repository
git clone https://github.com/ankitkr8540/MLFromGroundUp.git
cd MLFromGroundUp

# Install dependencies
pip install -r requirements.txt
```

## Usage

Each algorithm directory contains detailed usage instructions and examples. Generally, you can run an implementation with:

```bash
cd algorithm-directory
python implementation_file.py
```

## Features

- **Pure Python implementations**: All algorithms are implemented using only basic Python libraries and NumPy for numerical operations
- **Detailed documentation**: Each implementation includes comprehensive explanations of the underlying mathematical concepts
- **Visualizations**: Visual aids to help understand algorithm behavior
- **Performance comparisons**: Benchmarks against standard library implementations
- **Educational focus**: Code clarity prioritized over optimization for learning purposes

## Contributing

Contributions are welcome! If you'd like to add a new algorithm implementation, improve existing code, or enhance documentation, please feel free to submit a pull request.

When contributing, please:

- Follow the existing code structure and documentation style
- Include mathematical foundations and explanations
- Add appropriate visualizations where helpful
- Ensure code is well-commented and readable

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- **Ankit Kumar** - [GitHub Profile](https://github.com/ankitkr8540)

## Acknowledgments

- Special thanks to the open-source community for inspiration and resources
- Academic resources and papers that provided theoretical foundations
