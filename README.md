# LLR Attack

This is the Github Repo of the paper "Label Leakage in Regression Federated Learning using Cryptographic Tools".

### Packages

In order to be able to run the code, the following packages are required:
- PyTorch and Torchvision (and its dependancies) (the code run with cpu only or gpu versions)
- Hydra (and its dependancies)
- SageMath
- Scikit-Learn

### Description

```bash

$ tree
.
├── outputs           --> result directory containing information about the experiments
├── config
│   ├── default.yaml  --> default configuration
│   ├── experiment    --> configuration files to run our experiments
│   └── hydra         --> configuration files of Hydra
├── main.py           
├── README.md
└── solving_hssp      --> files to execute HSSP Attacks
```

To run the default config, simply run `python main.py`. It will attack the Boston Housing dataset with a batch size of 8. The results will appear in the `outputs` directory.

