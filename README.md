## Dependencies

- Python 3.9
- PyTorch 1.13.0
- dgl 0.9.1


## Datasets
| Dataset          | # Nodes | # Edges | # Classes | # Features |
| ---------------- | ------- | ------- | --------- | ---------- |
| Cora             | 2,708   | 10,556  | 7         | 1,433      |
| Citeseer         | 3,327   | 9,228   | 6         | 3,703      |
| Pubmed           | 19,717  | 88,651  | 3         | 500        |
| Amazon-Computer  | 13,752  | 574,418 | 10        | 767        |
| Amazon-Photo     | 7,650   | 287,326 | 8         | 745        |
| Coauthor-CS      | 18,333  | 327,576 | 15        | 6,805      |
| Coauthor-Physics | 34,493  | 991,848 | 5         | 8,451      |

## Usage
To run the codes in the terminal, use the following commands:

Cora
python train.py --dataname cora --epochs 100  --lr2 1e-2 --wd2 1e-4 --scale 100  --multiple 1

Citeseer
python train.py --dataname citeseer --epochs 25 --n_layers 1 --lr2  1e-2  --wd2 1e-1 --scale 100 --multiple 2

Pubmed  
python train.py --dataname pubmed --epochs 50 --lr2 1e-2 --wd2 1e-4  --scale 1000 --multiple 0

Amazon-Photo
python train.py --dataname photo --epochs 50  --lr2 1e-3 --wd2  1e-5  --scale  100 --multiple 1

Amazon-Computer
python train.py --dataname comp --epochs 50 --lr2 1e-2 --wd2 1e-4  --scale  100 --multiple 1

Coauthor-CS
python train.py --dataname cs --epochs 50 --lr2 1e-2 --wd2 0 --scale  100 --multiple 2 --dfr 0.2 --der 0.2

Coauthor- physics
python train.py --dataname physics --epochs 50 --gpu -1 --lr2 5e-2 --wd2 0 --scale  100 --multiple 1 --ratio 0.1
