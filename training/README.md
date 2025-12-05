### Structure
.
├── images
│   └── 2.jpg
├── model_v0.0.1.pt
├── pretrained_model.pt
├── processed_images
│   └── 2.jpg
├── process_images.py
├── README.md
├── robust_optimization.py
├── test.tsv
├── train_multitask.py
└── train.tsv

### Execution
```bash
cp from_source_folder/enet_b0_8_va_mtl.pt ./pretrained_model.pt
touch train.tsv test.tsv
python3 process_images.py
python3 train_multitask.p
```
