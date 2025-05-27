# Automatic Speaker Verification Spoofing and Deepfake Detection

This repository contains our implementation of the paper published in the Speaker Odyssey 2022 workshop, **"Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation"**. This work produced state-of-the-art results on the challenging ASVspoof 2021 LA and DF datasets.

📄 **[Paper Link](https://arxiv.org/abs/2202.12233)**  
📂 **[Main Repository SSL](https://github.com/TakHemlata/SSL_Anti-spoofing.git)**
📂 **[Main Repository AASIST](https://github.com/clovaai/aasist)**

---

## 🚀 Installation

Clone the repository, create a conda environment, and install dependencies:

```bash
git clone https://github.com/Rothiii/ssl-antispoofing-bahasa-indonesia.git
cd ssl-antispoofing-bahasa-indonesia
conda create -n SSL_Spoofing python=3.7
conda activate SSL_Spoofing
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/pytorch/fairseq.git@a54021305d6b3c4c5959ac9395135f63202db8f1
pip install -r requirements.txt
```

---
## Configuration
### 📂 Dataset

Update your dataset path accordingly before training or evaluation in args_config.py.

```python
parser.add_argument('--database_path', type=str, default='dataset/LA/',
    help='Change this to user\'s full directory address of LA database (ASVspoof2019 for training & development, ASVspoof2021 for evaluation).')
'''
% dataset/
%   |- LA
%      |- ASVspoof2019_LA_eval/
%      |- ASVspoof2019_LA_train/
%      |- ASVspoof2019_LA_dev/
'''

parser.add_argument('--protocols_path', type=str, default='dataset/LA',
    help='Change with path to user\'s LA database protocols directory address')
'''
% dataset/LA
%   |- ASVspoof_LA_cm_protocols
%      |- ASVspoof2019.LA.cm.eval.trl.txt
%      |- ASVspoof2019.LA.cm.dev.trl.txt 
%      |- ASVspoof2019.LA.cm.train.trn.txt
'''
```

Dataset links:
- **[ASVspoof 2019 Dataset](https://datashare.is.ed.ac.uk/handle/10283/3336)**
- **[ASVspoof 2021 LA](https://zenodo.org/record/4837263#.YnDIinYzZhE)**
- **[ASVspoof 2021 DF](https://zenodo.org/record/4835108#.YnDIb3YzZhE)**

---

### Pre-trained wav2vec 2.0 XLSR (300M)

Put wav2vec 2.0 model file to inside this repository.

- **[wav2vec 2.0 XLSR (300M) Model](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)**

---

## 🎯 Training & Evaluation

### 🔥 Training

Train use wav2vec 2.0 (ssl)
```bash
python train-model.py --model ssl --sa --comment=your-comment
```

Train use sinclayer
```bash
python train-model.py --model sinclayer --sa
```

### 📊 Evaluation

```bash
python train-model.py --sa--is_eval --eval --model_path='models/your-folder/your-model.pth' --eval_output='scores/your-result.txt'

python train-model.py --sa --is_eval --eval --models_folder='models/[your models]/' --eval_output='score_indo/[folder output]'
```


---

## 📌 Experimental Notes

This if you want to train without using SA according to main paper experiment
```bash
python train-model.py --comment=your-comment
```

Eval specified model 
```bash
python train-model.py --is_eval --eval --model_path='models/your-folder/your-model.pth' --eval_output='scores/your-result.txt'
```

If you want to eval model that you have been train before and eval one by one the model (perfolder)
```bash
python train-model.py --is_eval --eval --models_folder='models/[your models]/' --eval_output='score_indo/[folder output]'
```

---

## 📈 Results
- **`check_eer.py`**: Computes EER directly per model without t-DCF, adapted from `evaluate_2021_LA.py`.
- **`check_eer_pefolder.py`**: Aggregates per-folder model scores into an Excel file, an extension of `check_eer.py`.

Compute the EER(%) using the evaluation dataset:
```bash
python check_eer.py your_score.txt dataset/LA
python check_eer_perfolder.py /path/to/score_folder /path/to/cm_protocols
```

---

## 📬 Contact

For any inquiries regarding this repository, please contact:
- **Rafid Al Khairy**: 11211068@student.itk.ac.id

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
```

```bibtex
@inproceedings{
  title={KLASIFIKASI AUDIO ANTI-SPOOFING BAHASA INDONESIA MENGGUNAKAN AASIST DAN WAV2VEC 2.0 DENGAN AUGMENTASI DATA},
  author={Al Khairy, Rafid},
  year={2025}
}
```

---

## 📝 Add Your Own Notes
📌 **Additional Experiments**: 
- Sample Rate change from 64600 to 130000
- Compare Learning Rate 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005
- Compare all Data Augmentation algoritm from 0 to 8 

📌 **Custom Dataset Links**:
Self Record and Generate TTS by Resemble.ai voice cloning with each speaker
- **[Self Record]()**
