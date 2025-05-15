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

Update your dataset path accordingly before training or evaluation in main_SSL_LA.py.

```python
parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/LA/',
    help='Change this to user\'s full directory address of LA database (ASVspoof2019 for training & development, ASVspoof2021 for evaluation).')
'''
% database_path/
%   |- LA
%      |- ASVspoof2021_LA_eval/flac
%      |- ASVspoof2019_LA_train/flac
%      |- ASVspoof2019_LA_dev/flac
'''

parser.add_argument('--protocols_path', type=str, default='database/',
    help='Change with path to user\'s LA database protocols directory address')
'''
% protocols_path/
%   |- ASVspoof_LA_cm_protocols
%      |- ASVspoof2021.LA.cm.eval.trl.txt
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

```bash
python main_SSL_LA_2019.py --track=LA --lr=0.000001 --batch_size=4 --num_epochs=100
```

### 📊 Testing

```bash
python main_SSL_LA_2019.py --track=LA --is_eval --eval --model_path='models/model_LA_WCE_15_4_1e-06/epoch_14.pth' --eval_output='eval_score.txt'

python main_SSL_DF.py --track=DF --is_eval --eval --model_path='/path/to/your/best_SSL_model_LA.pth' --eval_output='eval_CM_scores_file_SSL_DF.txt'
```

---

## 📈 Results (Pre-trained Model)

Compute the EER(%) using the evaluation dataset:
```bash
python evaluate_2021_LA.py Score_LA.txt ./keys eval
python evaluate_2021_DF.py Score_DF.txt ./keys eval
```

---

## 📌 Experimental Notes

This if you want to train without using SA according to main paper experiment
```bash
python main_SSL_LA_2019_tanpa-sa.py --track=LA --lr=0.000001 --batch_size=4 --num_epochs=100 --loss=WCE --algo=3 --comment=ssl-sa3
```

This if you want to eval model that you have been train before and eval one by one the model (perfolder)
```bash
python main_SSL_LA_2019_tanpa-sa.py --track=LA --is_eval --eval --models_folder='models/model_LA_WCE_100_4_1e-06_ssl-da1/' --eval_output='score_indo/ssl-da1'
```

- **`check_eer.py`**: Computes EER directly per model without t-DCF, adapted from `evaluate_2021_LA.py`.
- **`check_eer_pefolder.py`**: Aggregates per-folder model scores into an Excel file, an extension of `check_eer.py`.

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
@inproceedings{tak2022automatic,
  title={Implementasi SSL dan EER pada suara Bahasa Indonesia},
  author={Al Khairy, Rafid},
  year={2024}
}
```

---

**📌 [Your Notes & Future Improvements]**  
*(Tambahkan catatan eksperimen, hasil uji coba, atau ide pengembangan di sini...)*
Ubah path nya jadi fix, gaperlu ubah ubah lagi
buat preprocessing data supaya gampang
sesuaikan path di eval, hapus kata kata 2019 prefix atau apapun itu

pip install tensorboard
tensorboard --logdir=logs
---

## 📝 Add Your Own Notes
📌 **Additional Experiments**: ................................................

📌 **New Training Configurations**: ................................................

📌 **Custom Dataset Links**: ................................................

📌 **Future Improvements**: ................................................

