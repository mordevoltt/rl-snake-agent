# Snake RL Projesi

Bu dizin, Snake Reinforcement Learning projesinin kaynak kodlarını içerir.

## Dosyalar
- `env/snake_env.py`: Gymnasium tabanlı Snake ortamı.
- `train.py`: PPO modelini farklı seed ve hiperparametrelerle eğiten ana dosya.
- `evaluate.py`: Modelleri test eden ve deterministic (greedy) sonuçları çıkaran dosya.
- `config.yaml`: Ortam ve ajan hiperparametreleri.
- `requirements.txt`: Python kütüphane gereksinimleri.
- `seeds.txt`: Kullanılacak rastgelelik tohumları.
- `run_all.sh`: Eğitimi ve değerlendirmeyi başlatan otomasyon betiği.

## Kullanım
Eğitimi başlatmak için sadece terminalde sh betiğini çalıştırın:
```bash
bash run_all.sh
```
Sonuçlar `../sonuclar/sonuclar.csv` içerisine yazılır.
Loglar `../sonuclar/loglar/` dizininde Tensorboard formatında biriktirilir.
Modeller `../models/` altına otomatik kaydedilir.
