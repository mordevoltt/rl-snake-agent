#!/bin/bash

# Snake RL Eğitim ve Değerlendirme Betiği
# Bu betik, ödev gereksinimlerine göre 5 farklı seed ile RecurrentPPO eğitimi,
# Vanilla PPO ve DQN Baseline eğitimleri,
# ardından Seed 1 üzerinde hiperparametre duyarlılık analizi yapar.

echo "--- 1. RecurrentPPO Eğitimleri (5 Seed) Başlıyor ---"
while read seed; do
  echo "Eğitim başlatılıyor: RecurrentPPO Seed $seed"
  python train.py --algo recurrent_ppo --seed "$seed" --timesteps 500000
done < seeds.txt

echo "--- 2. Vanilla PPO ve DQN Baseline Eğitimleri Başlıyor ---"
echo "Vanilla PPO eğitiliyor..."
python train.py --algo ppo --seed 1 --baseline --timesteps 500000

echo "DQN eğitiliyor..."
python train.py --algo dqn --seed 1 --baseline --timesteps 500000

echo "--- 3. Hiperparametre Duyarlılık Analizi (RecurrentPPO Seed 1) Başlıyor ---"

# Değişken 1: Öğrenme Oranı (Learning Rate)
echo "Hiperparametre: LR = 0.0001"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0001 --ent_coef 0.02 --timesteps 500000

echo "Hiperparametre: LR = 0.001"
python train.py --algo recurrent_ppo --seed 1 --lr 0.001 --ent_coef 0.02 --timesteps 500000

# Değişken 2: Keşif Katsayısı (Entropy Coefficient)
echo "Hiperparametre: Ent_Coef = 0.005"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0003 --ent_coef 0.005 --timesteps 500000

echo "Hiperparametre: Ent_Coef = 0.05"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0003 --ent_coef 0.05 --timesteps 500000

# Değişken 3: İndirim Faktörü (Gamma)
echo "Hiperparametre: Gamma = 0.90"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0003 --ent_coef 0.02 --gamma 0.90 --timesteps 500000

echo "Hiperparametre: Gamma = 0.95"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0003 --ent_coef 0.02 --gamma 0.95 --timesteps 500000

echo "--- 4. Modellerin Değerlendirilmesi (Test) ---"
python evaluate.py --episodes 50

echo "--- 5. Sonuç Grafiklerinin Çizilmesi ---"
python grafik_ciz.py

echo "Tüm işlemler tamamlandı! Sonuçlar sonuclar/sonuclar.csv dosyasına, grafikler sunum/grafikler/ klasörüne kaydedildi."
