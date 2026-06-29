Write-Host "--- 1. RecurrentPPO Eğitimleri (5 Seed) Başlıyor ---"
Get-Content seeds.txt | ForEach-Object {
    $seed = $_
    Write-Host "Eğitim başlatılıyor: RecurrentPPO Seed $seed"
    python train.py --algo recurrent_ppo --seed $seed --timesteps 500000
}

Write-Host "--- 2. Vanilla PPO ve DQN Baseline Eğitimleri Başlıyor ---"
Write-Host "Vanilla PPO eğitiliyor..."
python train.py --algo ppo --seed 1 --baseline --timesteps 500000

Write-Host "DQN eğitiliyor..."
python train.py --algo dqn --seed 1 --baseline --timesteps 500000

Write-Host "--- 3. Hiperparametre Duyarlılık Analizi (RecurrentPPO Seed 1) Başlıyor ---"

Write-Host "Hiperparametre: LR = 0.0001"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0001 --ent_coef 0.02 --timesteps 500000

Write-Host "Hiperparametre: LR = 0.001"
python train.py --algo recurrent_ppo --seed 1 --lr 0.001 --ent_coef 0.02 --timesteps 500000

Write-Host "Hiperparametre: Ent_Coef = 0.005"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0003 --ent_coef 0.005 --timesteps 500000

Write-Host "Hiperparametre: Ent_Coef = 0.05"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0003 --ent_coef 0.05 --timesteps 500000

Write-Host "Hiperparametre: Gamma = 0.90"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0003 --ent_coef 0.02 --gamma 0.90 --timesteps 500000

Write-Host "Hiperparametre: Gamma = 0.95"
python train.py --algo recurrent_ppo --seed 1 --lr 0.0003 --ent_coef 0.02 --gamma 0.95 --timesteps 500000

Write-Host "--- 4. Modellerin Değerlendirilmesi (Test) ---"
python evaluate.py --episodes 50

Write-Host "--- 5. Sonuç Grafiklerinin Çizilmesi ---"
python grafik_ciz.py

Write-Host "Tüm işlemler tamamlandı! Sonuçlar sonuclar/sonuclar.csv dosyasına, grafikler sunum/grafikler/ klasörüne kaydedildi."
