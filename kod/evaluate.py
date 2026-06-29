import os
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.snake_env import SnakeEnv

def rastgele_politika_degerlendir(bolum_sayisi=50):
    ortam = SnakeEnv(difficulty=3)
    ortam = DummyVecEnv([lambda: ortam])
    
    skorlar = []
    
    for _ in range(bolum_sayisi):
        gozlem = ortam.reset()
        bitti_mi = False
        while not bitti_mi:
            eylem = [ortam.action_space.sample()]
            gozlem, odul, bitti_mi, bilgi = ortam.step(eylem)
            if bitti_mi[0]:
                skorlar.append(bilgi[0].get('score', 0))
    
    ortam.close()
    return np.mean(skorlar), np.std(skorlar)

def modeli_degerlendir(model_yolu, model_klasoru, bolum_sayisi=50):
    if not os.path.exists(model_yolu):
        print(f"Model bulunamadı: {model_yolu}")
        return None, None
        
    ortam = SnakeEnv(difficulty=3)
    ortam = DummyVecEnv([lambda: ortam])
    
    if model_klasoru.startswith("RECURRENT_PPO"):
        model = RecurrentPPO.load(model_yolu, env=ortam)
    elif model_klasoru.startswith("DQN"):
        model = DQN.load(model_yolu, env=ortam)
    else:
        model = PPO.load(model_yolu, env=ortam)
    
    skorlar = []
    
    # RecurrentPPO için LSTM durumları
    lstm_durumlari = None
    bolum_baslangiclari = np.ones((1,), dtype=bool)
    
    for _ in range(bolum_sayisi):
        gozlem = ortam.reset()
        bitti_mi = False
        lstm_durumlari = None
        bolum_baslangiclari = np.ones((1,), dtype=bool)
        while not bitti_mi:
            if model_klasoru.startswith("RECURRENT_PPO"):
                eylem, lstm_durumlari = model.predict(
                    gozlem, 
                    state=lstm_durumlari, 
                    episode_start=bolum_baslangiclari, 
                    deterministic=True
                )
            else:
                eylem, _ = model.predict(gozlem, deterministic=True)
            
            gozlem, odul, bitti_mi, bilgi = ortam.step(eylem)
            bolum_baslangiclari = bitti_mi
            
            if bitti_mi[0]:
                skorlar.append(bilgi[0].get('score', 0))
    
    ortam.close()
    return np.mean(skorlar), np.std(skorlar)

def ana_fonksiyon():
    ayristirici = argparse.ArgumentParser()
    ayristirici.add_argument("--episodes", type=int, default=50)
    argumanlar = ayristirici.parse_args()
    
    sonuclar = []
    
    print("Rastgele Politika (Baseline) Değerlendiriliyor...")
    rastgele_ortalama, rastgele_sapma = rastgele_politika_degerlendir(argumanlar.episodes)
    sonuclar.append({"Model": "Rastgele Politika", "Seed": "Yok", "Param_LR": "Yok", "Param_Ent": "Yok", "Param_Gamma": "Yok", "Ortalama_Skor": rastgele_ortalama, "Standart_Sapma": rastgele_sapma})
    print(f"Rastgele Politika - Ortalama Skor: {rastgele_ortalama:.2f} ± {rastgele_sapma:.2f}")

    modeller_dizini = "../models"
    if os.path.exists(modeller_dizini):
        for model_klasoru in os.listdir(modeller_dizini):
            if "PPO" in model_klasoru or "DQN" in model_klasoru:
                model_yolu = os.path.join(modeller_dizini, model_klasoru, "final_model.zip")
                if os.path.exists(model_yolu):
                    print(f"Değerlendiriliyor: {model_klasoru}...")
                    ortalama_skor, standart_sapma = modeli_degerlendir(model_yolu, model_klasoru, argumanlar.episodes)
                    
                    parcalar = model_klasoru.split("_")
                    tohum = "Yok"
                    ogrenme_orani = "Yok"
                    entropi = "Yok"
                    indirim = "Yok"
                    for p in parcalar:
                        if p.startswith("seed"): tohum = p[4:]
                        if p.startswith("lr"): ogrenme_orani = p[2:]
                        if p.startswith("ent"): entropi = p[3:]
                        if p.startswith("gamma"): indirim = p[5:]
                        
                    sonuclar.append({
                        "Model": model_klasoru,
                        "Seed": tohum,
                        "Param_LR": ogrenme_orani,
                        "Param_Ent": entropi,
                        "Param_Gamma": indirim,
                        "Ortalama_Skor": ortalama_skor,
                        "Standart_Sapma": standart_sapma
                    })
                    print(f"{model_klasoru} - Ortalama Skor: {ortalama_skor:.2f} ± {standart_sapma:.2f}")

    veri_cercevesi = pd.DataFrame(sonuclar)
    cikti_yolu = "../sonuclar/sonuclar.csv"
    if not os.path.exists("../sonuclar"):
        os.makedirs("../sonuclar")
    veri_cercevesi.to_csv(cikti_yolu, index=False)
    print(f"\nTüm sonuçlar {cikti_yolu} dosyasına kaydedildi.")

if __name__ == "__main__":
    ana_fonksiyon()
