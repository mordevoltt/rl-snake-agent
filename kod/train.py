import argparse
import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env.snake_env import SnakeEnv

with open("config.yaml", "r", encoding="utf-8") as f:
    ayarlar = yaml.safe_load(f)

def dogrusal_zamanlama(baslangic_degeri):
    def fonksiyon(kalan_ilerleme):
        return kalan_ilerleme * baslangic_degeri
    return fonksiyon

class YemekKaydediciGeriCagirim(BaseCallback):
    def __init__(self, log_yolu):
        super().__init__()
        self.log_yolu = log_yolu
        if not os.path.exists(os.path.dirname(self.log_yolu)): 
            os.makedirs(os.path.dirname(self.log_yolu))
        if not os.path.exists(self.log_yolu):
            with open(self.log_yolu, "w", encoding="utf-8") as dosya:
                dosya.write("Adım\tKırmızı\tSarı\tSkor\n")
        
    def _on_step(self) -> bool:
        for bilgi in self.locals.get("infos", []):
            if "episode" in bilgi.keys() or bilgi.get("terminal_observation") is not None:
                skor = bilgi.get("score", 0)
                kirmizi = bilgi.get("red_eaten", 0)
                sari = bilgi.get("gold_eaten", 0)
                with open(self.log_yolu, "a", encoding="utf-8") as dosya:
                    dosya.write(f"{self.num_timesteps}\t{kirmizi}\t{sari}\t{skor}\n")
        return True

class MufredatGeriCagirim(BaseCallback):
    def __init__(self, baslangic_zorlugu=1, verbose=0):
        super(MufredatGeriCagirim, self).__init__(verbose)
        self.son_zorluk = baslangic_zorlugu

    def _on_step(self) -> bool:
        guncel_adim = self.num_timesteps
        yeni_zorluk = 1
        if guncel_adim > 100000 and guncel_adim <= 250000:
            yeni_zorluk = 2
        elif guncel_adim > 250000:
            yeni_zorluk = 3

        if yeni_zorluk != self.son_zorluk:
            print(f"\n--- MÜFREDAT GÜNCELLENDİ: Zorluk Seviyesi {yeni_zorluk} Oldu ---")
            self.training_env.set_attr("difficulty", yeni_zorluk)
            self.son_zorluk = yeni_zorluk
        return True

def ortam_olustur(sira, tohum):
    def _baslat():
        ortam = SnakeEnv(difficulty=ayarlar['env']['difficulty'])
        ortam.reset(seed=tohum + sira)
        return ortam
    return _baslat

def egitimi_baslat(argumanlar):
    ogrenme_orani = argumanlar.lr if argumanlar.lr is not None else ayarlar['training']['learning_rate']
    kesif_katsayisi = argumanlar.ent_coef if argumanlar.ent_coef is not None else ayarlar['training']['ent_coef']
    indirim_faktoru = argumanlar.gamma if argumanlar.gamma is not None else ayarlar['training']['gamma']

    gamma_metni = f"_gamma{indirim_faktoru}" if argumanlar.gamma is not None else ""
    model_adi = f"{argumanlar.algo.upper()}_seed{argumanlar.seed}_ent{kesif_katsayisi}_lr{ogrenme_orani}{gamma_metni}"
    if argumanlar.baseline:
        model_adi = f"{argumanlar.algo.upper()}_baseline_seed{argumanlar.seed}"

    modeller_dizini = f"../models/{model_adi}"
    log_dizini = f"../sonuclar/loglar/{model_adi}"
    
    # %100 BİTİP BİTMEDİĞİNİ ANLAMA KONTROLÜ (KALDIĞI YERDEN DEVAM ETME)
    hedef_model_dosyasi = os.path.join(modeller_dizini, "final_model.zip")
    if os.path.exists(hedef_model_dosyasi):
        print(f"\n[ATLANDI] {model_adi} zaten %100 egitilmis ve 'final_model.zip' olusmus. Siradakine geciliyor...\n")
        return

    # Eğer klasör var ama içinde zip yoksa (yarım kalmışsa) veya hiç yoksa, klasörleri (yeniden) oluştur
    if not os.path.exists(modeller_dizini): os.makedirs(modeller_dizini)
    if not os.path.exists(log_dizini): os.makedirs(log_dizini)

    islemci_sayisi = ayarlar['training']['num_cpu']
    ortam = SubprocVecEnv([ortam_olustur(i, argumanlar.seed) for i in range(islemci_sayisi)])
    ortam = VecMonitor(ortam, filename=os.path.join(log_dizini, "monitor.csv"))



    if argumanlar.algo == "recurrent_ppo":
        model = RecurrentPPO(
            policy="MultiInputLstmPolicy" if "Dict" in str(ortam.observation_space) else "MlpLstmPolicy",
            env=ortam,
            verbose=1,
            seed=argumanlar.seed,
            tensorboard_log=log_dizini,
            learning_rate=dogrusal_zamanlama(ogrenme_orani),
            n_steps=ayarlar['training']['n_steps'],
            batch_size=ayarlar['training']['batch_size'],
            n_epochs=ayarlar['training']['n_epochs'],
            gamma=indirim_faktoru,
            gae_lambda=ayarlar['training']['gae_lambda'],
            ent_coef=kesif_katsayisi,
            clip_range=ayarlar['training']['clip_range'],
            policy_kwargs=dict(
                lstm_hidden_size=ayarlar['model']['lstm_hidden_size'],
                n_lstm_layers=ayarlar['model']['n_lstm_layers'],
                shared_lstm=ayarlar['model']['shared_lstm'],
                net_arch=dict(
                    pi=ayarlar['model']['pi_layers'],
                    vf=ayarlar['model']['vf_layers'],
                ),
            ),
            device="cuda" if argumanlar.cuda else "cpu"
        )
    elif argumanlar.algo == "dqn":
        model = DQN(
            policy="MlpPolicy",
            env=ortam,
            verbose=1,
            seed=argumanlar.seed,
            tensorboard_log=log_dizini,
            learning_rate=dogrusal_zamanlama(ogrenme_orani),
            batch_size=ayarlar['training']['batch_size'],
            gamma=indirim_faktoru,
            buffer_size=100000,
            learning_starts=1000,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            policy_kwargs=dict(
                net_arch=ayarlar['model']['pi_layers'],
            ),
            device="cuda" if argumanlar.cuda else "cpu"
        )
    else:  # vanilla PPO
        model = PPO(
            policy="MlpPolicy",
            env=ortam,
            verbose=1,
            seed=argumanlar.seed,
            tensorboard_log=log_dizini,
            learning_rate=dogrusal_zamanlama(ogrenme_orani),
            n_steps=ayarlar['training']['n_steps'],
            batch_size=ayarlar['training']['batch_size'],
            n_epochs=ayarlar['training']['n_epochs'],
            gamma=indirim_faktoru,
            gae_lambda=ayarlar['training']['gae_lambda'],
            ent_coef=kesif_katsayisi,
            clip_range=ayarlar['training']['clip_range'],
            policy_kwargs=dict(
                net_arch=dict(
                    pi=ayarlar['model']['pi_layers'],
                    vf=ayarlar['model']['vf_layers'],
                ),
            ),
            device="cuda" if argumanlar.cuda else "cpu"
        )

    kontrol_noktasi = CheckpointCallback(
        save_freq=50000,
        save_path=modeller_dizini,
        name_prefix=f"snake_{model_adi}"
    )
    yemek_kaydedici = YemekKaydediciGeriCagirim(os.path.join(log_dizini, "food_stats.txt"))
    mufredat_kaydedici = MufredatGeriCagirim(baslangic_zorlugu=ayarlar['env']['difficulty'])

    toplam_adim = argumanlar.timesteps if argumanlar.timesteps is not None else ayarlar['training']['total_timesteps']
    print(f"EĞİTİM BAŞLIYOR: {model_adi} | Adım: {toplam_adim}")

    model.learn(
        total_timesteps=toplam_adim,
        callback=[kontrol_noktasi, yemek_kaydedici, mufredat_kaydedici],
        tb_log_name=model_adi
    )

    model.save(f"{modeller_dizini}/final_model")
    print(f"\nEğitim Bitti! Model kaydedildi: {modeller_dizini}/final_model")

if __name__ == "__main__":
    ayristirici = argparse.ArgumentParser()
    ayristirici.add_argument("--algo", type=str, default="ppo", choices=["ppo", "recurrent_ppo", "dqn"], help="Algoritma seçimi")
    ayristirici.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    ayristirici.add_argument("--lr", type=float, default=None, help="Öğrenme oranı")
    ayristirici.add_argument("--ent_coef", type=float, default=None, help="Keşif (Entropy) katsayısı")
    ayristirici.add_argument("--gamma", type=float, default=None, help="İndirim faktörü (Gamma)")
    ayristirici.add_argument("--timesteps", type=int, default=None, help="Toplam eğitim adımı")
    ayristirici.add_argument("--baseline", action="store_true", help="Baseline koşusu olarak işaretle")
    ayristirici.add_argument("--cuda", action="store_true", help="CUDA kullan", default=True)
    
    argumanlar = ayristirici.parse_args()
    egitimi_baslat(argumanlar)
