from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.snake_env import SnakeEnv
import numpy as np
import time
import os

def test():
    # Çevreyi oluştur (En zor seviye: 3)
    env = SnakeEnv(render_mode="human", difficulty=3)
    env = DummyVecEnv([lambda: env])
    
    # Modeller klasöründeki en son kaydedilen modeli otomatik bul
    models_dir = "../models/RECURRENT_PPO_seed1_ent0.02_lr0.0003"
    if not os.path.exists(models_dir):
        # Eğer özel klasör bulunamazsa genel models klasörüne bak
        models_dir = "../models"
        
    model_path = None
    if os.path.exists(models_dir):
        # Önce doğrudan models_dir içindeki zip dosyalarına bak
        files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
        if files:
            latest_model = sorted(files)[-1]
            model_path = os.path.join(models_dir, latest_model)
        else:
            # Alt klasörleri tara
            for root, dirs, filenames in os.walk(models_dir):
                zip_files = [f for f in filenames if f.endswith(".zip")]
                if zip_files:
                    latest_model = sorted(zip_files)[-1]
                    model_path = os.path.join(root, latest_model)
                    break
                    
    if not model_path or not os.path.exists(model_path):
        print("Model dosyası bulunamadı! Lütfen önce eğitimi başlatın veya modelin eğitildiğinden emin olun.")
        return

    try:
        model = RecurrentPPO.load(model_path, env=env)
        print(f"En güncel model yüklendi: {model_path}")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return

    obs = env.reset()

    # RecurrentPPO için LSTM gizli state başlat
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)  # İlk adımda episode başlıyor

    episode_count = 0

    while True:
        # Modelden eylem tahmini al (LSTM state ile birlikte)
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )

        obs, reward, done, info = env.step(action)

        # Bir sonraki adımda episode_start False olacak
        episode_starts = done

        time.sleep(0.1)

        if done[0]:
            episode_count += 1
            score = info[0].get('score', 0)
            reds = info[0].get('red_eaten', 0)
            golds = info[0].get('gold_eaten', 0)
            print(f"[Bölüm {episode_count}] Oyun bitti! Skor: {score} | Detay: {reds}K {golds}S")
            # LSTM state'i sıfırla (yeni episode başlarken)
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    test()
