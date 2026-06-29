import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "../sonuclar/loglar"
OUT_DIR = "../sunum/grafikler"
CSV_PATH = "../sonuclar/sonuclar.csv"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def load_monitor_data(model_prefix):
    """Belirli bir model öneki ile başlayan klasörlerdeki monitor.csv dosyalarını bulur ve okur."""
    dfs = []
    if not os.path.exists(LOG_DIR): return dfs
    for folder in os.listdir(LOG_DIR):
        if folder.startswith(model_prefix) and "seed" in folder:
            csv_file = os.path.join(LOG_DIR, folder, "monitor.csv")
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file, skiprows=1)
                    # 'r': reward, 'l': length, 't': time
                    df['cumulative_steps'] = df['l'].cumsum()
                    dfs.append(df)
                except Exception as e:
                    print(f"Okuma hatası: {csv_file} - {e}")
    return dfs

def plot_learning_curve_with_std(dfs, title, filename):
    if not dfs:
        print(f"Veri bulunamadı: {title}")
        return

    plt.figure(figsize=(10, 6))
    
    # Ortak bir X ekseni (Adım) oluşturup verileri enterpole edelim
    max_steps = min([df['cumulative_steps'].max() for df in dfs])
    common_x = np.linspace(0, max_steps, 500)
    
    interpolated_rewards = []
    for df in dfs:
        # Hareketli ortalama al (window=100)
        smoothed_r = df['r'].rolling(window=100, min_periods=1).mean()
        # Ortak X eksenine oturt
        interp_r = np.interp(common_x, df['cumulative_steps'], smoothed_r)
        interpolated_rewards.append(interp_r)
        
    rewards_matrix = np.array(interpolated_rewards)
    mean_r = rewards_matrix.mean(axis=0)
    std_r = rewards_matrix.std(axis=0)
    
    plt.plot(common_x, mean_r, label=f"Ortalama Ödül ({len(dfs)} Seed)", color='blue', linewidth=2)
    plt.fill_between(common_x, mean_r - std_r, mean_r + std_r, color='blue', alpha=0.2, label='±1 Std Sapma')
    
    plt.title(title)
    plt.xlabel("Kümülatif Adım Sayısı")
    plt.ylabel("Bölüm Getirisi (Episode Reward)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()
    print(f"Oluşturuldu: {filename}")

def plot_eval_curve():
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH)
    
    plt.figure(figsize=(10, 6))
    # Sadece seed modellerini al
    seed_df = df[df['Seed'].str.isnumeric() == True]
    if len(seed_df) > 0:
        plt.bar(seed_df['Model'], seed_df['Ortalama_Skor'], yerr=seed_df['Standart_Sapma'], capsize=5, color='orange')
        plt.title("2. Test (Eval) Eğrisi: Modellerin Deterministik Koşusu")
        plt.xlabel("Model")
        plt.ylabel("Ortalama Skor")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "2_test_egrisi.png"))
        plt.close()
        print("Oluşturuldu: 2_test_egrisi.png")

def plot_baseline_comparison():
    dfs_ppo = load_monitor_data("RECURRENT_PPO")
    dfs_base = load_monitor_data("PPO_baseline")
    dfs_dqn = load_monitor_data("DQN_baseline")
    
    plt.figure(figsize=(10, 6))
    
    def plot_group(dfs, label, color):
        if not dfs: return
        max_steps = min([df['cumulative_steps'].max() for df in dfs])
        common_x = np.linspace(0, max_steps, 500)
        interp_list = []
        for df in dfs:
            smoothed = df['r'].rolling(window=100, min_periods=1).mean()
            interp_list.append(np.interp(common_x, df['cumulative_steps'], smoothed))
        mean_y = np.mean(interp_list, axis=0)
        std_y = np.std(interp_list, axis=0)
        plt.plot(common_x, mean_y, label=label, color=color)
        plt.fill_between(common_x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.1)

    plot_group(dfs_ppo, "RecurrentPPO (Ajan)", "blue")
    plot_group(dfs_base, "Vanilla PPO (Baseline)", "red")
    plot_group(dfs_dqn, "DQN (Baseline)", "green")
    
    plt.title("5. Baseline Karşılaştırması")
    plt.xlabel("Kümülatif Adım Sayısı")
    plt.ylabel("Bölüm Getirisi")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "5_baseline_karsilastirma.png"))
    plt.close()
    print("Oluşturuldu: 5_baseline_karsilastirma.png")

def get_loss_from_tfevents(folder_path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None, None
        
    for file in os.listdir(folder_path):
        if "events.out.tfevents" in file:
            event_file = os.path.join(folder_path, file)
            ea = EventAccumulator(event_file)
            ea.Reload()
            # SB3 logs loss under 'train/loss' for DQN or 'train/loss' / 'train/value_loss' for PPO
            loss_key = None
            for key in ea.Tags()['scalars']:
                if 'loss' in key.lower():
                    loss_key = key
                    break
            
            if loss_key:
                events = ea.Scalars(loss_key)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                return steps, vals
    return None, None

def plot_loss_curve():
    dfs_loss = []
    if not os.path.exists(LOG_DIR): return
    
    # Tüm seed'leri tara
    for folder in os.listdir(LOG_DIR):
        if folder.startswith("RECURRENT_PPO") and "seed" in folder:
            # Model klasörünün içindeki PPO_x (tensorboard klasörü) veya direkt kendi içindeki tfevents aranır
            tb_dir = os.path.join(LOG_DIR, folder)
            steps, vals = get_loss_from_tfevents(tb_dir)
            
            # Alt klasörlere de bak (sb3 bazen alt klasöre yazar)
            if not steps:
                for sub in os.listdir(tb_dir):
                    sub_path = os.path.join(tb_dir, sub)
                    if os.path.isdir(sub_path):
                        steps, vals = get_loss_from_tfevents(sub_path)
                        if steps: break
                        
            if steps and vals:
                df = pd.DataFrame({'step': steps, 'loss': vals})
                dfs_loss.append(df)
                
    if not dfs_loss:
        print("[UYARI] Tensorboard event dosyalarından loss verisi okunamadı. (Tensorboard kütüphanesi eksik veya log henüz oluşmamış olabilir).")
        return

    plt.figure(figsize=(10, 6))
    max_steps = min([df['step'].max() for df in dfs_loss])
    common_x = np.linspace(0, max_steps, 300)
    
    interp_list = []
    for df in dfs_loss:
        smoothed = df['loss'].rolling(window=10, min_periods=1).mean()
        interp_list.append(np.interp(common_x, df['step'], smoothed))
        
    mean_y = np.mean(interp_list, axis=0)
    std_y = np.std(interp_list, axis=0)
    
    plt.plot(common_x, mean_y, label="Ortalama Loss (5 Seed)", color='purple')
    plt.fill_between(common_x, mean_y - std_y, mean_y + std_y, color='purple', alpha=0.2, label="±1 Std Sapma")
    
    plt.title("3. Loss (Kayıp) Eğrisi")
    plt.xlabel("Adım Sayısı")
    plt.ylabel("Kayıp (Loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "3_loss_egrisi.png"))
    plt.close()
    print("Oluşturuldu: 3_loss_egrisi.png")

def plot_hyperparameters():
    if not os.path.exists(CSV_PATH): return
    df = pd.read_csv(CSV_PATH)
    
    # Sadece seed 1'in hiperparametre testlerini alalım
    hp_df = df[(df['Seed'] == '1') | (df['Seed'] == 1)]
    if len(hp_df) < 2:
        return
        
    plt.figure(figsize=(12, 6))
    
    # Model isimlerini x ekseni olarak kullanarak skorları çiz
    x_labels = []
    for _, row in hp_df.iterrows():
        name = f"LR:{row['Param_LR']} Ent:{row['Param_Ent']} G:{row['Param_Gamma']}"
        x_labels.append(name)
        
    plt.bar(x_labels, hp_df['Ortalama_Skor'], yerr=hp_df['Standart_Sapma'], capsize=5, color='teal')
    plt.title("4. Hiperparametre Duyarlılık Analizi (Keşif ve Sömürü Etkisi)")
    plt.xlabel("Hiperparametre Varyasyonları (Öğrenme Oranı, Entropi, Gamma)")
    plt.ylabel("Test Skoru (Ortalama ± Std)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "4_hiperparametre_duyarlilik.png"))
    plt.close()
    print("Oluşturuldu: 4_hiperparametre_duyarlilik.png")

if __name__ == "__main__":
    print("Tüm zorunlu 5 grafiğin çizimi başlatılıyor...")
    
    # 1. Öğrenme Eğrisi
    dfs = load_monitor_data("RECURRENT_PPO")
    plot_learning_curve_with_std(dfs, "1. Öğrenme Eğrisi (RecurrentPPO)", "1_ogrenme_egrisi.png")
    
    # 2. Test Eğrisi
    plot_eval_curve()
    
    # 3. Loss Eğrisi (Tensorboard tfevents üzerinden okunur)
    plot_loss_curve()
    
    # 4. Hiperparametre Duyarlılık Grafiği
    plot_hyperparameters()
    
    # 5. Baseline Karşılaştırması
    plot_baseline_comparison()
    
    print("TÜM GRAFİKLER sunum/grafikler/ klasörüne BAŞARIYLA KAYDEDİLDİ!")
