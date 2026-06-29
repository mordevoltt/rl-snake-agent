from gymnasium.utils.env_checker import check_env
from env.snake_env import SnakeEnv

# Çevreyi oluştur
env = SnakeEnv()

# Gymnasium resmi env_checker modülü ile doğrula
# Eğer bu fonksiyon hata vermeden çalışırsa, çevreniz resmi standartlara %100 uyumludur.
print("Gymnasium resmi uyumluluk testi başlatılıyor...")
try:
    check_env(env)
    print("\n[BAŞARILI] Uyumluluk testi başarıyla tamamlandı! Çevre (SnakeEnv) Gymnasium standartlarına tam uyumludur.")
except Exception as e:
    print(f"\n[HATA] Gymnasium uyumluluk testinde hata oluştu: {e}")
