import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import deque

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=20, difficulty=1):
        super(SnakeEnv, self).__init__()
        self.size = size  # OYUN ALANI BOYUTU (Örn: 20x20)
        self.difficulty = difficulty # 1: Sabit, 2: Yavaş, 3: Hızlı
        
        # ÖDÜL AYARLARI
        self.REWARD_DEATH = -30    # Ölme cezası: -50'den düşürüldü, diğer sinyalleri bastırmasın
        self.REWARD_STEP = -0.01   # Her adımda sabit küçük ceza (hayatta kal, beklemede değer yok)
        
        # ENGEL (WALLS) TANIMI: Oyun alanındaki sabit engeller
        self.obstacles = []
        self.window_size = 400
        self.cell_size = self.window_size // self.size
        
        # Action space: 0: Düz, 1: Sağ, 2: Sol
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 38 values
        # [danger×3, direction×4, food_type0×7, food_type1×7, food_type2×7, energy×1, radar×8, length×1]
        # Her yemek tipi için: [var_mı, sol, sağ, yukarı, aşağı, mesafe, kalan_sure] = 7
        self.observation_space = spaces.Box(low=0, high=1, shape=(38,), dtype=np.float32)

        self.max_energy = 200  # 100→200: Ajan daha uzun yaşayabilsin, enerji sistemi anlamlı hale gelsin
        self.visited_memory = 30  # Döngü cezası için takip edilen hücre sayısı

        # Yemek tipi tanımları
        # lifetime=None → kalıcı | energy_gain: pozitif=dolar, negatif=erir
        self.FOOD_DEFS = {
            # Kırmızı: Ana yemek. Enerji kazancı artırıldı (25→50), ödül biraz düşürüldü (15→12).
            # Ajan kırmızıyı temel hayatta kalma kaynağı olarak öğrensin.
            0: {"color": (255,  80,  80), "reward": 12, "energy_gain":  50, "lifetime": 150},
            # Altın: Nadir ama değerli. Ödül 50→25'e düşürüldü (dominans kırıldı).
            # Enerji kazancı 100→70 (hâlâ çok iyi ama kırmızıdan sadece 1.4x fazla).
            1: {"color": (255, 215,   0), "reward": 25, "energy_gain":  70, "lifetime":   50},
            # Zehir: Ceza -5→-20, enerji kaybı -15→-30. Artık aktif kaçınmayı zorluyor.
            2: {"color": (180,   0, 220), "reward": -20, "energy_gain": -30, "lifetime":   30},
        }
        self.MAX_FOODS = 3
        self.spawn_timer = 0
        self.spawn_interval = 20

        self.max_episode_steps = 2000 # Maksimum adım sınırı (Truncation)
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _generate_obstacles(self):
        # 16 tane nokta yerine sadece 6 tane bağımsız engel oluşturalım
        self.obstacles = []
        for _ in range(6):
            while True:
                # self.np_random.integers(low, high) -> high is exclusive
                p = (int(self.np_random.integers(2, self.size - 2)), 
                     int(self.np_random.integers(2, self.size - 2)))
                if p not in self.snake and p not in self.obstacles:
                    self.obstacles.append(p)
                    break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial snake position (middle)
        self.snake = [(self.size // 2, self.size // 2),
                      (self.size // 2, self.size // 2 + 1),
                      (self.size // 2, self.size // 2 + 2)]
        self.direction = (0, -1)  # Initially moving UP
        
        # Engelleri yılanın doğduğu yere basmayacak şekilde her resetlendiğinde baştan oluştur
        self.obstacles = []
        self._generate_obstacles()
        
        # Multi-food: önce listeyi boş başlatıyoruz (spawn_food içinde hata almamak için)
        self.foods = []
        self.foods.append(self._spawn_food(0))  # Başlangıçta 1 normal yemek
        self.spawn_timer = 0
        self.score = 0
        self.red_eaten = 0
        self.gold_eaten = 0
        self.steps = 0
        self.energy = self.max_energy  # Enerjiyi sıfırla
        self.visited_cells = deque(maxlen=self.visited_memory)  # Döngü takibi

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def _spawn_food(self, food_type):
        """Verilen tipte yeni bir yemek üretir: [konum, tip, kalan_ömür]"""
        occupied = set(self.snake) | set(self.obstacles) | {f[0] for f in self.foods}
        while True:
            pos = (int(self.np_random.integers(0, self.size)), 
                   int(self.np_random.integers(0, self.size)))
            if pos not in occupied:
                lt = self.FOOD_DEFS[food_type]["lifetime"]
                # lifetime rasgeleliği
                lifetime = int(self.np_random.integers(int(lt * 0.75), lt + 1)) if lt else None
                return [pos, food_type, lifetime]

    def _get_obs(self):
        head = self.snake[0]
        
        # 4 ana yöndeki noktalar (1 ve 2 birim uzaklık)
        def get_points(dist):
            return {
                'l': (head[0] - dist, head[1]),
                'r': (head[0] + dist, head[1]),
                'u': (head[0], head[1] - dist),
                'd': (head[0], head[1] + dist)
            }
        
        p1 = get_points(1)
        p2 = get_points(2)
        
        # Mevcut yön
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        # Tehlike tespiti (Ön, Sağ, Sol) - 1 VEYA 2 birim ilerde tehlike var mı?
        def is_danger(p_list):
            for p in p_list:
                if self._is_collision(p): return True
            return False

        if dir_u:
            d_f = is_danger([p1['u'], p2['u']])
            d_r = is_danger([p1['r'], p2['r']])
            d_l = is_danger([p1['l'], p2['l']])
        elif dir_d:
            d_f = is_danger([p1['d'], p2['d']])
            d_r = is_danger([p1['l'], p2['l']])
            d_l = is_danger([p1['r'], p2['r']])
        elif dir_l:
            d_f = is_danger([p1['l'], p2['l']])
            d_r = is_danger([p1['u'], p2['u']])
            d_l = is_danger([p1['d'], p2['d']])
        elif dir_r:
            d_f = is_danger([p1['r'], p2['r']])
            d_r = is_danger([p1['d'], p2['d']])
            d_l = is_danger([p1['u'], p2['u']])

        # Radar (Mesafe)
        def get_dist_to_danger(direction):
            dist = 0
            curr = [head[0], head[1]]
            while True:
                dist += 1
                curr[0] += direction[0]
                curr[1] += direction[1]
                if self._is_collision(tuple(curr)):
                    break
                if dist >= self.size: break
            return dist / self.size

        # Her yemek tipi için 7 değer: [var_mı, sol, sağ, yukarı, aşağı, mesafe, kalan_sure]
        food_obs = []
        for ftype in range(3):
            candidates = [f for f in self.foods if f[1] == ftype]
            if candidates:
                nearest = min(candidates,
                              key=lambda f: abs(f[0][0]-head[0]) + abs(f[0][1]-head[1]))
                fpos = nearest[0]
                dist = np.sqrt((fpos[0]-head[0])**2 + (fpos[1]-head[1])**2) / (self.size * 1.414)
                
                # Süre bilgisini normalize et (1.0 = sonsuz veya taze, 0.0 = bitmek üzere)
                max_lt = self.FOOD_DEFS[ftype]["lifetime"]
                time_left = 1.0
                if max_lt and nearest[2] is not None:
                    time_left = nearest[2] / max_lt

                food_obs += [
                    1.0,
                    float(fpos[0] < head[0]),
                    float(fpos[0] > head[0]),
                    float(fpos[1] < head[1]),
                    float(fpos[1] > head[1]),
                    float(dist),
                    float(time_left)
                ]
            else:
                food_obs += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        obs = [
            float(d_f), float(d_r), float(d_l),          # Tehlike (3)
            float(dir_l), float(dir_r), float(dir_u), float(dir_d),  # Yön (4)
            *food_obs,                                     # Yemekler (21)
            self.energy / self.max_energy,                 # Enerji (1)
            get_dist_to_danger((0, -1)),                   # Up
            get_dist_to_danger((0, 1)),                    # Down
            get_dist_to_danger((-1, 0)),                   # Left
            get_dist_to_danger((1, 0)),                    # Right
            get_dist_to_danger((-1, -1)),                  # Up-Left
            get_dist_to_danger((1, -1)),                   # Up-Right
            get_dist_to_danger((-1, 1)),                   # Down-Left
            get_dist_to_danger((1, 1)),                    # Down-Right  (radar 8)
            len(self.snake) / (self.size * self.size),     # Doluluk (1)
        ]  # Toplam: 3+4+21+1+8+1 = 38

        return np.array(obs, dtype=np.float32)

    def _is_collision(self, p):
        # Wall collision (Dış duvarlar)
        if p[0] < 0 or p[0] >= self.size or p[1] < 0 or p[1] >= self.size:
            return True
        # Body collision (Kendi kuyruğu)
        if p in self.snake:
            return True
        # Static Obstacle collision (Sabit engeller)
        if p in self.obstacles:
            return True
        return False

    def _move_obstacles(self):
        # Her engeli rastgele bir yöne kaydırmayı dene
        new_obstacles = []
        # Aktif yemek konumlarını al
        food_positions = [f[0] for f in self.foods]
        
        for obs in self.obstacles:
            # 4 ana yönden birini seç (veya yerinde kal - 5 ihtimal)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            self.np_random.shuffle(directions)
            
            moved = False
            for dx, dy in directions:
                new_p = (obs[0] + dx, obs[1] + dy)
                
                # Geçerli bir yer mi? (Sınırlar içinde mi, yılanın üstünde mi, yemeğin üstünde mi?)
                if (0 <= new_p[0] < self.size and 0 <= new_p[1] < self.size and
                    new_p not in self.snake and new_p not in food_positions and 
                    new_p not in new_obstacles):
                    new_obstacles.append(new_p)
                    moved = True
                    break
            
            if not moved:
                new_obstacles.append(obs)
        
        self.obstacles = new_obstacles

    def step(self, action):
        self.steps += 1
        self.energy -= 1 # Her adımda enerji azalır
        
        # HAREKETLİ ENGEL MANTIĞI (Müfredata Göre)
        if self.difficulty == 1:
            pass # Sabit engeller
        elif self.difficulty == 2:
            if self.steps % 6 == 0: self._move_obstacles() # Çok yavaş
        elif self.difficulty == 3:
            if self.steps % 3 == 0: self._move_obstacles() # Mevcut hız
        
        # Update direction
        # 0: Straight, 1: Right Turn, 2: Left Turn
        clock_wise = [(0, -1), (1, 0), (0, 1), (-1, 0)] # UP, RIGHT, DOWN, LEFT
        idx = clock_wise.index(self.direction)
        
        if action == 1: # Right turn
            new_idx = (idx + 1) % 4
            self.direction = clock_wise[new_idx]
        elif action == 2: # Left turn
            new_idx = (idx - 1) % 4
            self.direction = clock_wise[new_idx]
        # action 0 is straight, no change
        
        # Yılan hareketi
        head = self.snake[0]
        # --- EN MANTIKLI HEDEFİ SEÇME ALGORİTMASI (HİBRİT MANTIK) ---
        head = self.snake[0]
        good_foods = [f for f in self.foods if f[1] in (0, 1)]
        
        def get_logic_score(f):
            fpos, ftype, lifetime = f
            manhattan_dist = abs(fpos[0] - head[0]) + abs(fpos[1] - head[1])
            # 1. Ömür Kontrolü (Yumuşatıldı): Süre yetmese bile direkt pes etme, mesafe cezasını katla ama şansını dene
            if lifetime is not None and manhattan_dist > lifetime:
                return float(manhattan_dist * 2.5) 
            
            reward = self.FOOD_DEFS[ftype]["reward"]
            # Enerji Paniği: Eşik %15→%25 yükseltildi. Ajan daha erken tedbir alır (max_energy=200 → 50 adım kaldığında).
            if self.energy < self.max_energy * 0.25:
                return float(manhattan_dist)
            return float(manhattan_dist / (reward + 0.1))

        if good_foods:
            best_food = min(good_foods, key=get_logic_score)
            target_pos, target_ftype, _ = best_food
            prev_dist = abs(target_pos[0] - head[0]) + abs(target_pos[1] - head[1])
        else:
            prev_dist, target_ftype, target_pos = 0, None, None

        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        terminated = False
        reward = 0

        # Çarpışma veya açlık kontrolü
        if self._is_collision(new_head) or self.energy <= 0:
            terminated = True
            reward = self.REWARD_DEATH
            return self._get_obs(), reward, terminated, False, {
                "score": self.score,
                "red_eaten": self.red_eaten,
                "gold_eaten": self.gold_eaten
            }

        self.snake.insert(0, new_head)

        # --- YEMEK ZAMANLAYICI: Süreli yemeklerin ömrünü azalt ---
        still_alive = []
        for f in self.foods:
            if f[2] is not None:  # Süreli yemek
                f[2] -= 1
                if f[2] <= 0:
                    continue  # Süresi doldu, kaldır
            still_alive.append(f)
        self.foods = still_alive

        # --- YENİ YEMEK SPAWN ---
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval and len(self.foods) < self.MAX_FOODS:
            self.spawn_timer = 0
            # Ağırlıklı spawn: Kırmızı %50, Altın %30, Zehir %20
            
            rand_val = self.np_random.random()
            if rand_val < 0.50:
                new_ftype = 0
            elif rand_val < 0.80:
                new_ftype = 1
            else:
                new_ftype = 2
                
            try:
                self.foods.append(self._spawn_food(int(new_ftype)))
            except Exception:
                pass  # Alan doluysa geç

        # Temel yemek garantisi: Haritada hiç yenecek (Kırmızı veya Sarı) yemek yoksa bir tane oluştur!
        if not any(f[1] in (0, 1) for f in self.foods):
            try:
                # Garanti yemek her zaman KIRMIZI (tip 0): Ömrü 150 adım, ajan mutlaka yetişebilir.
                # Altın (50 adım ömür) garanti olarak çıkarsa, ajan uzaktaysa expire olur → döngü riski.
                guarantee_type = 0
                self.foods.append(self._spawn_food(guarantee_type))
            except Exception:
                pass

        # --- YEMEK YEME ---
        eaten = next((f for f in self.foods if f[0] == new_head), None)
        if eaten:
            ftype = eaten[1]
            if ftype in (0, 1): self.score += 1
            if ftype == 0: self.red_eaten += 1
            if ftype == 1: self.gold_eaten += 1
            fdef  = self.FOOD_DEFS[ftype]
            # Ödül: normal ve bonus'ta dinamik artış, zehirde sabit negatif
            reward = fdef["reward"] + (self.score // 5 if ftype in (0, 1) else 0)
            # Enerji: değere göre artar veya azalır, 0-max aralığında kalır
            self.energy = max(0, min(self.max_energy, self.energy + fdef["energy_gain"]))
            self.foods.remove(eaten)
        else:
            self.snake.pop()
            reward = self.REWARD_STEP
            # Mesafe ödülü: ASİMETRİK yapıldı.
            # Yaklaşınca ödül al, uzaklaşınca sadece adim cezası yeter (ek ceza yok).
            # Ajan engel etrafında dolanan akıllıca hareketi cezalandırmasın.
            # Kırmızı: +0.05 | Altın: +0.08
            if target_pos:
                cur_dist = abs(target_pos[0] - new_head[0]) + abs(target_pos[1] - new_head[1])
                multiplier = 0.05 if target_ftype == 0 else 0.08
                if cur_dist < prev_dist:
                    reward += multiplier  # Yaklastı: ödül
                # Uzaklaştı: ek ceza yok, REWARD_STEP zaten ceza veriyor

        # DÖNGÜ CEZASI: Eşik 3→4 yapıldı.
        # Yılan büyüdükçe doğal olarak aynı bölgelere geliyor; çok sert ceza onu kitliyor.
        self.visited_cells.append(new_head)
        if self.visited_cells.count(new_head) >= 4:
            reward -= 0.20

        # SOSYAL MESAFE CEZASI: -0.02→0.05
        # Eskisi çok küçükçüydü, yaklaşma ödülü onu eziyordu. Artık davranışı gerçekten etkiler.
        for n in [(new_head[0]+1, new_head[1]), (new_head[0]-1, new_head[1]),
                  (new_head[0], new_head[1]+1), (new_head[0], new_head[1]-1)]:
            if n in self.obstacles:
                reward -= 0.05
                break

        truncated = False
        if self.steps >= self.max_episode_steps:
            truncated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {"score": self.score, "red_eaten": self.red_eaten, "gold_eaten": self.gold_eaten}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_caption("Snake RL")
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
            )
            
        # Draw obstacles (Engeller)
        for (x, y) in self.obstacles:
            pygame.draw.rect(
                canvas,
                (100, 100, 100), # Gri renk
                pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
            )
            
        # Draw foods (tip bazlı renkler)
        for food in self.foods:
            fpos, ftype, lifetime = food
            color = self.FOOD_DEFS[ftype]["color"]
            # Süreli yemekleri titret (son 15 adımda)
            if lifetime is not None and lifetime <= 15 and (self.steps % 4 < 2):
                color = (255, 255, 255)  # Beyaza titret
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(fpos[0] * self.cell_size, fpos[1] * self.cell_size,
                            self.cell_size, self.cell_size),
            )

        # Draw Energy Bar (Enerji Çubuğu)
        bar_width = (self.energy / self.max_energy) * self.window_size
        pygame.draw.rect(
            canvas,
            (255, 255, 0), # Sarı renk
            pygame.Rect(0, 0, bar_width, 5), # En üstte 5 pixel kalınlığında
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
