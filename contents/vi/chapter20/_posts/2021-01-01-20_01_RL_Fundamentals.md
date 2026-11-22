---
layout: post
title: 20-01 Cơ sở Học Tăng cường
chapter: '20'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter20
---

# Học Tăng cường: Học từ Tương tác

![Reinforcement Learning Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/800px-Reinforcement_learning_diagram.svg.png)
*Hình ảnh: Sơ đồ học tăng cường với tác tử, môi trường, hành động và phần thưởng. Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm

Học tăng cường (*reinforcement learning*, RL) khác biệt căn bản so với học có giám sát và học không giám sát. Thay vì học từ dữ liệu đã gán nhãn hay tìm cấu trúc trong dữ liệu không nhãn, tác tử RL học bằng cách tương tác trực tiếp với môi trường: thực hiện hành động, quan sát trạng thái mới và nhận phần thưởng. Mục tiêu của tác tử là tìm ra một chính sách (*policy*) — chiến lược chọn hành động — sao cho tối đa hóa phần thưởng tích lũy theo thời gian. Quá trình học thử–sai với phần thưởng trễ này gần với cách con người và động vật rèn luyện kỹ năng: thử nghiệm, quan sát kết quả và điều chỉnh hành vi dần dần.

Học tăng cường có những thách thức đặc thù so với học có giám sát. Trong học có giám sát, mô hình được cung cấp đáp án đúng cho từng đầu vào. Trong RL, tác tử chỉ nhận được tín hiệu phần thưởng cho biết kết quả tốt hay xấu, chứ không biết hành động nào là tối ưu. Do đó, tác tử phải đối mặt với sự đánh đổi giữa khám phá và khai thác (*exploration–exploitation*): nên tiếp tục dùng các hành động tốt đã biết hay thử các hành động mới có thể mang lại phần thưởng cao hơn? Bên cạnh đó, phần thưởng thường bị trễ — hệ quả của một hành động có thể chỉ bộc lộ sau nhiều bước (chẳng hạn một nước đi sớm trong cờ vua ảnh hưởng đến kết quả cuối ván). Điều này tạo ra bài toán gán công trạng (*credit assignment*): xác định hành động nào trong chuỗi đã góp phần tạo nên phần thưởng cuối cùng.

Khung toán học của quá trình quyết định Markov (*Markov Decision Process*, MDP) mô tả chặt chẽ bài toán ra quyết định tuần tự dưới sự bất định. Trạng thái (*state*) biểu diễn tình trạng của môi trường; hành động (*action*) là lựa chọn của tác tử; hàm chuyển trạng thái (*transition*) mô tả cách trạng thái thay đổi (có thể mang tính ngẫu nhiên); và phần thưởng (*reward*) đóng vai trò tín hiệu phản hồi. Tính chất Markov — trạng thái kế tiếp chỉ phụ thuộc vào trạng thái và hành động hiện tại, không phụ thuộc toàn bộ lịch sử trước đó — giúp đơn giản hóa việc phân tích và phù hợp với nhiều bài toán thực tế khi trạng thái được thiết kế đầy đủ. Hàm giá trị (*value function*) ước lượng tổng phần thưởng tương lai kỳ vọng từ một trạng thái hoặc cặp trạng thái–hành động, còn chính sách tối ưu chọn hành động để tối đa hóa kỳ vọng đó.

Khi kết hợp với học sâu thành học tăng cường sâu (*deep reinforcement learning*), RL có thể xử lý không gian trạng thái chiều cao (như hình ảnh) và không gian hành động lớn nhờ mạng nơ-ron xấp xỉ hàm giá trị hoặc chính sách. Hướng tiếp cận này đã đạt nhiều kết quả nổi bật như AlphaGo học chơi cờ vây qua tự đấu, chơi trò chơi Atari trực tiếp từ pixel, điều khiển thao tác robot và căn chỉnh mô hình ngôn ngữ thông qua học tăng cường từ phản hồi con người (RLHF). Các kiến thức nền tảng trong chương này chuẩn bị trực tiếp cho phần học tăng cường sâu ở chương tiếp theo.

## 2. Nền tảng toán học

Một MDP được định nghĩa bởi bộ năm thành phần $$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$:

- $$\mathcal{S}$$: Không gian trạng thái (tập hợp mọi cấu hình khả dĩ của môi trường).
- $$\mathcal{A}$$: Không gian hành động (tập các lựa chọn của tác tử).
- $$\mathcal{P}$$: Hàm xác suất chuyển trạng thái $$P(s_{t+1}|s_t, a_t)$$ (động lực học môi trường).
- $$\mathcal{R}$$: Hàm phần thưởng $$R(s_t, a_t, s_{t+1})$$ (tín hiệu phản hồi).
- $$\gamma \in [0,1)$$: Hệ số chiết khấu (cân bằng giữa phần thưởng tức thời và phần thưởng tương lai).

Tác tử hành động theo chính sách $$\pi(a|s)$$ — xác suất chọn hành động $$a$$ tại trạng thái $$s$$. Mục tiêu là tìm chính sách tối ưu $$\pi^*$$ tối đa hóa tổng phần thưởng chiết khấu kỳ vọng (*expected return*):

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^\infty \gamma^t r_t\right]$$

trong đó $$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$$ là một quỹ đạo (*trajectory*).

### Hàm giá trị

Hàm giá trị trạng thái đo tổng phần thưởng chiết khấu kỳ vọng khi bắt đầu từ trạng thái $$s$$ và tuân theo chính sách $$\pi$$:

$$V^\pi(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]$$

Hàm giá trị hành động (hàm $$Q$$) đo kỳ vọng đó khi thực hiện hành động $$a$$ tại trạng thái $$s$$ rồi tiếp tục theo chính sách $$\pi$$:

$$Q^\pi(s,a) = \mathbb{E}_{\pi}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a\right]$$

Hai hàm này thỏa mãn các phương trình Bellman thể hiện mối quan hệ đệ quy giữa trạng thái hiện tại và trạng thái kế tiếp:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]$$

Các hàm giá trị tối ưu thỏa mãn phương trình tối ưu Bellman:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')]$$

Khi biết $$Q^*$$, chính sách tối ưu được xác định bằng cách chọn tham lam: $$\pi^*(s) = \arg\max_a Q^*(s,a)$$.

### Q-Learning

Q-learning học trực tiếp $$Q^*$$ mà không cần biết trước xác suất chuyển trạng thái, dựa trên học chênh lệch thời gian (*temporal difference learning*):

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

Quy tắc cập nhật sử dụng phần thưởng quan sát được $$r_t$$ cùng ước lượng giá trị tương lai $$\max_{a'} Q(s_{t+1}, a')$$ để hiệu chỉnh ước lượng hiện tại $$Q(s_t, a_t)$$. Cơ chế *bootstrap* — dùng một ước lượng để cập nhật ước lượng khác — cho phép tác tử học trực tiếp từ trải nghiệm mà không cần mô hình môi trường.

## 3. Ví dụ và trực giác

Xét bài toán huấn luyện tác tử di chuyển trên lưới 5×5 để tìm đường tới đích. Trạng thái là tọa độ ô (25 trạng thái), hành động gồm `{lên, xuống, trái, phải}`, phần thưởng là `+10` khi đến đích và `-1` ở mỗi bước khác (nhằm khuyến khích đi đường ngắn nhất).

Ban đầu, các giá trị $$Q$$ đều bằng 0 hoặc ngẫu nhiên. Tác tử xuất phát từ `(0, 0)` và di chuyển ngẫu nhiên cho đến khi tình cờ chạm đích `(4, 4)` và nhận `+10`. Q-learning cập nhật:

$$Q((4,3), \text{right}) \leftarrow Q((4,3), \text{right}) + \alpha[10 + 0 - Q((4,3), \text{right})]$$

Nhờ đó, giá trị $$Q$$ của hành động đi sang phải từ ô `(4, 3)` tăng lên. Ở các lượt sau, khi quay lại `(4, 3)`, tác tử sẽ ưu tiên đi sang phải nếu dùng chính sách $$\varepsilon$$-greedy.

Qua nhiều lượt chơi (*episode*), thông tin phần thưởng lan truyền ngược dần về các ô trước đó: $$Q((4, 2), \text{right})$$ tăng vì dẫn tới `(4, 3)` đã có giá trị cao. Cuối cùng, bảng giá trị $$Q$$ tạo thành một trường giá trị dẫn hướng về đích từ mọi vị trí xuất phát trên lưới.

### Ứng dụng: $$\varepsilon$$-greedy trong luồng gợi ý

Các hệ thống gợi ý video hoặc sản phẩm cũng đối mặt với bài toán **khám phá–khai thác**: nếu chỉ hiển thị những nội dung chắc chắn người dùng thích, hệ thống dễ nhốt người dùng trong “bong bóng sở thích” (*filter bubble*); do đó cần xen kẽ **khám phá** các chủ đề mới.

Chính sách **$$\varepsilon$$-greedy** được dùng phổ biến: với xác suất $$1-\varepsilon$$ hệ thống chọn mục tối ưu theo mô hình (khai thác), và với xác suất $$\varepsilon$$ chọn ngẫu nhiên (khám phá). Chẳng hạn với $$\varepsilon=0.1$$, khoảng 90% lựa chọn là tối ưu và 10% là thử nghiệm ngẫu nhiên.

![ε-greedy 90% / 10%](/deep-learning-self-learning/img/chapter_img/chapter20/recsys_epsilon_greedy.jpg)
*Hình: Chính sách ε-greedy — phần lớn thời gian chọn phương án tối ưu, phần nhỏ còn lại thử nghiệm ngẫu nhiên. (Minh họa từ video giải thích recommendation system)*

![Chi tiết ε-greedy trong ranking](/deep-learning-self-learning/img/chapter_img/chapter20/recsys_epsilon_greedy_detail.jpg)
*Hình: Khám phá có chủ đích giúp phát hiện chủ đề mới (ví dụ “công nghệ”) ngay cả khi mô hình đang ưu tiên chủ đề cũ. (Minh họa từ video giải thích recommendation system)*

Trong hệ thống gợi ý, “phần thưởng” có thể là thời gian xem (*watch-time*), lượt thích hoặc tổ hợp nhiều mục tiêu — vẫn nằm trong khung bài toán MDP/bandit với sự đánh đổi khám phá–khai thác quen thuộc.

## 4. Mã minh họa

```python
import numpy as np

class GridWorld:
    """Simple grid environment for RL demonstration"""
    
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.state = (0, 0)
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """Execute action, return (next_state, reward, done)"""
        actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up,down,left,right
        dx, dy = actions[action]
        
        x, y = self.state
        new_x = max(0, min(self.size-1, x + dx))
        new_y = max(0, min(self.size-1, y + dy))
        
        self.state = (new_x, new_y)
        reward = 10 if self.state == self.goal else -1
        done = self.state == self.goal
        
        return self.state, reward, done

# Q-Learning
env = GridWorld(size=5)
Q = np.zeros((5, 5, 4))  # Q(state, action)

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount
epsilon = 0.1  # Exploration

print("Training Q-Learning agent...")

for episode in range(500):
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = Q[state].argmax()
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Q-learning update
        best_next_action = Q[next_state].max()
        Q[state + (action,)] += alpha * (reward + gamma * best_next_action - Q[state + (action,)])
        
        state = next_state
        
        if done:
            break
    
    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {total_reward}")

print("Learned optimal policy!")
```

## 5. Khái niệm liên quan

Học tăng cường liên hệ chặt chẽ với lý thuyết điều khiển, vận trù học và kinh tế học thông qua bài toán ra quyết định tối ưu dưới bất định. Quy hoạch động (*dynamic programming*) giải bài toán chính sách tối ưu khi đã biết rõ mô hình động lực học môi trường, còn RL mở rộng sang trường hợp không biết trước mô hình và phải học thông qua tương tác.

RL cũng liên quan đến học có giám sát qua học bắt chước (*imitation learning*) và học tăng cường nghịch (*inverse RL*). Thay vì học từ tín hiệu phần thưởng tường minh, tác tử có thể học trực tiếp từ dữ liệu minh họa của chuyên gia (học bắt chước) hoặc suy luận ngược lại hàm phần thưởng từ hành vi chuyên gia (*inverse RL*).

## 6. Các bài báo và tài liệu nền tảng

**["Reinforcement Learning: An Introduction" (2018)](http://incompleteideas.net/book/the-book-2nd.html)**  
*Tác giả*: Richard Sutton, Andrew Barto  
Giáo trình chuẩn mực về học tăng cường, trình bày hệ thống nền tảng toán học và các thuật toán cốt lõi.

**["Playing Atari with Deep Reinforcement Learning" (2013)](https://arxiv.org/abs/1312.5602)**  
*Tác giả*: Volodymyr Mnih et al.  
DQN kết hợp Q-learning với mạng nơ-ron sâu, bộ đệm phát lại trải nghiệm (*experience replay*) và mạng đích (*target network*) để chơi trò chơi Atari trực tiếp từ pixel, mở đầu cho lĩnh vực học tăng cường sâu.

## Bẫy thường gặp và mẹo thực hành

Cân bằng giữa khám phá và khai thác là yếu tố quyết định hiệu quả huấn luyện. Khai thác thuần túy (chỉ chọn hành động tốt nhất đã biết) dễ mắc kẹt ở nghiệm cục bộ, còn khám phá thuần túy (chọn ngẫu nhiên liên tục) không tận dụng được kiến thức đã học. Các chiến lược như $$\varepsilon$$-greedy, chọn hành động theo softmax hoặc UCB giúp điều tiết hợp lý hai yếu tố này.

## Điểm chính cần nhớ

Học tăng cường huấn luyện tác tử thông qua tương tác với môi trường để tìm chính sách tối đa hóa phần thưởng tích lũy. MDP mô hình hóa bài toán ra quyết định tuần tự bằng trạng thái, hành động, xác suất chuyển trạng thái và phần thưởng. Hàm giá trị ước lượng tổng phần thưởng tương lai kỳ vọng, đóng vai trò mục tiêu học. Q-learning học hàm giá trị hành động tối ưu qua cập nhật chênh lệch thời gian mà không cần biết trước mô hình môi trường. Dù phần thưởng trễ, bài toán gán công trạng và sự đánh đổi khám phá–khai thác khiến RL khó hơn học có giám sát, phương pháp này đặc biệt hữu ích khi không có sẵn nhãn giám sát tối ưu.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến):

- [Thuật toán gợi ý (TikTok-style recommendation)](https://www.facebook.com/reel/1444915507374636)

## Video tham khảo

- **Xem trước** [RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=2pWv7GOvuf0) — Google DeepMind, tiếng Anh. Giới thiệu phần thưởng, chính sách và hàm giá trị.
- [RL Course by David Silver - Lecture 2: Markov Decision Process](https://www.youtube.com/watch?v=lfHX2hHRMVQ) — Google DeepMind, tiếng Anh. Trình bày trạng thái, hành động và hệ phương trình Bellman tương ứng với phần toán học trong bài.
- [Reinforcement Learning with Neural Networks: Essential Concepts](https://www.youtube.com/watch?v=9hbQieQh7-o) — StatQuest with Josh Starmer, tiếng Anh. Minh họa cách dùng mạng nơ-ron xấp xỉ các đối tượng RL, bước đệm cho chương tiếp theo.
