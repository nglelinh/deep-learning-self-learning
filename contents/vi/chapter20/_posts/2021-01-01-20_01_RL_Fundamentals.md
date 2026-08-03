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
Học tăng cường (*reinforcement learning*, RL) đại diện cho mô hình học khác biệt cơ bản so với học có giám sát và không giám sát mà ta đã nghiên cứu. Thay vì học từ ví dụ có nhãn hay khám phá mẫu trong dữ liệu không nhãn, các tác tử RL học bằng cách tương tác với môi trường, thực hiện hành động, quan sát kết quả và nhận phần thưởng. Mục tiêu của tác tử là khám phá một chính sách (*policy*) — chiến lược chọn hành động — tối đa hóa phần thưởng tích lũy theo thời gian. Việc học thử–sai với phần thưởng trễ phản ánh cách con người và động vật học nhiều kỹ năng: ta thử hành động, trải nghiệm hậu quả, và dần cải thiện hành vi để đạt kết quả mong muốn.

Hiểu RL đòi hỏi đánh giá những gì làm nó thách thức hơn học có giám sát. Trong học có giám sát, ta có câu trả lời đúng cho mọi đầu vào — mô hình học từ ví dụ hành vi tối ưu. Trong RL, ta chỉ có phần thưởng cho biết kết quả tốt đến mức nào, chứ không biết hành động cụ thể nào là tối ưu. Tác tử phải khám phá các hành động khác nhau để tìm hành động dẫn đến phần thưởng cao, tạo ra đánh đổi khám phá–khai thác (*exploration–exploitation*): nên khai thác các hành động tốt đã biết hay khám phá các phương án có thể tốt hơn? Hơn nữa, phần thưởng thường bị trễ — hậu quả của một hành động có thể chưa rõ cho đến nhiều bước sau (trong cờ vua, nước đi đầu ván ảnh hưởng thắng/thua rất muộn). Bài toán gán công trạng (*credit assignment*) trở nên khó: trong nhiều hành động đã thực hiện, hành động nào đóng góp vào phần thưởng cuối?

Khung toán học của quá trình quyết định Markov (*Markov Decision Process*, MDP) hình thức hóa thanh lịch việc ra quyết định tuần tự dưới bất định. Trạng thái (*state*) biểu diễn cấu hình môi trường; hành động (*action*) là lựa chọn của tác tử; chuyển trạng thái (*transition*) mô tả cách hành động thay đổi trạng thái (có thể ngẫu nhiên); phần thưởng (*reward*) cung cấp tín hiệu học. Tính chất Markov — tương lai chỉ phụ thuộc trạng thái hiện tại, không phụ thuộc toàn bộ lịch sử — đơn giản hóa phân tích và xấp xỉ đúng cho nhiều bài toán thực tế khi trạng thái được chọn phù hợp. Hàm giá trị (*value function*) định lượng phần thưởng tương lai kỳ vọng từ trạng thái hoặc cặp trạng thái–hành động, cung cấp mục tiêu học. Chính sách ánh xạ trạng thái sang hành động; chính sách tối ưu chọn hành động tối đa hóa phần thưởng tích lũy kỳ vọng.

Mối liên hệ với học sâu qua học tăng cường sâu (*deep reinforcement learning*) cho phép RL mở rộng sang không gian trạng thái chiều cao như ảnh và không gian hành động lớn. Mạng neuron xấp xỉ hàm giá trị hoặc chính sách, học từ trải nghiệm qua gradient descent. Sự kết hợp này đã đạt thành công đáng kể: AlphaGo làm chủ cờ vây qua tự chơi, chơi game Atari từ pixel, học thao tác robot qua thử–sai, và căn chỉnh mô hình ngôn ngữ tinh vi qua học tăng cường từ phản hồi con người (RLHF). Hiểu cơ sở RL đặt nền cho các phương pháp deep RL được trình bày ở chương tiếp theo.

## 2. Nền tảng toán học
Một MDP được định nghĩa bởi bộ $$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$:

- $$\mathcal{S}$$: Không gian trạng thái (mọi cấu hình môi trường khả dĩ)
- $$\mathcal{A}$$: Không gian hành động (lựa chọn của tác tử)
- $$\mathcal{P}$$: Hàm chuyển $$P(s_{t+1}|s_t, a_t)$$ (động lực học)
- $$\mathcal{R}$$: Hàm phần thưởng $$R(s_t, a_t, s_{t+1})$$ (phản hồi)
- $$\gamma \in [0,1)$$: Hệ số chiết khấu (cân bằng phần thưởng tức thời so với tương lai)

Tác tử theo chính sách $$\pi(a|s)$$ — xác suất hành động $$a$$ ở trạng thái $$s$$. Mục tiêu là tìm chính sách tối ưu $$\pi^*$$ tối đa hóa lợi nhuận kỳ vọng:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^\infty \gamma^t r_t\right]$$

trong đó $$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$$ là một quỹ đạo (*trajectory*).

### Hàm Giá trị

Hàm giá trị trạng thái định lượng lợi nhuận kỳ vọng bắt đầu từ trạng thái $$s$$:

$$V^\pi(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]$$

Hàm giá trị hành động (hàm Q) bao gồm hành động đầu tiên:

$$Q^\pi(s,a) = \mathbb{E}_{\pi}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s, a_0 = a\right]$$

Các hàm này thỏa phương trình Bellman thể hiện quan hệ đệ quy:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]$$

Hàm giá trị tối ưu thỏa phương trình tối ưu Bellman:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')]$$

Chính sách tối ưu tham lam theo $$Q^*$$: $$\pi^*(s) = \arg\max_a Q^*(s,a)$$.

### Q-Learning

Q-learning học $$Q^*$$ mà không cần biết xác suất chuyển trạng thái, thông qua học chênh lệch thời gian (*temporal difference learning*):

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

Cập nhật dùng phần thưởng quan sát $$r_t$$ và ước lượng giá trị tương lai $$\max_{a'} Q(s_{t+1}, a')$$ để cải thiện ước lượng hiện tại $$Q(s_t, a_t)$$. Việc bootstrap — dùng một ước lượng để cải thiện ước lượng khác — cho phép học từ trải nghiệm mà không cần mô hình môi trường.

## 3. Ví dụ / Trực giác

Hãy xét huấn luyện tác tử chơi game đơn giản: điều hướng lưới 5×5 để đến đích. Trạng thái là vị trí (25 trạng thái), hành động là {lên, xuống, trái, phải}, phần thưởng +10 tại đích, −1 ở nơi khác (khuyến khích đến đích nhanh).

Ban đầu, giá trị Q ngẫu nhiên. Tác tử bắt đầu tại (0,0), thực hiện hành động ngẫu nhiên. Sau khi lang thang, nó tình cờ đến đích (4,4), nhận +10. Q-learning cập nhật:

$$Q((4,3), \text{right}) \leftarrow Q((4,3), \text{right}) + \alpha[10 + 0 - Q((4,3), \text{right})]$$

Giá trị Q cho “đi phải từ vị trí cạnh đích” tăng. Lần sau khi tác tử đến (4,3), nó có xu hướng đi phải hơn (nếu dùng chính sách ε-greedy dựa trên Q).

Sau nhiều episode, giá trị lan truyền ngược. Q((4,2), right) tăng vì dẫn đến (4,3) vốn đã có Q cao. Cuối cùng, các giá trị Q tối ưu tạo gradient hướng về đích từ mọi trạng thái, và tác tử học điều hướng thẳng đến đích từ bất kỳ vị trí xuất phát nào.

### Ứng dụng: ε-greedy trong feed gợi ý

Feed video/sản phẩm cũng đối mặt **khám phá–khai thác**: luôn chỉ hiện nội dung “chắc chắn được thích” sẽ khóa người dùng trong bong bóng sở thích (*filter bubble*); thỉnh thoảng cần **khám phá** chủ đề mới.

Chính sách **ε-greedy** rất phổ biến: với xác suất $$1-\varepsilon$$ chọn hành động/item tối ưu theo mô hình (khai thác), với xác suất $$\varepsilon$$ chọn ngẫu nhiên (khám phá). Ví dụ $$\varepsilon=0.1$$ → ~90% tối ưu, ~10% ngẫu nhiên.

![ε-greedy 90% / 10%](/deep-learning-self-learning/img/chapter_img/chapter20/recsys_epsilon_greedy.jpg)
*Hình: ε-greedy — phần lớn thời gian theo lựa chọn tối ưu, một phần nhỏ thử ngẫu nhiên. (Minh họa từ video giải thích recommendation system)*

![Chi tiết ε-greedy trong ranking](/deep-learning-self-learning/img/chapter_img/chapter20/recsys_epsilon_greedy_detail.jpg)
*Hình: Khám phá có chủ đích giúp phát hiện chủ đề mới (ví dụ “công nghệ”) dù mô hình đang tin chắc chủ đề cũ. (Minh họa từ video giải thích recommendation system)*

Trong recsys, “phần thưởng” có thể là watch-time, like, hoặc tổ hợp đa mục tiêu — vẫn là MDP/bandit với đánh đổi exploration–exploitation quen thuộc.

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
RL gắn với lý thuyết điều khiển, nghiên cứu vận hành và kinh tế qua ra quyết định tối ưu dưới bất định. Quy hoạch động (*dynamic programming*) cung cấp thuật toán tính chính sách tối ưu khi động lực học môi trường đã biết. RL mở rộng sang động lực học chưa biết, học qua tương tác.

RL liên quan đến học có giám sát qua học bắt chước (*imitation learning*) và RL nghịch (*inverse RL*). Thay vì học từ phần thưởng, tác tử có thể học từ minh họa (có giám sát), hoặc suy ra hàm phần thưởng từ hành vi chuyên gia (inverse RL).

## 6. Các Bài báo Nền tảng

**["Reinforcement Learning: An Introduction" (2018)](http://incompleteideas.net/book/the-book-2nd.html)**  
*Tác giả*: Richard Sutton, Andrew Barto  
Giáo trình RL chuẩn mực, thiết lập nền tảng toán học và các thuật toán cốt lõi. Tài liệu thiết yếu cho bất kỳ ai học RL.

**["Playing Atari with Deep Reinforcement Learning" (2013)](https://arxiv.org/abs/1312.5602)**  
*Tác giả*: Volodymyr Mnih et al.  
DQN cho thấy học sâu + RL có thể học chơi game Atari từ pixel, khởi đầu cuộc cách mạng deep RL. Kết hợp Q-learning với mạng neuron sâu, experience replay và target network.

## Bẫy Thường gặp và Mẹo

Đánh đổi khám phá–khai thác là then chốt. Khai thác thuần (luôn chọn hành động tốt đã biết) không bao giờ khám phá phương án tốt hơn. Khám phá thuần (hành động ngẫu nhiên) không dùng tri thức đã học. ε-greedy, chính sách softmax hoặc phương pháp dựa trên UCB cân bằng cả hai.

## Điểm Chính Cần Nhớ

Học tăng cường huấn luyện tác tử qua tương tác với môi trường, học chính sách tối đa hóa phần thưởng tích lũy qua thử–sai. MDP hình thức hóa ra quyết định tuần tự với trạng thái, hành động, chuyển trạng thái và phần thưởng. Hàm giá trị ước lượng lợi nhuận tương lai kỳ vọng, cung cấp mục tiêu học. Q-learning học giá trị hành động tối ưu qua cập nhật chênh lệch thời gian, cho phép học không cần mô hình môi trường. Đánh đổi khám phá–khai thác đòi hỏi cân bằng giữa khám phá chiến lược mới và dùng chiến lược tốt đã biết. Phần thưởng trễ và gán công trạng khiến RL khó hơn học có giám sát nhưng mở ra ứng dụng nơi giám sát không có sẵn hoặc quá đắt.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Thuật toán gợi ý (TikTok-style recommendation)](https://www.facebook.com/reel/1444915507374636)
