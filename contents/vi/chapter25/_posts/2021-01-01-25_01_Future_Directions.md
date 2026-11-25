---
layout: post
title: 25-01 Hướng Phát triển Tương lai của Học sâu
chapter: '25'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter25
---

Một trong những biểu đồ ấn tượng nhất của khoa học máy tính thế kỷ 21 là đường thẳng tắp trên trục log–log của **Scaling Laws**: trải dài qua hơn 6 bậc độ lớn về sức mạnh tính toán, hàm mất mát của mô hình ngôn ngữ giảm đều đặn như một định luật vật lý! Bài học này đi sâu vào toán học của **Scaling Laws**, **Few-Shot / Meta-Learning (MAML)**, **Học liên tục chống quên thảm họa** và các câu hỏi mở về hiệu quả mẫu, suy luận nhân quả và an toàn AI.

## Kiến thức cần nhớ: Khung lý thuyết các hướng nghiên cứu biên giới của Học sâu

- **Định luật mở rộng dạng hàm lũy thừa (Chinchilla):** $$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$, trong đó $$E$$ là entropy bất khả giảm của ngôn ngữ tự nhiên còn hai số hạng sau phản ánh sai số do thiếu tham số và thiếu dữ liệu.
- **Meta-Learning và Few-Shot Learning (MAML, Prototypical Networks):** Học cách khởi tạo tham số $$\theta^*$$ sao cho chỉ sau 1–5 bước hạ gradient trên vài mẫu mới, mô hình đã thích nghi ngay với tác vụ chưa từng thấy.
- **Chống quên thảm họa trong Học liên tục (Continual Learning — EWC):** Dùng ma trận thông tin Fisher $$F_i$$ để phạt nặng việc làm lệch các trọng số quan trọng đối với tác vụ cũ khi học tác vụ mới.
- **AI tin cậy, nhân quả và bền vững năng lượng:** Kết hợp suy luận nhân quả (do-calculus), tiết kiệm năng lượng phần cứng và căn chỉnh an toàn cho các hệ thống tự chủ.

### Giải thích: Tính tay một bước vòng trong (Inner Loop) của MAML và phạt Fisher trong EWC

**Phần A — Một bước thích nghi nhanh của MAML:** Giả sử siêu tham số khởi tạo là $$\theta = 1{,}0$$, tốc độ học vòng trong $$\alpha = 0{,}1$$, và trên 5 mẫu hỗ trợ của tác vụ mới $$\mathcal{T}_i$$, đạo hàm hàm loss là $$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(1{,}0) = -3{,}0$$.
- Tham số thích nghi chỉ sau 1 bước: $$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta) = 1{,}0 - 0{,}1(-3{,}0) = 1{,}3$$.
- Sau đó MAML tính loss trên tập truy vấn của $$\mathcal{T}_i$$ tại $$\theta_i' = 1{,}3$$ và lấy đạo hàm *xuyên qua bước cập nhật đó* về lại $$\theta$$ để tối ưu hóa chính khả năng học nhanh!

**Phần B — Hàm phạt EWC (Elastic Weight Consolidation):** Với hai trọng số đã học xong tác vụ A tại $$\theta^* = (2{,}0,\; 0{,}5)$$ có độ quan trọng Fisher $$F_1 = 10{,}0$$ (rất quan trọng), $$F_2 = 0{,}2$$ (ít quan trọng), hệ số $$\lambda = 1{,}0$$. Nếu tác vụ B kéo cả hai lệch đi $$0{,}5$$ đơn vị (tới $$(2{,}5,\; 1{,}0)$$):
- Mức phạt cho trọng số 1: $$\frac{1}{2} \lambda F_1 (0{,}5)^2 = 0{,}5(10{,}0)(0{,}25) = 1{,}25$$.
- Mức phạt cho trọng số 2: $$\frac{1}{2} \lambda F_2 (0{,}5)^2 = 0{,}5(0{,}2)(0{,}25) = 0{,}025$$ (nhỏ hơn **50 lần**!). EWC tự động "khóa cứng" trọng số 1 để bảo vệ ký ức tác vụ A, đồng thời để trống trọng số 2 cho tác vụ B.

# Tương lai của Học sâu: Xu hướng Mới nổi và Thách thức Còn mở

## 1. Tổng quan khái niệm

Học sâu đã đạt nhiều bước tiến lớn trong thập kỷ qua, nhưng nhiều câu hỏi nền tảng vẫn đang mở. Dù các hệ thống hiện nay hoạt động hiệu quả trên những tác vụ cụ thể khi có đủ dữ liệu và tài nguyên tính toán, giới nghiên cứu vẫn tiếp tục giải quyết các giới hạn về hiệu quả mẫu, khả năng suy luận, độ tin cậy và chi phí vận hành. Nắm được các hướng đi mới và thách thức còn tồn tại giúp người thực hành đánh giá thực tế hơn quỹ đạo phát triển của lĩnh vực.

Giả thuyết mở rộng quy mô (*scaling hypothesis*) là động lực chính của nhiều tiến bộ gần đây: tăng đồng thời kích thước mô hình, lượng dữ liệu và tài nguyên tính toán giúp cải thiện hiệu năng một cách đều đặn. Quy mô 175 tỷ tham số của GPT-3 lớn hơn 100 lần so với 1.5 tỷ của GPT-2, kéo theo các năng lực mới như học từ vài ví dụ (*few-shot learning*). Tuy nhiên, hướng mở rộng này đặt ra nhiều câu hỏi quan trọng về giới hạn nguồn dữ liệu, chi phí năng lượng cũng như hạ tầng huấn luyện và phục vụ.

Học đa phương thức (*multimodal learning*) — xử lý đồng thời nhiều dạng dữ liệu như văn bản, hình ảnh và âm thanh — là một hướng trọng tâm vì tri giác con người vốn đa phương thức. Các mô hình như CLIP (học từ cặp ảnh–văn bản) và GPT-4 (xử lý cả văn bản và ảnh) cho thấy tiền huấn luyện đa phương thức hỗ trợ hiệu quả nhiều tác vụ liên phương thức như mô tả ảnh, hỏi đáp thị giác và sinh ảnh từ văn bản.

Hiệu quả và tính bền vững trở thành ưu tiên lớn khi quy mô mô hình tăng mạnh. Chi phí tính toán và tiêu thụ năng lượng để huấn luyện các mô hình lớn như GPT-3 rất cao, thúc đẩy nghiên cứu về kiến trúc gọn nhẹ, thuật toán tiết kiệm dữ liệu, chưng cất tri thức và phần cứng chuyên dụng để mở rộng khả năng tiếp cận ra ngoài các tổ chức có nguồn lực tính toán khổng lồ.

An toàn, độ vững chắc và căn chỉnh (*alignment*) là những thách thức đặc biệt quan trọng. Các mô hình hiện tại vẫn có thể thất bại khó lường trước dữ liệu ngoài phân bố, khuếch đại thiên kiến xã hội từ dữ liệu huấn luyện hoặc tối ưu mục tiêu chưa khớp hoàn toàn với ý định của con người. Khi AI được triển khai rộng rãi hơn trong các lĩnh vực trọng yếu, nghiên cứu về độ vững chắc đối kháng, công bằng, khả năng diễn giải và căn chỉnh giá trị trở thành yêu cầu bắt buộc.

## 2. Nền tảng toán học

### Định luật mở rộng (Scaling Laws)

Các quan sát thực nghiệm cho thấy mối quan hệ dạng hàm lũy thừa giữa hàm mất mát và quy mô:

$$L \propto N^{-\alpha}$$

trong đó $$L$$ là mất mát, $$N$$ là số tham số (hoặc lượng dữ liệu, hoặc ngân sách tính toán) và $$\alpha$$ là hằng số (thường trong khoảng 0.05–0.1). Quan hệ này thể hiện lợi tức giảm dần: để giảm mất mát theo cấp số cộng, cần tăng quy mô theo cấp số nhân. Dù vậy, tính ổn định của quy luật trên nhiều kiến trúc và miền dữ liệu cho thấy quy luật nhất quán trong quá trình học của mạng nơ-ron.

Đường biên tối ưu theo tính toán (*compute-optimal frontier*) cân bằng giữa kích thước mô hình và lượng dữ liệu huấn luyện:

$$N_{\text{optimal}} \propto C^{0.5}, \quad D_{\text{optimal}} \propto C^{0.5}$$

trong đó $$C$$ là ngân sách tính toán, $$N$$ là số tham số và $$D$$ là số token huấn luyện. Kết quả này chỉ ra rằng khi tăng ngân sách tính toán, cần tăng cân đối cả kích thước mô hình lẫn lượng dữ liệu.

### Học từ vài ví dụ (Few-Shot Learning)

Học meta (*meta-learning*) hình thức hóa bài toán học từ rất ít ví dụ. Với một phân bố tác vụ $$p(\mathcal{T})$$, thuật toán học tham số khởi tạo $$\theta_0$$ sao cho có thể thích nghi nhanh với tác vụ mới:

$$\theta_0 = \arg\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}\left[\mathcal{L}_\mathcal{T}(\theta - \alpha \nabla_\theta \mathcal{L}_\mathcal{T}(\theta))\right]$$

MAML tối ưu $$\theta_0$$ để chỉ cần một vài bước hạ gradient trên tập dữ liệu nhỏ của tác vụ mới đã đạt hiệu năng tốt.

### Học liên tục (Continual Learning)

Học thêm tác vụ mới mà không quên tác vụ cũ đòi hỏi cân bằng giữa tính dẻo (*plasticity* — tiếp thu kiến thức mới) và tính ổn định (*stability* — giữ kiến thức cũ). Phương pháp *Elastic Weight Consolidation* (EWC) phạt sự thay đổi của các tham số quan trọng đối với tác vụ trước đó:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}} + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^*)^2$$

trong đó $$F_i$$ là đường chéo ma trận thông tin Fisher (phản ánh độ nhạy của hàm mất mát tác vụ cũ theo tham số $$i$$) và $$\theta_i^*$$ là giá trị tham số sau khi học tác vụ cũ.

## 3. Ví dụ và trực giác

Xét bước chuyển qua các thế hệ GPT-2 (1.5 tỷ tham số), GPT-3 (175 tỷ tham số) và GPT-4 (quy mô không công bố chính thức, thường được ước đoán khoảng 1.7 nghìn tỷ tham số):

- **GPT-2**: sinh văn bản mạch lạc, hoàn thành đoạn văn cơ bản.
- **GPT-3**: học *few-shot*, suy luận đơn giản và hỗ trợ lập trình cơ bản.
- **GPT-4**: suy luận phức tạp hơn, xử lý đa phương thức và lập trình nâng cao.

Sự thay đổi không chỉ dừng ở cải thiện định lượng mà còn xuất hiện các năng lực mới theo ngưỡng quy mô (*emergent abilities*), chẳng hạn khả năng làm theo chỉ dẫn mới mà không cần huấn luyện lại riêng cho từng tác vụ.

Tương tự, các mô hình đa phương thức như DALL-E có thể sinh ảnh từ mô tả văn bản như “một phi hành gia cưỡi ngựa theo phong cách chân thực” nhờ kết hợp hiểu ngôn ngữ tự nhiên với biểu diễn đặc trưng thị giác.

### Hệ thống gợi ý: Tối ưu đa mục tiêu và trách nhiệm

Các nền tảng video ngắn cho thấy bài toán “căn chỉnh” không chỉ giới hạn ở chatbot: mỗi lần làm mới bảng tin là một bài toán **tối ưu đa mục tiêu** (*multi-objective optimization*) phải cân bằng đồng thời nhiều chỉ số như thời gian xem, mức độ tương tác, đa dạng chủ đề, an toàn nội dung và doanh thu quảng cáo — không tồn tại một hàm mất mát đơn lẻ nào phản ánh đủ mọi tiêu chí.

![Cân bằng đa mục tiêu](/deep-learning-self-learning/img/chapter_img/chapter25/recsys_multi_objective.jpg)
*Hình: Hệ thống xếp hạng bảng tin cần cân bằng đồng thời nhiều mục tiêu thay vì chỉ tối ưu một chỉ số duy nhất. (Minh họa từ video giải thích recommendation system)*

Các thách thức thực tế đi kèm gồm:
- **Bong bóng lọc / buồng vang (*filter bubble / echo chamber*)**: tối ưu hóa thuần túy theo mức độ tương tác dễ thu hẹp phạm vi nội dung người dùng tiếp cận.
- **Vòng lặp phản hồi (*feedback loop*)**: mô hình khuếch đại hành vi cũ, khiến dữ liệu thu thập về sau ngày càng thiên lệch.
- **Xung đột giá trị**: mục tiêu kéo dài thời gian sử dụng có thể mâu thuẫn với sự đa dạng thông tin và trải nghiệm lành mạnh của người dùng.

Do đó, các hướng nghiên cứu như công bằng (*fairness*), ràng buộc đa dạng, hệ gợi ý nhân quả (*causal recsys*) và con người trong vòng lặp (*human-in-the-loop*) nằm tại giao điểm giữa mở rộng quy mô, căn chỉnh và kỹ thuật hệ thống học máy thực tế.

## 4. Mã minh họa

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Few-shot learning example
class FewShotLearner(nn.Module):
    """
    Meta-learning for few-shot classification.
    
    Learns from episodes: each episode has support set (few examples)
    and query set (test examples). Model learns to adapt quickly
    to new classes from few examples.
    """
    
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        
        # Feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Classifier (dynamically created for each episode)
        self.classifier = nn.Linear(hidden_dim, 1)  # Placeholder
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
    
    def adapt(self, support_x, support_y, num_steps=5, lr=0.01):
        """
        Adapt to new task from support set.
        
        Creates and trains task-specific classifier on few examples.
        """
        # Extract features from support set
        with torch.no_grad():
            support_features = self.encoder(support_x)
        
        # Create and train task-specific classifier
        num_classes = support_y.max().item() + 1
        task_classifier = nn.Linear(support_features.size(1), num_classes)
        optimizer = torch.optim.SGD(task_classifier.parameters(), lr=lr)
        
        # Quick adaptation
        for _ in range(num_steps):
            logits = task_classifier(support_features)
            loss = F.cross_entropy(logits, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return task_classifier

print("="*70)
print("Few-Shot Learning: Rapid Adaptation to New Tasks")
print("="*70)
print("\nMeta-learning enables learning from few examples (5-10 per class)")
print("by learning how to learn quickly across many related tasks.")
print("This is crucial for data-scarce domains!")
```

## 5. Khái niệm liên quan

Tính toán mô phỏng thần kinh (*neuromorphic computing*) và mạng nơ-ron xung (*spiking neural networks*) xử lý các xung sự kiện rời rạc thay vì giá trị liên tục, hướng tới mức tiêu thụ năng lượng thấp hơn và gần với cơ chế sinh học hơn. Song song đó, đồng thiết kế giữa thuật toán và phần cứng chuyên dụng (TPU, chip neuromorphic) tiếp tục là hướng tối ưu hóa quan trọng.

Học máy lượng tử (*quantum machine learning*) nghiên cứu khả năng tận dụng máy tính lượng tử để tăng tốc một số phép toán tuyến tính và tối ưu hóa; hiện chủ yếu ở giai đoạn lý thuyết và thử nghiệm quy mô nhỏ.

Tìm kiếm kiến trúc nơ-ron (NAS) tự động hóa quá trình thiết kế mạng, góp phần tạo ra các họ kiến trúc hiệu quả như EfficientNet.

## 6. Các bài báo nền tảng

**["Attention is All You Need" (2017)](https://arxiv.org/abs/1706.03762)**  
Kiến trúc Transformer trở thành nền tảng cho BERT, GPT và hầu hết mô hình ngôn ngữ lớn hiện đại.

**["CLIP: Learning Transferable Visual Models From Natural Language Supervision" (2021)](https://arxiv.org/abs/2103.00020)**  
*Tác giả*: Alec Radford, Jong Wook Kim, et al. (OpenAI)  
CLIP học biểu diễn liên phương thức ảnh–văn bản từ 400 triệu cặp dữ liệu, cho phép phân loại ảnh *zero-shot* thông qua câu lệnh văn bản.

**["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)](https://arxiv.org/abs/2010.11929)**  
Vision Transformer (ViT) chứng minh kiến trúc dựa trên *attention* có thể đạt hiệu năng ngang bằng hoặc vượt CNN trong thị giác máy tính ở quy mô lớn.

**["Training Compute-Optimal Large Language Models" (2022)](https://arxiv.org/abs/2203.15556)**  
*Tác giả*: Jordan Hoffmann, Sebastian Borgeaud, et al. (DeepMind)  
Bài báo Chinchilla chỉ ra rằng nhiều LLM trước đó bị thiếu dữ liệu huấn luyện so với số tham số, đồng thời đề xuất tỷ lệ mở rộng cân bằng giữa kích thước mô hình và lượng dữ liệu.

**["Sparks of Artificial General Intelligence: Early experiments with GPT-4" (2023)](https://arxiv.org/abs/2303.12712)**  
*Tác giả*: Microsoft Research  
Khảo sát năng lực của GPT-4 trên nhiều nhóm bài toán đa dạng, ghi nhận sự mở rộng rõ rệt về khả năng tổng quát hóa cũng như những hạn chế còn tồn tại.

## Bẫy thường gặp và mẹo thực hành

Một sai lệch nhận thức phổ biến là đánh giá quá cao tiến độ ngắn hạn nhưng lại đánh giá thấp tác động dài hạn. Các hệ thống hiện nay vẫn còn những hạn chế rõ rệt về lẽ thường, suy luận vững chắc và hiệu quả dữ liệu — những vấn đề khó giải quyết triệt để trong thời gian ngắn; trong khi đó, tiến bộ tích lũy qua giai đoạn 10–20 năm có thể tạo ra thay đổi lớn hơn nhiều so với dự đoán tuyến tính.

## Điểm chính cần nhớ

Các hướng phát triển chính của học sâu bao gồm mở rộng quy mô hợp lý theo định luật *scaling*, xây dựng hệ thống đa phương thức tích hợp văn bản–thị giác–âm thanh, nâng cao hiệu quả tính toán và năng lượng, cải thiện khả năng học từ ít dữ liệu (*few-shot / meta-learning*), học liên tục không quên, và tăng cường độ vững chắc, công bằng cùng căn chỉnh giá trị. Những bài toán mở cốt lõi vẫn xoay quanh hiệu quả mẫu, suy luận đáng tin cậy, khả năng diễn giải và độ bền trước dịch chuyển phân bố dữ liệu.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến):

- [Thuật toán gợi ý (TikTok-style recommendation)](https://www.facebook.com/reel/1444915507374636)

## Video tham khảo

- **Xem trước** [Chinchilla Explained: Compute-Optimal Massive Language Models](https://www.youtube.com/watch?v=PZXN7jm9IC0) — Edan Meyer, tiếng Anh. Giải thích định luật mở rộng tối ưu theo tính toán tương ứng với phần hàm lũy thừa trong bài.
- [MIT 6.S191 (2024): Language Models and New Frontiers](https://www.youtube.com/watch?v=N1fbskTpwZ0) — Alexander Amini, tiếng Anh. Trình bày mô hình đa phương thức và các bài toán nghiên cứu mở.
- [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) — 3Blue1Brown, tiếng Anh. Minh họa trực quan ngắn gọn về các mô hình ngôn ngữ lớn được nhắc tới trong phần *scaling*.
