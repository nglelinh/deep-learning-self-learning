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

# Tương lai của Học sâu: Xu hướng Mới nổi và Thách thức Còn mở

## 1. Tổng quan khái niệm
Học sâu đã đạt thành công đáng kể trong thập kỷ qua, song ta đang đứng ở điểm khởi đầu chứ không phải kết thúc của tiềm năng tác động. Dù hệ thống hiện tại vượt trội ở các tác vụ cụ thể với dữ liệu và tính toán dồi dào, nhiều thách thức cơ bản vẫn chưa giải và các hướng hứa hẹn đang nổi lên. Hiểu các biên giới này — cả thách thức kỹ thuật lẫn cách tiếp cận hứa hẹn — chuẩn bị cho người thực hành và nhà nghiên cứu đóng góp vào sự tiến hóa tiếp tục của lĩnh vực và giúp dự đoán cách năng lực AI sẽ mở rộng trong những năm tới.

Giả thuyết mở rộng quy mô (*scaling hypothesis*) đã thúc đẩy nhiều tiến bộ gần đây: mô hình lớn hơn, huấn luyện trên nhiều dữ liệu hơn với nhiều tính toán hơn, liên tục cải thiện hiệu năng, chưa quan sát thấy trần rõ ràng. 175 tỷ tham số của GPT-3 vượt 1.5 tỷ của GPT-2 gấp 100×, sinh năng lực định tính mới như học few-shot. Song mở rộng đặt ra câu hỏi then chốt: Xu hướng này có bền vững với chi phí năng lượng và sẵn có dữ liệu không? Ta có chạm giới hạn cơ bản hay khám phá năng lực nổi? Làm sao huấn luyện và phục vụ mô hình lớn hơn nhiều bậc? Hiểu lời hứa và hạn chế của scaling định hình kỳ vọng về quỹ đạo AI.

Học đa phương thức (*multimodal learning*) — hệ thống xử lý nhiều phương thức (thị giác, ngôn ngữ, âm thanh) cùng lúc — đại diện cho biên giới then chốt vì trí thông minh con người vốn đa phương thức. Các mô hình như CLIP (học từ cặp ảnh–văn bản) và GPT-4 (xử lý cả văn bản lẫn ảnh) chứng minh rằng tiền huấn luyện đa phương thức cho phép hiểu liên phương thức phong phú: mô tả ảnh, trả lời câu hỏi thị giác, sinh ảnh từ mô tả văn bản. Hệ thống tương lai nhiều khả năng vốn đa phương thức, học biểu diễn thống nhất trải các phương thức giống như con người tích hợp liền mạch thị giác, âm thanh và ngôn ngữ.

Hiệu quả và tính bền vững nổi lên như mối quan tâm then chốt khi mô hình lớn lên. Huấn luyện GPT-3 được báo cáo tiêu tốn hàng trăm nghìn đô la tính toán và lượng phát thải carbon đáng kể. Dân chủ hóa AI đòi hỏi kỹ thuật làm năng lực tiên tiến tiếp cận được ngoài các gã khổng lồ công nghệ: kiến trúc hiệu quả, thuật toán tốt hơn đòi hỏi ít dữ liệu hoặc tính toán hơn, chưng cất tri thức từ mô hình lớn sang nhỏ, và phần cứng chuyên biệt. Hiểu biên hiệu quả — cách tối đa hóa năng lực trên mỗi đơn vị tính toán — sẽ ngày càng quan trọng khi triển khai AI mở rộng.

An toàn, độ vững chắc và căn chỉnh (*alignment*) đại diện cho có lẽ những thách thức quan trọng nhất. Hệ thống hiện tại có thể mong manh (thất bại khó dự đoán trên đầu vào ngoài phân bố), thiên kiến (phản ánh và khuếch đại thiên kiến xã hội trong dữ liệu huấn luyện), và lệch căn (*misaligned* — tối ưu mục tiêu không khớp giá trị thật của con người). Khi hệ thống AI trở nên năng lực hơn và được triển khai trong ứng dụng then chốt, đảm bảo chúng hành xử đáng tin cậy và có lợi trở thành ưu tiên hàng đầu. Nghiên cứu về độ vững chắc đối kháng, công bằng, diễn giải và căn chỉnh giá trị sẽ then chốt cho phát triển AI có trách nhiệm.

## 2. Nền tảng toán học
### Định luật Mở rộng (Scaling Laws)

Quan sát thực nghiệm gợi ý quan hệ lũy thừa giữa hiệu năng mô hình và quy mô:

$$L \propto N^{-\alpha}$$

trong đó $$L$$ là mất mát, $$N$$ là số tham số (hoặc kích thước dữ liệu, hoặc tính toán), và $$\alpha$$ là hằng số (thường 0.05–0.1). Điều này hàm ý lợi tức giảm dần: nhân đôi tham số có thể giảm mất mát 5–10%, đòi hỏi tham số tăng theo hàm mũ cho cải thiện mất mát tuyến tính. Song quan hệ nhất quán đáng ngạc nhiên qua kiến trúc và miền, gợi ý quy luật cơ bản trong cách mạng neuron học.

Biên tối ưu tính toán đánh đổi kích thước mô hình và dữ liệu huấn luyện:

$$N_{\text{optimal}} \propto C^{0.5}, \quad D_{\text{optimal}} \propto C^{0.5}$$

trong đó $$C$$ là ngân sách tính toán, $$N$$ là tham số, $$D$$ là token huấn luyện. Điều này gợi ý việc dùng tính toán tối ưu cân bằng đều kích thước mô hình và lượng dữ liệu, thông tin cho cách phân bổ tài nguyên khi mở rộng.

### Học Few-Shot

Meta-learning hình thức hóa học từ ít ví dụ. Cho phân bố tác vụ $$p(\mathcal{T})$$, học khởi tạo $$\theta_0$$ thích ứng nhanh:

$$\theta_0 = \arg\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}\left[\mathcal{L}_\mathcal{T}(\theta - \alpha \nabla_\theta \mathcal{L}_\mathcal{T}(\theta))\right]$$

MAML học $$\theta_0$$ sao cho một bước gradient trên tác vụ mới cho hiệu năng tốt. Điều này cho phép thích ứng nhanh với dữ liệu tối thiểu.

### Học Liên tục (Continual Learning)

Học tác vụ mới mà không quên tác vụ cũ đòi hỏi cân bằng tính dẻo (học mới) và ổn định (giữ cũ). Elastic Weight Consolidation phạt thay đổi tham số quan trọng cho tác vụ trước:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}} + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^*)^2$$

trong đó $$F_i$$ là thông tin Fisher (đạo hàm bậc hai của mất tác vụ cũ), $$\theta_i^*$$ là tham số sau khi học tác vụ cũ. Điều này cho phép học tác vụ mới trong khi bảo vệ tham số then chốt cho tác vụ cũ.

## 3. Ví dụ / Trực giác

Xét lộ trình từ GPT-2 (1.5B tham số) đến GPT-3 (175B tham số) đến GPT-4 (đồn khoảng 1.7T tham số). Mỗi bước mở rộng mở ra năng lực mới:

**GPT-2**: Sinh văn bản mạch lạc, hoàn thành cơ bản  
**GPT-3**: Học few-shot, suy luận đơn giản, lập trình cơ bản  
**GPT-4**: Suy luận phức tạp, hiểu đa phương thức, lập trình tinh vi  

Đây không chỉ là cải thiện định lượng mà thay đổi năng lực định tính. GPT-3 có thể tuân theo chỉ dẫn mà nó không được huấn luyện tường minh. GPT-4 có thể suy luận về ảnh. Các khả năng nổi (*emergent abilities*) — năng lực xuất hiện đột ngột ở quy mô nhất định — gợi ý scaling có thể tiếp tục mang lại bất ngờ.

Học đa phương thức mở ra ứng dụng mới. DALL-E sinh ảnh từ văn bản: “một phi hành gia cưỡi ngựa theo phong cách chân thực”. Mô hình phải hiểu cả ngôn ngữ (phân tích mô tả) lẫn thị giác (phi hành gia và ngựa trông như thế nào, chân thực nghĩa là gì) và ánh xạ giữa chúng (khái niệm ngôn ngữ chuyển thành đặc trưng hình ảnh ra sao). Hệ thống tương lai có thể xử lý liền mạch video, âm thanh và văn bản cùng lúc, gần hơn nhiều với tri giác kiểu con người.

### Hệ thống gợi ý: tối ưu đa mục tiêu & trách nhiệm

Nền tảng video ngắn cho thấy AI “căn chỉnh” không chỉ là chatbot: mỗi lần vuốt feed là một quyết định tối ưu hóa đồng thời nhiều chỉ số (thời gian xem, tương tác, đa dạng chủ đề, an toàn nội dung, doanh thu quảng cáo…). Đó là **multi-objective optimization** — không có một loss duy nhất.

![Cân bằng đa mục tiêu](/deep-learning-self-learning/img/chapter_img/chapter25/recsys_multi_objective.jpg)
*Hình: Xếp hạng feed phải cân bằng nhiều mục tiêu cùng lúc, không chỉ một metric. (Minh họa từ video giải thích recommendation system)*

Thách thức liên quan:

- **Filter bubble / echo chamber** — tối ưu engagement thuần túy thu hẹp thế giới quan người dùng  
- **Feedback loop** — mô hình khuếch đại hành vi cũ, làm dữ liệu ngày càng lệch  
- **Căn chỉnh giá trị** — “giữ chân lâu” có thể xung đột với sức khỏe số / đa dạng thông tin  

Các hướng tương lai (fairness, diversity constraints, causal recsys, human-in-the-loop) nằm đúng giao của scaling, alignment và hệ thống ML production — không chỉ “huấn luyện model cho accuracy cao hơn”.

## 4. Mã minh họa
```python
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
Tính toán neuromorphic và kiến trúc lấy cảm hứng từ não gợi ý cách tiếp cận khác biệt căn bản. Mạng neuron spiking xử lý sự kiện rời rạc (spike) thay vì giá trị liên tục, tiềm năng hiệu quả hơn và khả dĩ sinh học hơn. Phần cứng chuyên cho tính toán neuron (TPU, chip neuromorphic) đồng thiết kế thuật toán và phần cứng vì hiệu quả.

Học máy lượng tử (*quantum machine learning*) khám phá tính toán lượng tử cho ML, tiềm năng mang lại tăng tốc hàm mũ cho một số phép toán. Dù hiện chủ yếu lý thuyết, lợi thế lượng tử cho tác vụ ML cụ thể có thể xuất hiện khi phần cứng lượng tử trưởng thành.

Tìm kiếm kiến trúc neuron tự động hóa thiết kế kiến trúc, khám phá kiến trúc mới (EfficientNet, mạng dẫn xuất NAS) mà con người có thể không nghĩ ra. Tương lai: AI thiết kế hệ thống AI, đồng tiến hóa kiến trúc và thủ tục huấn luyện.

## 6. Các Bài báo Nền tảng

**["Attention is All You Need" (2017)](https://arxiv.org/abs/1706.03762)**  
Tác động của Transformer tiếp tục lan rộng — nền tảng cho GPT, BERT, về cơ bản mọi LLM hiện đại. Hiểu Transformer là hiểu tương lai của học sâu.

**["CLIP: Learning Transferable Visual Models From Natural Language Supervision" (2021)](https://arxiv.org/abs/2103.00020)**  
*Tác giả*: Alec Radford, Jong Wook Kim, et al. (OpenAI)  
CLIP học biểu diễn thị giác–ngôn ngữ từ 400M cặp ảnh–văn bản, cho phép phân loại ảnh zero-shot qua prompt văn bản. Chứng minh sức mạnh của tiền huấn luyện đa phương thức và đặc tả tác vụ linh hoạt qua ngôn ngữ tự nhiên.

**["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)](https://arxiv.org/abs/2010.11929)**  
Vision Transformer cho thấy kiến trúc dựa trên attention có thể ngang hoặc vượt CNN trên thị giác, gợi ý Transformer có thể trở thành kiến trúc phổ quát qua các phương thức.

**["Training Compute-Optimal Large Language Models" (2022)](https://arxiv.org/abs/2203.15556)**  
*Tác giả*: Jordan Hoffmann, Sebastian Borgeaud, et al. (DeepMind)  
Bài báo Chinchilla cho thấy hầu hết LLM bị undertrain — mở rộng tối ưu cân bằng kích thước mô hình và dữ liệu. Ảnh hưởng cách ta nghĩ về định luật scaling và phân bổ tài nguyên.

**["Sparks of Artificial General Intelligence: Early experiments with GPT-4" (2023)](https://arxiv.org/abs/2303.12712)**  
*Tác giả*: Microsoft Research  
Phân tích năng lực của GPT-4 trên nhiều tác vụ đa dạng, ghi nhận khả năng nổi gợi ý tiến bộ hướng tới trí thông minh tổng quát hơn. Dù gây tranh luận, làm nổi bật sự tiến bộ năng lực nhanh chóng.

## Bẫy Thường gặp và Mẹo

Phóng đại năng lực ngắn hạn trong khi đánh giá thấp tiềm năng dài hạn là phổ biến. Hệ thống hiện tại có hạn chế đáng kể (thiếu lẽ thường, suy luận mong manh, kém hiệu quả dữ liệu) sẽ không được giải trong năm tới. Song tiến bộ dài hạn (10–20 năm) có thể kịch tính hơn hiện có thể hình dung.

## Điểm Chính Cần Nhớ

Tương lai của học sâu liên quan mở rộng sang mô hình lớn hơn khám phá năng lực nổi, hệ thống đa phương thức tích hợp thị giác–ngôn ngữ–âm thanh cho hiểu biết phong phú hơn, đổi mới hiệu quả cho phép tiếp cận dân chủ hóa dù kích thước mô hình tăng, học few-shot và meta-learning giảm yêu cầu dữ liệu, học liên tục cho phép học trọn đời không quên, và nghiên cứu cơ bản về độ vững chắc, công bằng và căn chỉnh đảm bảo triển khai có lợi. Các thách thức còn mở gồm hiệu quả mẫu (học từ ít dữ liệu hơn), suy luận và lẽ thường (vượt khớp mẫu), diễn giải (hiểu quyết định), độ vững chắc (xử lý dịch phân bố), và khả năng mở rộng (huấn luyện và phục vụ mô hình ngày càng lớn). Lĩnh vực tiến bộ qua các tiến bộ song song về kiến trúc (Transformer), kỹ thuật huấn luyện (học tự giám sát), ứng dụng (mô hình đa phương thức), và lý thuyết (hiểu vì sao học sâu hoạt động), với đổi mới đột phá thường đến từ hướng bất ngờ. Hiểu các biên giới hiện tại và bài toán còn mở chuẩn bị cho người thực hành đóng góp vào sự tiến hóa tiếp tục của học sâu trong khi duy trì kỳ vọng thực tế về năng lực ngắn hạn và tiềm năng dài hạn.

Tương lai của học sâu sẽ được định hình bởi đổi mới kỹ thuật, tiến bộ tính toán, và xem xét thấu đáo tác động xã hội, đòi hỏi cả nghiên cứu tham vọng đẩy năng lực tiến lên lẫn công việc cẩn thận đảm bảo hệ thống mang lại lợi ích cho nhân loại.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Thuật toán gợi ý (TikTok-style recommendation)](https://www.facebook.com/reel/1444915507374636)
