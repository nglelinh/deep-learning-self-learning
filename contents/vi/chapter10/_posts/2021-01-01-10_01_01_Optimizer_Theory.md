---
layout: post
title: 10-01-01 Lý thuyết Bộ tối ưu
chapter: '10'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter10
---

Vào năm 2014, Diederik Kingma và Jimmy Ba công bố bài báo về **Adam** (*Adaptive Moment Estimation*) — một trong những bài báo được trích dẫn nhiều nhất mọi thời đại trong khoa học máy tính (hơn 180.000 lượt trích dẫn). Nhưng để không dùng Adam như một chiếc "hộp đen", ta cần giải phẫu từng mảnh ghép toán học đã tạo nên nó: từ lực quán tính của Momentum, cú "nhìn trước" của Nesterov, bài toán triệt tiêu tốc độ học của AdaGrad, cách RMSprop sửa lỗi bằng trung bình trượt hàm mũ, cho đến bước hiệu chỉnh độ lệch thiên về 0 ở những vòng lặp đầu tiên.

## Kiến thức cần nhớ: Bốn trụ cột toán học của các Bộ tối ưu

1. **Momentum và Nesterov Accelerated Gradient (NAG)**:
   Momentum cộng dồn vận tốc $$v_t = \beta v_{t-1} + g_t$$ (giống hòn bi nặng lăn xuống dốc). Nesterov tinh tế hơn: nó tính gradient tại **vị trí dự đoán phía trước** $$\theta_{t-1} - \eta \beta v_{t-1}$$ để kịp "hãm phanh" khi sắp vượt qua đáy thung lũng.
2. **AdaGrad và điểm yếu tích lũy đơn điệu**:
   AdaGrad cộng dồn toàn bộ bình phương gradient từ bước 1: $$G_t = G_{t-1} + g_t^2$$. Rất hay cho đặc trưng thưa (NLP cổ điển, hệ gợi ý), nhưng trong mạng sâu $$G_t$$ tăng mãi không giảm khiến tốc độ học hiệu dụng $$\frac{\eta}{\sqrt{G_t} + \epsilon} \to 0$$ quá sớm!
3. **RMSprop — Quên dần quá khứ bằng Trung bình trượt mũ (EMA)**:
   Thay tổng tích lũy bằng $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$ (thường $$\beta_2 = 0{,}99$$ hoặc $$0{,}999$$) để chỉ đo độ lớn gradient gần đây, giữ tốc độ học không bao giờ bị chết hẳn.
4. **Adam và Hiệu chỉnh độ lệch (Bias Correction)**:
   Vì $$m_0 = 0$$ và $$v_0 = 0$$, ở bước $$t = 1$$ ta có $$v_1 = (1 - 0{,}999) g_1^2 = 0{,}001 g_1^2$$ — nhỏ hơn giá trị thật tới **1.000 lần**! Chia cho $$(1 - \beta_1^t)$$ và $$(1 - \beta_2^t)$$ đưa kỳ vọng ước lượng về đúng không chệch ngay từ bước đầu tiên.

### Giải thích: Tính tay trọn vẹn 2 bước đầu tiên của Adam để thấy phép màu của Bias Correction

Cho tham số khởi đầu $$\theta_0 = 1{,}0$$, tốc độ học $$\eta = 0{,}1$$, các hệ số chuẩn $$\beta_1 = 0{,}9, \; \beta_2 = 0{,}999, \; \epsilon = 10^{-8}$$, và giả sử gradient đo được ở hai bước đầu đều là $$g_1 = g_2 = 2{,}0$$:

**Bước $$t = 1$$ (khởi tạo $$m_0 = 0, \; v_0 = 0$$)**:
1. Cập nhật moment thô:
   $$m_1 = 0{,}9(0) + 0{,}1(2{,}0) = \mathbf{0{,}20}, \qquad v_1 = 0{,}999(0) + 0{,}001(2{,}0^2) = \mathbf{0{,}004}$$
2. Hiệu chỉnh độ lệch (*Bias Correction*):
   $$\hat{m}_1 = \frac{0{,}20}{1 - 0{,}9^1} = \mathbf{2{,}00}, \qquad \hat{v}_1 = \frac{0{,}004}{1 - 0{,}999^1} = \mathbf{4{,}00}$$
3. Cập nhật tham số:
   $$\theta_1 = 1{,}0 - 0{,}1 \cdot \frac{2{,}00}{\sqrt{4{,}00} + 10^{-8}} = 1{,}0 - 0{,}1(1{,}0) = \mathbf{0{,}90}$$

**Thử hình dung nếu BỎ hiệu chỉnh độ lệch ở bước 1**: bước nhảy sẽ là $$0{,}1 \times \frac{0{,}20}{\sqrt{0{,}004}} \approx 0{,}316$$ — vọt gấp **3,16 lần** tốc độ học $$\eta = 0{,}1$$! Nhờ Bias Correction, bước đầu tiên có biên độ chính xác bằng $$\eta = 0{,}1$$, bất kể gradient lớn hay nhỏ!

# Tối ưu Nâng cao: Vượt qua Hạ Gradient Thuần túy

## 1. Tổng quan khái niệm

Dù hạ gradient cung cấp nguyên lý nền tảng để huấn luyện mạng nơ-ron — cập nhật tham số theo hướng làm giảm hàm mất mát — dạng thuần túy của nó tồn tại nhiều hạn chế khiến việc huấn luyện mạng sâu trở nên kém hiệu quả. Trước hết, learning rate phải được điều chỉnh rất cẩn thận: quá lớn sẽ gây dao động hoặc phân kỳ, còn quá nhỏ lại khiến quá trình hội tụ diễn ra rất chậm. Bên cạnh đó, thuật toán áp dụng cùng một learning rate cho mọi tham số, mặc dù các tham số khác nhau thường có thang độ lớn gradient và tần suất cập nhật tối ưu khác nhau. Hạ gradient thuần túy cũng đối xử với mọi hướng trong không gian tham số như nhau, trong khi bề mặt mất mát thường có dạng "hẻm núi" (dốc đứng theo một chiều nhưng rất thoải theo chiều kia), đòi hỏi bước đi khác nhau theo từng phương. Cuối cùng, vì không lưu giữ lịch sử các gradient trước đó, phương pháp này không thể tích lũy động lượng để vượt qua các cực tiểu địa phương nông hoặc điểm yên ngựa.

Các thuật toán tối ưu nâng cao khắc phục những hạn chế trên thông qua ba cơ chế chính: duy trì động lượng (*momentum*) để tăng tốc theo các hướng nhất quán đồng thời hạn chế dao động; điều chỉnh learning rate thích nghi cho từng tham số dựa trên lịch sử gradient, cho phép cập nhật mạnh hơn với các đặc trưng thưa và thận trọng hơn với các đặc trưng có gradient lớn xuất hiện thường xuyên; và khai thác thông tin bậc hai xấp xỉ về độ cong của bề mặt mất mát mà không phải chịu chi phí tính toán quá lớn của việc tính toàn bộ ma trận Hessian. Đây không phải là những tinh chỉnh thứ yếu mà là các kỹ thuật cốt lõi giúp huấn luyện những mô hình ngày càng sâu và lớn — các mô hình ngôn ngữ hiện đại với hàng tỷ tham số khó có thể huấn luyện hiệu quả chỉ bằng hạ gradient thuần túy.

Hiểu rõ các bộ tối ưu nghĩa là nhận ra chúng không loại trừ lẫn nhau mà phù hợp với từng bối cảnh khác nhau. Stochastic Gradient Descent (SGD) kết hợp momentum hoạt động hiệu quả và ít tốn chi phí tính toán khi bề mặt mất mát có hướng gradient nhất quán, nên thường được dùng trong các bài toán thị giác máy tính với batch size lớn. RMSprop điều chỉnh learning rate dựa trên độ lớn gradient gần đây, đặc biệt hữu ích với mạng hồi quy khi thang gradient thay đổi mạnh qua các bước thời gian. Adam kết hợp cả momentum và learning rate thích nghi, cho hiệu năng mặc định ổn định trên nhiều bài toán và trở thành lựa chọn phổ biến trong thực tế. AdamW cải tiến cách Adam xử lý weight decay, đóng vai trò quan trọng khi huấn luyện các mô hình Transformer lớn. Mỗi bộ tối ưu dựa trên những giả định khác nhau về bề mặt mất mát và động lực gradient; chọn đúng thuật toán có thể quyết định mô hình hội tụ nhanh hay chậm, thậm chí thành công hay thất bại.

Sự phát triển của các thuật toán tối ưu luôn song hành với tiến hóa kiến trúc mạng nơ-ron. Khi mạng trở nên sâu hơn và đối mặt với hiện tượng gradient biến mất hoặc bùng nổ, các bộ tối ưu được cải tiến để tích lũy động lượng và thích nghi learning rate. Khi mô hình lớn dần và buộc phải huấn luyện với batch nhỏ hơn do giới hạn bộ nhớ, các thuật toán lại cần hoạt động ổn định trước ước lượng gradient nhiễu. Khi phạm vi ứng dụng mở rộng từ thị giác sang xử lý ngôn ngữ tự nhiên (NLP) và học tăng cường, bộ tối ưu ngày càng phải thích nghi tốt hơn với nhiều dạng bề mặt gradient khác nhau. Quá trình đồng tiến hóa này vẫn đang tiếp diễn: kiến trúc mới thường kéo theo nhu cầu về bộ tối ưu mới, và ngược lại.

Tuy nhiên, dù các thuật toán ngày càng tinh vi, nguyên lý cốt lõi vẫn không đổi: ta vẫn tính gradient thông qua lan truyền ngược và cập nhật tham số ngược hướng gradient. Các bộ tối ưu nâng cao chỉ thay đổi cách xác định kích thước và hướng bước đi bằng cách tận dụng lịch sử và thống kê gradient. Vì vậy, hiểu vững hạ gradient thuần túy là nền tảng để nắm bắt mọi biến thể sau này — nên xem chúng như những cải tiến có chủ đích nhằm xử lý các chế độ thất bại cụ thể hơn là các phương pháp tách biệt hoàn toàn.

## 2. Nền tảng toán học

Ta sẽ xây dựng cơ sở toán học của các bộ tối ưu nâng cao theo trình tự từ đơn giản đến phức tạp, bắt đầu từ momentum để thấy rõ vai trò của từng thành phần.

### Momentum: Tích lũy vận tốc

Hạ gradient thuần túy cập nhật tham số chỉ dựa trên gradient tại bước hiện tại:

$$\theta_t = \theta_{t-1} - \eta \nabla_\theta \mathcal{L}(\theta_{t-1})$$

Momentum bổ sung một biến vận tốc để tích lũy gradient theo thời gian:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta \mathbf{v}_t$$

trong đó $$\beta \in [0, 1)$$ là hệ số momentum (thường chọn bằng 0,9). Vận tốc $$\mathbf{v}_t$$ là tổng có trọng số giảm dần theo hàm mũ của các gradient. Khai triển đệ quy cho thấy ảnh hưởng của các gradient quá khứ lên bước cập nhật hiện tại:

$$\mathbf{v}_t = \nabla_\theta \mathcal{L}(\theta_{t-1}) + \beta \nabla_\theta \mathcal{L}(\theta_{t-2}) + \beta^2 \nabla_\theta \mathcal{L}(\theta_{t-3}) + \ldots$$

Gradient gần nhất giữ nguyên trọng số, trong khi các gradient cũ hơn giảm dần theo hệ số $$\beta^k$$. Cơ chế này mang lại ba lợi ích chính. Thứ nhất, nếu gradient liên tục cùng chiều, vận tốc sẽ tích lũy và tăng tốc quá trình dịch chuyển — tương tự một quả bóng lăn xuống dốc. Thứ hai, nếu gradient liên tục đổi dấu, các thành phần ngược chiều sẽ triệt tiêu một phần, giúp giảm dao động. Thứ ba, momentum giúp quá trình tối ưu vượt qua các cực tiểu địa phương nông hoặc vùng phẳng nơi gradient hiện tại gần bằng 0 nhưng các gradient trước đó vẫn duy trì hướng đi hợp lý.

Về mặt hình học, momentum chuyển vai trò của gradient từ bước dịch chuyển trực tiếp sang thành phần cộng vào vận tốc. Dù phép tương tự với cơ học cổ điển không hoàn toàn tuyệt đối, nó phản ánh chính xác tính quán tính: tham số tiếp tục đi theo hướng tốt đã tích lũy ngay cả khi gradient tại một bước riêng lẻ hơi lệch hướng.

### Nesterov Accelerated Gradient (NAG)

NAG là một cải tiến của momentum, trong đó gradient không được tính tại vị trí hiện tại mà tại vị trí dự kiến sau bước trượt theo quán tính:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1} - \eta \beta \mathbf{v}_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta \mathbf{v}_t$$

Điểm khác biệt nằm ở số hạng $$\nabla_\theta \mathcal{L}(\theta_{t-1} - \eta \beta \mathbf{v}_{t-1})$$ thay cho $$\nabla_\theta \mathcal{L}(\theta_{t-1})$$. Ta đánh giá gradient tại điểm mà động lượng sắp đưa tới, rồi dùng thông tin đó để hiệu chỉnh hướng đi. Nhờ cơ chế "nhìn trước" này, nếu quán tính đang đẩy tham số vào vùng có loss tăng, gradient tại vị trí dự kiến sẽ kịp thời phản hồi để giảm tốc hoặc đổi hướng.

So với momentum chuẩn, NAG thường hội tụ nhanh hơn và ít bị vượt quá điểm cực tiểu hơn. Có thể hình dung momentum chuẩn mang tính phản ứng (điều chỉnh dựa trên vị trí hiện tại), còn NAG mang tính chủ động (dự đoán vị trí kế tiếp để hiệu chỉnh sớm). Trong thực tế, chênh lệch hiệu năng giữa hai phương pháp đôi khi không lớn, nhưng NAG có cơ sở lý thuyết chặt chẽ hơn và thường đem lại cải thiện ổn định.

### AdaGrad: Learning Rate thích nghi

AdaGrad điều chỉnh learning rate riêng cho từng tham số dựa trên tổng bình phương các gradient đã tích lũy:

$$\mathbf{G}_t = \mathbf{G}_{t-1} + (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

trong đó phép bình phương và căn bậc hai được thực hiện theo từng phần tử, $$\mathbf{G}_t$$ lưu tổng bình phương gradient, và $$\epsilon$$ (thường là $$10^{-8}$$) giúp tránh chia cho 0. Việc chia cho $$\sqrt{\mathbf{G}_t + \epsilon}$$ khiến các tham số có gradient tích lũy lớn nhận bước cập nhật nhỏ hơn, còn các tham số có gradient tích lũy nhỏ nhận bước cập nhật lớn hơn.

Cơ chế này giải quyết tốt bài toán dữ liệu thưa — chẳng hạn trong NLP, nơi phần lớn từ vựng chỉ xuất hiện rất ít trong tập tài liệu. Với các đặc trưng hiếm gặp, tham số tương ứng ít được cập nhật nên $$\mathbf{G}_t$$ nhỏ, giúp chúng nhận bước cập nhật lớn hơn mỗi khi xuất hiện; ngược lại, các đặc trưng phổ biến sẽ nhận bước cập nhật nhỏ và ổn định hơn.

Tuy nhiên, AdaGrad có nhược điểm lớn khi huấn luyện kéo dài: $$\mathbf{G}_t$$ là tổng các số không âm nên chỉ tăng chứ không bao giờ giảm. Theo thời gian, $$\sqrt{\mathbf{G}_t}$$ trở nên rất lớn, kéo learning rate hiệu dụng về gần 0 và làm quá trình học dừng lại quá sớm. Điều này có thể chấp nhận được trong tối ưu lồi, nhưng với bề mặt mất mát không lồi của mạng nơ-ron (nhiều cực tiểu địa phương, vùng phẳng và điểm yên ngựa), việc suy giảm bước học quá nhanh sẽ ngăn mô hình thoát khỏi các vùng nghiệm kém.

### RMSprop: Trung bình trượt có trọng số mũ

RMSprop khắc phục sự suy giảm quá mức của AdaGrad bằng cách thay tổng tích lũy bằng trung bình trượt có trọng số mũ của bình phương gradient:

$$\mathbf{E}[g^2]_t = \beta \mathbf{E}[g^2]_{t-1} + (1-\beta)(\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\mathbf{E}[g^2]_t + \epsilon}} \odot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

Với giá trị phổ biến $$\beta = 0.9$$, thuật toán xét trọng số chủ yếu trên khoảng $$1/(1-\beta) = 10$$ bước cập nhật gần nhất. Nhờ khả năng "quên" các gradient quá cũ, learning rate thích nghi có thể tự điều chỉnh lại khi thang gradient thay đổi qua các vùng khác nhau của bề mặt mất mát. RMSprop đặc biệt hữu ích khi huấn luyện mạng hồi quy (RNN), nơi độ lớn gradient biến thiên mạnh, và vẫn là lựa chọn tốt khi phân phối thống kê của gradient thay đổi trong suốt quá trình huấn luyện.

### Adam: Ước lượng Moment thích nghi

Adam kết hợp ý tưởng momentum và learning rate thích nghi của RMSprop bằng cách duy trì đồng thời ước lượng moment bậc nhất (trung bình) và moment bậc hai (phương sai chưa trừ trung bình) của gradient:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}(\theta_{t-1})$$ (thành phần momentum)

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$ (thành phần RMSprop)

Vì khởi tạo $$\mathbf{m}_0 = \mathbf{v}_0 = 0$$, hai ước lượng này ban đầu bị lệch về 0. Adam khắc phục bằng bước hiệu chỉnh độ lệch (*bias correction*):

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

Quy tắc cập nhật tham số kết hợp cả hai ước lượng đã hiệu chỉnh:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

Các siêu tham số mặc định $$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$$ hoạt động ổn định trên nhiều bài toán, khiến Adam trở thành bộ tối ưu được ưa chuộng vì ít đòi hỏi tinh chỉnh. Thuật toán vừa thích nghi theo thang độ lớn gradient (qua $$\mathbf{v}_t$$), vừa duy trì động lượng (qua $$\mathbf{m}_t$$).

Cơ chế hiệu chỉnh độ lệch có vai trò rất cụ thể ở giai đoạn đầu huấn luyện. Do khởi tạo bằng 0, tại bước đầu tiên ta có $$\mathbf{m}_1 = (1-\beta_1)g_1$$, nhỏ hơn nhiều so với kỳ vọng thật $$\mathbb{E}[g]$$ khi $$\beta_1 = 0.9$$. Phép chia cho $$1-\beta_1^t$$ đưa ước lượng về đúng thang: $$\hat{\mathbf{m}}_1 = \frac{(1-\beta_1)g_1}{1-\beta_1} = g_1$$. Khi $$t \to \infty$$, $$\beta_1^t \to 0$$ nên mẫu số tiến về 1 và phép hiệu chỉnh tự động mờ dần, giúp thuật toán ổn định ngay từ bước đầu tiên mà không ảnh hưởng đến hành vi tiệm cận.

### AdamW: Weight Decay tách rời

Một điểm tinh tế ở Adam chuẩn là cách xử lý chính quy hóa L2 (weight decay). Thông thường, weight decay được thêm trực tiếp vào gradient:

$$\nabla \mathcal{L}_{\text{reg}} = \nabla \mathcal{L} + \lambda \theta$$

Tuy nhiên, trong Adam, thành phần này sau đó lại đi qua phép chia cho $$\sqrt{\hat{\mathbf{v}}_t} + \epsilon$$, làm cường độ chính quy hóa bị biến dạng theo thang gradient của từng tham số. AdamW giải quyết bằng cách tách weight decay ra khỏi bước cập nhật thích nghi dựa trên gradient:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

Số hạng weight decay $$\lambda \theta_{t-1}$$ được áp dụng độc lập sau bước chuẩn hóa thích nghi, giúp cường độ suy giảm trọng số đồng đều hơn và không phụ thuộc vào thống kê gradient. Thay đổi nhỏ này cải thiện rõ rệt khả năng tổng quát hóa, đặc biệt trên Transformer và các mô hình quy mô lớn.

## 3. Ví dụ và Trực giác

Để hình dung sự khác biệt giữa các bộ tối ưu, hãy xét một bề mặt mất mát 2D có dạng hẻm núi: hai sườn rất dốc (độ cong cao) nhưng đáy hẻm lại thoải (độ cong thấp), với điểm cực tiểu nằm sâu dọc theo đáy hẻm.

**Hạ Gradient Thuần túy:** Gradient luôn vuông góc với các đường đồng mức của hàm mất mát. Trong hẻm núi, gradient chủ yếu hướng thẳng về hai sườn dốc và chỉ có thành phần rất nhỏ dọc theo đáy hẻm. Do đó, thuật toán liên tục nhảy qua lại giữa hai sườn và nhích rất chậm về phía cực tiểu.

**SGD với Momentum:** Các thành phần gradient cùng chiều dọc theo đáy hẻm cộng dồn thành vận tốc lớn, trong khi các thành phần đổi dấu liên tục giữa hai sườn dốc tự triệt tiêu lẫn nhau. Nhờ vậy, quỹ đạo tăng tốc dọc theo đáy hẻm và giảm hẳn dao động ngang. Tương tự quả bóng lăn xuống dốc, quán tính cũng giúp quá trình tối ưu băng qua các vùng phẳng hoặc vùng lõm nông.

**AdaGrad / RMSprop:** Thuật toán nhận thấy gradient theo phương dốc luôn lớn còn theo phương thoải luôn nhỏ, từ đó tự động giảm bước học theo phương dốc (hạn chế dao động) và giữ bước học lớn hơn theo phương thoải (tăng tốc tiến về cực tiểu). Việc tự cân bằng theo độ cong từng chiều giúp đường đi hiệu quả hơn nhiều so với dùng một learning rate duy nhất.

**Adam:** Kết hợp cả hai lợi thế trên — momentum tăng tốc dọc theo đáy hẻm còn learning rate thích nghi triệt tiêu dao động ngang. Ngoài ra, khi tiến dần từ vùng xa cực tiểu (gradient lớn) về gần cực tiểu (gradient nhỏ), cơ chế thích nghi tự động điều chỉnh thang bước cập nhật tương ứng.

Xét thêm một tình huống thực tế: huấn luyện mạng nơ-ron trên tập dữ liệu có những đặc trưng hiếm nhưng mang thông tin quan trọng. SGD thuần túy dùng chung một learning rate nên các tham số gắn với đặc trưng hiếm được cập nhật rất ít và học chậm. Trong khi đó, AdaGrad và Adam cấp cho các tham số này learning rate hiệu dụng lớn hơn (do $$\mathbf{v}_t$$ tích lũy nhỏ), giúp chúng học nhanh từ số ít mẫu xuất hiện; còn các đặc trưng phổ biến nhận learning rate hiệu dụng nhỏ hơn để tránh phản ứng quá mạnh trước từng mẫu riêng lẻ. Đây là lý do Adam thường hội tụ nhanh hơn SGD trong các bài toán NLP có từ vựng thưa.

## Video tham khảo

- **Xem trước:** [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, tiếng Anh. Giải thích cách mỗi thuật toán lấy trung bình các gradient quá khứ, bám sát phần lý thuyết của bài.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, tiếng Anh. Ôn lại bước cập nhật cơ bản trước khi bổ sung trung bình hóa và moment.
- [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown, tiếng Anh. Trực giác hình học về việc tối thiểu hóa hàm chi phí trước khi đi vào các thuật toán theo dõi moment.
