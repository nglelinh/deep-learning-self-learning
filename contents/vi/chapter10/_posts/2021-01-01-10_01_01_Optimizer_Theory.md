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

# Tối ưu Nâng cao: Vượt ngoài Hạ Gradient Thuần túy

## 1. Tổng quan khái niệm
Dù hạ gradient cung cấp nguyên lý nền tảng để huấn luyện mạng neuron — di chuyển tham số theo hướng giảm loss — dạng thuần túy của nó chịu nhiều hạn chế nghiêm trọng khiến việc huấn luyện mạng sâu trở nên không thực tế. Learning rate phải được chỉnh cẩn thận: quá lớn gây dao động hoặc phân kỳ, quá nhỏ khiến hội tụ chậm đau đớn. Cùng một learning rate được dùng cho mọi tham số, dù các tham số khác nhau có thang gradient và tần suất cập nhật tối ưu khác nhau. Hạ gradient đối xử mọi hướng trong không gian tham số như nhau, dù một số hướng là “hẻm núi” (dốc theo một chiều, thoải theo chiều kia) nơi ta cần di chuyển cẩn thận. Và nó không có bộ nhớ về các gradient trước, không thể tích lũy động lượng để thoát cực tiểu địa phương nông hay điểm yên ngựa.

Các thuật toán tối ưu nâng cao giải quyết các hạn chế này qua nhiều cơ chế: duy trì động lượng (*momentum*) để tăng tốc theo hướng nhất quán đồng thời dập dao động; thích nghi learning rate theo từng tham số dựa trên lịch sử gradient, cho phép cập nhật mạnh với gradient thưa và cập nhật bảo thủ với gradient lớn thường xuyên; và kết hợp thông tin bậc hai về độ cong của bề mặt loss mà không tốn chi phí cấm đoán của việc tính Hessian đầy đủ. Những cải tiến này không phải chỉnh sửa nhỏ mà là kỹ thuật thiết yếu đã cho phép huấn luyện các mô hình ngày càng lớn và phức tạp — các mô hình ngôn ngữ hiện đại với hàng tỷ tham số đơn giản là không thể huấn luyện bằng hạ gradient thuần túy.

Hiểu sâu các bộ tối ưu nghĩa là nhận ra chúng không phải lựa chọn cạnh tranh mà là công cụ phù hợp với các tình huống khác nhau. Stochastic Gradient Descent với momentum xuất sắc khi bề mặt loss có hướng gradient rõ ràng, nhất quán và hiệu quả tính toán, nên phổ biến với tác vụ thị giác có batch size lớn. RMSprop thích nghi learning rate dựa trên độ lớn gradient gần đây, đặc biệt hữu ích với mạng hồi quy nơi thang gradient thay đổi mạnh qua các bước thời gian. Adam kết hợp momentum và learning rate thích nghi, cho hiệu năng mặc định tốt trên nhiều tác vụ và trở thành chuẩn thực tế cho nhiều ứng dụng. AdamW cải thiện cách Adam xử lý weight decay, then chốt khi huấn luyện Transformer lớn. Mỗi bộ tối ưu hiện thân các giả định khác nhau về bề mặt loss và động lực gradient, và lựa chọn phù hợp có thể nghĩa là sự khác biệt giữa mô hình huấn luyện trong vài giờ so với vài ngày, hoặc huấn luyện thành công so với thất bại.

Sự tiến hóa của thuật toán tối ưu song hành với tiến hóa kiến trúc neuron. Khi mạng trở nên sâu hơn (cần kỹ thuật xử lý gradient biến mất/bùng nổ), bộ tối ưu tiến hóa để thích nghi learning rate và xây động lượng. Khi mạng trở nên lớn hơn (cần huấn luyện với batch nhỏ hơn do hạn chế bộ nhớ), bộ tối ưu được phát triển để làm việc hiệu quả với ước lượng gradient nhiễu. Khi tác vụ đa dạng hóa (từ thị giác sang NLP sang học tăng cường), bộ tối ưu trở nên thích nghi hơn với các cảnh quan gradient khác nhau. Sự đồng tiến hóa này của kiến trúc và bộ tối ưu vẫn đang tiếp diễn — kiến trúc mới thường cần đổi mới bộ tối ưu, và bộ tối ưu mới cho phép kiến trúc mới.

Song với mọi thuật toán tinh vi này, nền tảng vẫn: ta vẫn tính gradient qua lan truyền ngược và bước ngược hướng các gradient đó. Các bộ tối ưu nâng cao thay đổi cách ta xác định kích thước và hướng bước, khai thác lịch sử và thống kê gradient, nhưng nguyên lý cốt lõi — tinh chỉnh lặp dựa trên gradient của loss — giữ nguyên. Điều này nghĩa là hiểu sâu hạ gradient thuần túy cung cấp nền tảng để hiểu mọi biến thể, tốt nhất nên xem chúng như các chỉnh sửa tinh vi giải quyết các chế độ thất bại cụ thể chứ không phải các cách tiếp cận hoàn toàn khác.

## 2. Nền tảng toán học
Hãy xây dựng toán học của các bộ tối ưu nâng cao một cách có hệ thống, hiểu mục đích của từng thành phần và cách chúng kết hợp để cải thiện hạ gradient thuần túy. Ta bắt đầu với momentum rồi tiến tới các kỹ thuật ngày càng tinh vi.

### Momentum: Xây dựng Vận tốc

Hạ gradient thuần túy cập nhật tham số chỉ dùng gradient hiện tại:

$$\theta_t = \theta_{t-1} - \eta \nabla_\theta \mathcal{L}(\theta_{t-1})$$

Momentum giới thiệu một số hạng vận tốc tích lũy gradient theo thời gian:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta \mathbf{v}_t$$

trong đó $$\beta \in [0, 1)$$ là hệ số momentum (thường 0,9). Vận tốc $$\mathbf{v}_t$$ là trung bình trượt có trọng số mũ của gradient. Mở rộng đệ quy cho thấy các gradient quá khứ ảnh hưởng cập nhật hiện tại như thế nào:

$$\mathbf{v}_t = \nabla_\theta \mathcal{L}(\theta_{t-1}) + \beta \nabla_\theta \mathcal{L}(\theta_{t-2}) + \beta^2 \nabla_\theta \mathcal{L}(\theta_{t-3}) + \ldots$$

Gradient gần đây có trọng số đầy đủ, trong khi gradient cũ hơn đóng góp với trọng số giảm mũ $$\beta^k$$. Điều này tạo ra nhiều hiệu ứng có lợi. Thứ nhất, nếu gradient liên tục chỉ cùng hướng, vận tốc tích lũy, tăng tốc tiến trình — như quả bóng lăn xuống dốc tăng tốc. Thứ hai, nếu gradient dao động (dương rồi âm), vận tốc dập dao động — các gradient đối nghịch triệt tiêu một phần. Thứ ba, momentum có thể mang tối ưu qua cực tiểu địa phương nông hoặc vùng phẳng nơi gradient hiện tại gần zero nhưng gradient quá khứ chỉ ra hướng tốt.

Trực giác hình học là momentum biến gradient từ lực thành vận tốc. Trong vật lý, lực (gradient) gây gia tốc, dẫn đến thay đổi vận tốc. Ở đây, gradient đóng góp trực tiếp vào vận tốc, quyết định cập nhật vị trí. Ẩn dụ vật lý này không hoàn hảo nhưng nắm bắt cách momentum tạo quán tính — tối ưu tiếp tục di chuyển theo hướng trước đó tốt ngay cả khi gradient hiện tại hơi bất đồng.

### Nesterov Accelerated Gradient (NAG)

Một chỉnh sửa thông minh của momentum tính gradient không tại vị trí hiện tại mà tại vị trí tương lai dự kiến:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta \mathcal{L}(\theta_{t-1} - \beta \mathbf{v}_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta \mathbf{v}_t$$

Sự khác biệt then chốt là $$\nabla_\theta \mathcal{L}(\theta_{t-1} - \beta \mathbf{v}_{t-1})$$ thay vì $$\nabla_\theta \mathcal{L}(\theta_{t-1})$$. Ta tính gradient tại nơi momentum sẽ đưa ta đến, rồi dùng gradient đó để tinh chỉnh cập nhật. “Nhìn trước” này cung cấp một dạng hiệu chỉnh: nếu momentum đang mang ta tới vùng xấu, gradient tại vị trí dự kiến sẽ chỉ ra điều đó, cho phép ta chậm lại hoặc đổi hướng.

Cải thiện so với momentum chuẩn tinh tế nhưng nhất quán trên nhiều tác vụ. NAG thường hội tụ nhanh hơn và vượt đỉnh ít hơn tại cực tiểu. Trực giác là momentum chuẩn phản ứng (đáp ứng gradient tại vị trí hiện tại) trong khi NAG chủ động (dự đoán nơi ta đang đi và lập kế hoạch tương ứng). Trong thực tế, sự khác biệt giữa momentum và NAG thường nhỏ, nhưng NAG có động cơ lý thuyết tốt hơn và thỉnh thoảng mang lại cải thiện đáng chú ý.

### AdaGrad: Learning Rate Thích nghi

AdaGrad thích nghi learning rate theo từng tham số dựa trên bình phương gradient tích lũy:

$$\mathbf{G}_t = \mathbf{G}_{t-1} + (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

trong đó bình phương và căn bậc hai theo phần tử, $$\mathbf{G}_t$$ tích lũy bình phương gradient, và $$\epsilon$$ (thường $$10^{-8}$$) ngăn chia cho zero. Phép chia bởi $$\sqrt{\mathbf{G}_t}$$ nghĩa là tham số với gradient tích lũy lớn nhận cập nhật nhỏ hơn, trong khi tham số với gradient tích lũy nhỏ nhận cập nhật lớn hơn.

Scale thích nghi này giải quyết hạn chế then chốt của hạ gradient thuần túy. Với đặc trưng thưa (phổ biến trong NLP nơi hầu hết từ không xuất hiện trong hầu hết tài liệu), một số tham số hiếm khi nhận cập nhật gradient. AdaGrad cho các tham số thưa này cập nhật lớn hơn khi chúng nhận gradient, trong khi tham số cập nhật thường xuyên (tương ứng đặc trưng phổ biến) nhận cập nhật nhỏ hơn. Điều này đặc biệt giá trị với tác vụ dữ liệu thưa hoặc tần suất đặc trưng biến thiên mạnh.

Tuy nhiên, AdaGrad có khuyết điểm chí mạng với các lần chạy huấn luyện dài: $$\mathbf{G}_t$$ chỉ tăng, không bao giờ giảm. Khi huấn luyện tiến triển, $$\sqrt{\mathbf{G}_t}$$ trở nên rất lớn, làm learning rate hiệu dụng tiến về zero, và việc học dừng lại. Sự suy giảm learning rate mạnh này phù hợp với tối ưu lồi nơi ta muốn chậm lại khi tiến gần cực tiểu, nhưng bề mặt loss của mạng neuron là không lồi với nhiều cực tiểu địa phương, cao nguyên và điểm yên ngựa. Dừng thích nghi quá sớm ngăn thoát các vùng dưới tối ưu này.

### RMSprop: Trung bình Trượt Mũ

RMSprop sửa suy giảm mạnh của AdaGrad bằng cách dùng trung bình trượt có trọng số mũ của bình phương gradient thay vì tích lũy:

$$\mathbf{E}[g^2]_t = \beta \mathbf{E}[g^2]_{t-1} + (1-\beta)(\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\mathbf{E}[g^2]_t + \epsilon}} \odot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\beta = 0.9$$ điển hình nghĩa là ta xem xét khoảng $$1/(1-\beta) = 10$$ cập nhật gradient gần nhất. Điều này cho phép thuật toán quên gradient cũ, nên nếu thang gradient thay đổi (khi ta di chuyển qua các vùng khác nhau của bề mặt loss), thích nghi learning rate điều chỉnh. RMSprop trở nên đặc biệt phổ biến khi huấn luyện RNN nơi thang gradient biến thiên mạnh, và vẫn là lựa chọn vững khi thống kê gradient thay đổi trong huấn luyện.

### Adam: Ước lượng Moment Thích nghi

Adam kết hợp momentum và learning rate thích nghi của RMSprop, duy trì cả ước lượng moment bậc nhất (trung bình) và moment bậc hai (phương sai không tâm):

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}(\theta_{t-1})$$ (số hạng momentum)

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$ (số hạng RMSprop)

Các ước lượng này ban đầu lệch về zero (vì $$\mathbf{m}_0 = \mathbf{v}_0 = 0$$). Adam hiệu chỉnh độ lệch này:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

Quy tắc cập nhật kết hợp cả hai:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

Siêu tham số mặc định $$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$$ hoạt động tốt trên nhiều tác vụ, khiến Adam phổ biến như bộ tối ưu “ít chỉnh”. Thuật toán thích nghi với thống kê gradient (qua $$\mathbf{v}_t$$) đồng thời xây động lượng (qua $$\mathbf{m}_t$$), kết hợp lợi ích của cả hai cách tiếp cận.

Hiệu chỉnh độ lệch xứng đáng chú ý kỹ. Đầu huấn luyện, $$\mathbf{m}_t$$ và $$\mathbf{v}_t$$ bị thống trị bởi khởi tạo zero, khiến chúng là ước lượng lệch của moment thật. Ví dụ, $$\mathbf{m}_1 = (1-\beta_1)g_1$$ đánh giá thấp đáng kể $$\mathbb{E}[g]$$ khi $$\beta_1 = 0.9$$. Chia bởi $$1-\beta_1^t$$ hiệu chỉnh: $$\hat{\mathbf{m}}_1 = \frac{(1-\beta_1)g_1}{1-\beta_1} = g_1$$. Khi $$t \to \infty$$, $$\beta_1^t \to 0$$, nên hệ số hiệu chỉnh tiến về 1 và không còn ảnh hưởng. Điều này đảm bảo hành vi tốt từ cập nhật đầu tiên đồng thời tiệm cận hành xử như trung bình mũ không hiệu chỉnh.

### AdamW: Weight Decay Tách rời

Một vấn đề tinh tế với Adam là cách nó xử lý chính quy hóa L2 (weight decay). Thực hành chuẩn thêm $$\lambda \theta$$ vào gradient:

$$\nabla \mathcal{L}_{\text{reg}} = \nabla \mathcal{L} + \lambda \theta$$

Nhưng trong Adam, gradient đã chính quy hóa này bị xử lý qua learning rate thích nghi, có thể làm loãng hiệu ứng chính quy hóa. AdamW tách weight decay khỏi tối ưu dựa trên gradient:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}(\theta_{t-1})$$

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}(\theta_{t-1}))^2$$

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

Số hạng weight decay $$\lambda \theta_{t-1}$$ được thêm sau scale thích nghi, đảm bảo cường độ chính quy hóa độc lập với thống kê gradient. Thay đổi tưởng chừng nhỏ này cải thiện rõ rệt tổng quát hóa, đặc biệt với Transformer và các mô hình lớn nơi chính quy hóa đúng là then chốt.

## 3. Ví dụ / Trực giác

Để hiểu các bộ tối ưu khác nhau hành xử ra sao, hãy tưởng tượng tối ưu một hàm với hẻm núi: sườn dốc và dốc thoải dọc đáy hướng tới cực tiểu. Hình dung bề mặt loss 2D nơi một hướng có độ cong cao (dốc) và hướng vuông góc có độ cong thấp (thoải). Cực tiểu nằm ở đáy hẻm núi này.

**Hạ Gradient Thuần túy**: Bước vuông góc với đường mức loss không đổi. Trong hẻm núi, gradient chủ yếu chỉ về đáy hẻm (hướng dốc), hầu như không dọc theo nó (hướng thoải). Ta bước lớn về hai bên, nảy giữa chúng, và tiến chậm dọc hẻm về cực tiểu. Không hiệu quả — hầu hết độ lớn gradient nằm ở hướng sai (vuông góc đường tới cực tiểu) thay vì hướng đúng (dọc hẻm).

**SGD với Momentum**: Tích lũy vận tốc dọc hẻm khi các gradient nhất quán theo hướng đó xây động lượng. Khi gradient dao động vuông góc hẻm (dương rồi âm khi nảy giữa hai sườn), vận tốc theo hướng đó bị dập. Ta tăng tốc dọc hẻm trong khi dao động vuông góc bị triệt. Ẩn dụ quả bóng lăn xuống dốc rất phù hợp — momentum mang ta qua vùng phẳng và giúp thoát bát nông.

**AdaGrad/RMSprop**: Nhận ra gradient theo hướng dốc liên tục lớn, trong khi gradient theo hướng thoải nhỏ. Nó giảm learning rate theo hướng dốc (để ngăn nảy) và giữ nguyên theo hướng thoải (để tiến). Điều này tự động scale lại gradient dựa trên các độ cong khác nhau, cho phép bước hiệu dụng lớn hơn dọc hẻm ngay cả với bước nhỏ hơn vuông góc.

**Adam**: Kết hợp cả hai cơ chế. Momentum tăng tốc dọc hẻm. Learning rate thích nghi ngăn nảy quá mức. Kết quả là tiến trình nhanh, ổn định về cực tiểu. Adam cũng xử lý thực tế rằng thống kê gradient thay đổi khi ta di chuyển — đầu huấn luyện, xa cực tiểu, gradient lớn; gần cực tiểu, chúng thu nhỏ. Scale thích nghi điều chỉnh tự động.

Xét một kịch bản cụ thể: huấn luyện mạng neuron trên tập dữ liệu với đặc trưng hiếm nhưng quan trọng. SGD thuần túy cập nhật mọi tham số như nhau, nên đặc trưng hiếm được cập nhật không thường xuyên (chỉ khi các mẫu chứa chúng xuất hiện). AdaGrad/Adam cho các tham số này learning rate hiệu dụng lớn hơn (vì $$\mathbf{v}_t$$ của chúng nhỏ hơn, đã tích lũy ít cập nhật gradient hơn), cho phép chúng học nhanh từ ít mẫu chúng thấy. Đặc trưng phổ biến, cập nhật thường xuyên, nhận learning rate hiệu dụng nhỏ hơn, ngăn phản ứng thái quá với từng mẫu. Tính thích nghi này là lý do Adam thường hội tụ nhanh hơn SGD, đặc biệt trong NLP nơi độ thưa từ vựng cực đoan.
