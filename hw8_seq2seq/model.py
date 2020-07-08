import torch
from torch import nn, random

from hw8_seq2seq import config

'''
seq2seq模型的編碼器為RNN。 對於每個輸入，Encoder 會輸出一個向量和一個隱藏狀態(hidden state)，並將隱藏狀態用於下一個輸入，換句話說，Encoder 會逐步讀取輸入序列，並輸出單個矢量（最終隱藏狀態）
參數:
en_vocab_size 是英文字典的大小，也就是英文的 subword 的個數
emb_dim 是 embedding 的維度，主要將 one-hot vector 的單詞向量壓縮到指定的維度，主要是為了降維和濃縮資訊的功用，可以使用預先訓練好的 word embedding，如 Glove 和 word2vector
hid_dim 是 RNN 輸出和隱藏狀態的維度
n_layers 是 RNN 要疊多少層
dropout 是決定有多少的機率會將某個節點變為 0，主要是為了防止 overfitting ，一般來說是在訓練時使用，測試時則不使用
Encoder 的輸入和輸出:
輸入:
英文的整數序列 e.g. 1, 28, 29, 205, 2
輸出:
outputs: 最上層 RNN 全部的輸出，可以用 Attention 再進行處理
hidden: 每層最後的隱藏狀態，將傳遞到 Decoder 進行解碼
'''
class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = [batch size, sequence len, vocab size]
        embedding = self.embedding(input)
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        # outputs 是最上層RNN的輸出

        return outputs, hidden


'''
Decoder 是另一個 RNN，在最簡單的 seq2seq decoder 中，僅使用 Encoder 每一層最後的隱藏狀態來進行解碼，
而這最後的隱藏狀態有時被稱為 “content vector”，因為可以想像它對整個前文序列進行編碼， 
此 “content vector” 用作 Decoder 的初始隱藏狀態， 而 Encoder 的輸出通常用於 Attention Mechanism。

參數:
en_vocab_size：英文字典的大小，也就是英文的 subword 的個數
emb_dim：embedding 的維度，是用來將 one-hot vector 的單詞向量壓縮到指定的維度，主要是為了降維和濃縮資訊的功用，可以使用預先訓練好的 word embedding，如 Glove 和 word2vector
hid_dim：RNN 輸出和隱藏狀態的維度
output_dim：最終輸出的維度，一般來說是將 hid_dim 轉到 one-hot vector 的單詞向量
n_layers：RNN 要疊多少層
dropout：決定有多少的機率會將某個節點變為0，主要是為了防止 overfitting ，一般來說是在訓練時使用，測試時則不用
isatt：是來決定是否使用 Attention Mechanism

Decoder 的輸入和輸出:
輸入:前一次解碼出來的單詞的整數表示
輸出:
hidden: 根據輸入和前一次的隱藏狀態，現在的隱藏狀態更新的結果
output: 每個字有多少機率是這次解碼的結果
'''
class Decoder(nn.Module):
  def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
    super().__init__()
    self.cn_vocab_size = cn_vocab_size
    self.hid_dim = hid_dim * 2
    self.n_layers = n_layers
    self.embedding = nn.Embedding(cn_vocab_size, config.emb_dim)
    self.isatt = isatt
    self.attention = Attention(hid_dim)
    # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
    # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
    # self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
    self.input_dim = emb_dim
    self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
    self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
    self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
    self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input, hidden, encoder_outputs):
    # input = [batch size, vocab size]
    # hidden = [batch size, n layers * directions, hid dim]
    # Decoder 只會是單向，所以 directions=1
    input = input.unsqueeze(1)
    embedded = self.dropout(self.embedding(input))
    # embedded = [batch size, 1, emb dim]
    if self.isatt:
      attn = self.attention(encoder_outputs, hidden)
      # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
    output, hidden = self.rnn(embedded, hidden)
    # output = [batch size, 1, hid dim]
    # hidden = [num_layers, batch size, hid dim]

    # 將 RNN 的輸出轉為每個詞出現的機率
    output = self.embedding2vocab1(output.squeeze(1))
    output = self.embedding2vocab2(output)
    prediction = self.embedding2vocab3(output)
    # prediction = [batch size, vocab size]
    return prediction, hidden


'''
當輸入過長，或是單獨靠 “content vector” 無法取得整個輸入的意思時，用 Attention Mechanism 來提供 Decoder 更多的資訊。
主要是根據現在 Decoder hidden state ，去計算在 Encoder outputs 中，那些與其有較高的關係，根據關系的數值來決定該傳給 Decoder 那些額外資訊。
常見 Attention 的實作是用 Neural Network / Dot Product 來算 Decoder hidden state 和 Encoder outputs 之間的關係，
再對所有算出來的數值做 softmax ，最後根據過完 softmax 的值對 Encoder outputs 做 weight sum。
TODO: 實作 Attention Mechanism
'''
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim]
        # 一般來說是取最後一層的 hidden state 來做 attention
        ########
        # TODO #
        ########
        attention = None

        return attention


'''
由 Encoder 和 Decoder 組成
接收輸入並傳給 Encoder
將 Encoder 的輸出傳給 Decoder
不斷地將 Decoder 的輸出傳回 Decoder ，進行解碼
當解碼完成後，將 Decoder 的輸出傳回
'''
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]  # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

