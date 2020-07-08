from torch import nn
from torch.utils import data
from hw8_seq2seq.config import configurations
from hw8_seq2seq.data import EN2CNDataset
from hw8_seq2seq.main import device
from hw8_seq2seq.utils import build_model, tokens2sentence, computebleu


def test(model, dataloader, loss_function):
  model.eval()
  loss_sum, bleu_score = 0.0, 0.0
  n = 0
  result = []
  for sources, targets in dataloader:
    sources, targets = sources.to(device), targets.to(device)
    batch_size = sources.size(0)
    outputs, preds = model.inference(sources, targets)
    # targets 的第一個 token 是 <BOS> 所以忽略
    outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
    targets = targets[:, 1:].reshape(-1)

    loss = loss_function(outputs, targets)
    loss_sum += loss.item()

    # 將預測結果轉為文字
    targets = targets.view(sources.size(0), -1)
    preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
    sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
    targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
    for source, pred, target in zip(sources, preds, targets):
      result.append((source, pred, target))
    # 計算 Bleu Score
    bleu_score += computebleu(preds, targets)

    n += batch_size

  return loss_sum / len(dataloader), bleu_score / n, result


def test_process(config):
  # 準備測試資料
  test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
  test_loader = data.DataLoader(test_dataset, batch_size=1)
  # 建構模型
  model, optimizer = build_model(config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
  print ("Finish build model")
  loss_function = nn.CrossEntropyLoss(ignore_index=0)
  model.eval()
  # 測試模型
  test_loss, bleu_score, result = test(model, test_loader, loss_function)
  # 儲存結果
  with open('./test_output.txt', 'w') as f:
    for line in result:
      print (line, file=f)

  return test_loss, bleu_score


# 在執行 Test 之前，請先行至 config 設定所要載入的模型位置
if __name__ == '__main__':
    config = configurations()
    print('config:\n', vars(config))
    test_loss, bleu_score = test_process(config)
    print('test loss: {test_loss}, bleu_score: {bleu_score}')

