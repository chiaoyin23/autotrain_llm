# llama_training
llm 模型訓練 by autotrain

### 1. 先確認資料夾有裝git ###
```
git init
git rev-parse --git-dir
```
有回傳 .git 表示成功
### 2. 整理資料 ###
```
python convert_data.py
```
### 3. 載入 autotrain ###
```
pip install autotrain-advanced  
```

### 4. 修改會出問題的檔案 如果不使用 gpu
(路徑舉例： C:\Users\jenny\AppData\Local\Programs\Python\Python312\Lib\site-packages\autotrain)

1. autotrain/cli/run_llm.py
在 init 加上
```
if isinstance(self.args.block_size, str):
            block_size_split = self.args.block_size.strip().split(",")
        else:
            block_size_split = [self.args.block_size]
```
在 run 設定
```
self.args.backend = 'local' 
```
3. autotrain/trainers/cli/train_clm_sft.py
註解字典格式的 ```
    trainer_args = dict(
       args=args,
       model=model,
        callbacks=callbacks,
    ) ```
    
修改為 ```args = SFTConfig(**training_args)```

3. autotrain/trainers/cli/utils.py
   AutoModelForCausalLM.from_pretrained 去掉 quantization_config=bnb_config
   
還有很多跟量化 quantization 有關的參數都要關掉~ 忘記切確位置需要慢慢解

### 5. 將參數寫在finetune.py後執行 ###
```
python finetune.py
```
### 6. 微調後的 model 
存放在剛剛 finetune.py 設定的名稱 (my-autotrain-llm) 中

### 7. chat api

1. 啟動 flask app 測試
```
python chat_method.py
```

2. cmd
```
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"input_text": "你今天下午有空嗎"}'
```
