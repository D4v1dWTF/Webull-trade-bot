Webull 自動化交易機器人（雙模型融合）
本專案提供一個基於 XGBoost + LSTM 雙模型 的美股短線交易機器人，整合 Webull OpenAPI 下單與 Futu OpenD 歷史數據訓練模型。
運行時無需 Futu OpenD，僅在重新訓練模型時才需要。

📌 目錄
系統需求

快速開始

Step 1: 取得 Webull API 金鑰

Step 2: 設定環境變數 (.env)

Step 3: 取得 Access Token (conf/token.txt)

Step 4: 安裝 Python 依賴

Step 5: 訓練模型（首次執行）

Step 6: 執行機器人

自訂參數

常見問題與注意事項

授權條款

系統需求
Python 3.10 或以上

Webull 帳戶（已開通美股交易）

手機 Webull App（用於產生 API 金鑰及授權）

Futu OpenD（僅在訓練模型時需要，下載連結）

快速開始
bash
# 1. 下載專案
git clone https://github.com/your-username/webull-trading-bot.git
cd webull-trading-bot

# 2. 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate      # Linux/Mac
# 或
venv\Scripts\activate         # Windows

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 設定 Webull API 金鑰（見 Step 1~3）

# 5. 訓練模型（需啟動 Futu OpenD）
python train_models.py

# 6. 執行程式
python trade_v1.py   # 基礎版
# 或
python trade_v2.py   # 增強版
Step 1: 取得 Webull API 金鑰
Webull 採用 OpenAPI，需在手機 App 內產生 App Key 及 App Secret。

操作步驟
開啟 Webull App（手機版）。

點擊右下角「帳戶」→ 選擇「設定」。

找到「OpenAPI 管理」或「API 管理」。

點擊「建立應用程式」，輸入應用名稱（可自訂）。

系統會產生 App Key 與 App Secret，請立即複製並妥善保存（App Secret 只會顯示一次）。

確保應用狀態為「已啟用」。

⚠️ 重要：App Secret 等同於密碼，請勿洩漏或上傳至公開倉庫。

Step 2: 設定環境變數 (.env)
在專案根目錄下建立 .env 檔案，填入上一步取得的金鑰。

env
WEBULL_APP_KEY=你的_App_Key
WEBULL_APP_SECRET=你的_App_Secret
WEBULL_REGION_ID=hk          # 若為美國帳戶請改為 us
WEBULL_ENVIRONMENT=prod      # 正式環境，若測試可改 uat
你可以複製 .env.example 範本並重新命名為 .env，再填入真實金鑰。

Step 3: 取得 Access Token (conf/token.txt)
Access Token 用於每次 API 請求的身分驗證，首次執行機器人時會自動產生，但需手機授權。

方法：讓機器人自動產生（推薦）
確保 .env 已正確設定。

執行任意機器人腳本（例如 python trade_v1.py）。

程式會顯示提示：「請在手機 Webull App 中確認 OpenAPI 授權」。

開啟手機 App，會收到一條「OpenAPI 授權請求」通知，點擊「同意」。

授權成功後，conf/token.txt 會自動建立並填入 Access Token。

下次執行時不再需要手機授權（除非 Token 過期）。

若 Token 過期
Token 約 30 天過期，屆時重新執行上述步驟即可。

⚠️ 注意：conf/token.txt 內含敏感資訊，請勿上傳至公開倉庫（已於 .gitignore 忽略）。

Step 4: 安裝 Python 依賴
確保已安裝所有必要套件：

bash
pip install -r requirements.txt
若遇到 TensorFlow 安裝問題，可參考 TensorFlow 官方文件。

Step 5: 訓練模型（首次執行）
模型訓練使用 Futu OpenD 獲取 2020 年至今的 15 分鐘 K 線資料，需先啟動 OpenD。

5.1 啟動 Futu OpenD
下載並安裝 Futu OpenD。

啟動 OpenD，登入你的富途帳戶（需已開通美股行情權限）。

確保 OpenD 監聽於 127.0.0.1:11111（預設）。

5.2 執行訓練腳本
bash
python train_models.py
訓練過程會自動：

獲取 20 隻股票的歷史 K 線（2020年至今）

計算 70+ 技術指標特徵

訓練 XGBoost（Optuna 超參數最佳化）

訓練雙向 LSTM

儲存模型至 models/ 目錄

訓練時間：約 1–2 小時（視電腦性能而定）。
訓練完成後，可關閉 Futu OpenD，日常運行無需 OpenD。

Step 6: 執行機器人
6.1 選擇版本
trade_v1.py：基礎版，固定 3 隻股票等權重，已測試穩定。

trade_v2.py：增強版，動態資金分配（若每隻投入不足 15 美元，自動減少持股數量），適合小資金。

6.2 執行
bash
python trade_v1.py
# 或
python trade_v2.py
6.3 觀察日誌
所有交易記錄會寫入 logs/webull_3stocks.log，同時輸出至終端機。

6.4 停止機器人
按 Ctrl + C 即可安全停止，若在交易時段內會先平倉再退出。

自訂參數
所有主要參數均集中在腳本頂部，可依自身需求調整。

python
# 核心參數（以 trade_v1.py 為例）
SYMBOLS = ['MRVL', 'INTC', 'MU']        # 監控股票（Webull 代碼，不加 US.）
TAKE_PROFIT_PCT = 0.008                 # 止盈 0.8%
STOP_LOSS_PCT = 0.004                   # 止損 0.4%
PROB_THRESHOLD_BUY = 0.70               # 買入門檻（機率 ≥70%）
RESERVE_RATIO = 0.07                    # 預留資金 7%（手續費/風控）
MAX_DAILY_LOSS = 1.00                   # 單日最大虧損（美元）

# V2 額外參數
MIN_PER_STOCK_USD = 15.0                # 每隻股票最低投入金額
時間參數（美東時間）
python
HIGH_CONFIDENCE_START_TIME = dt_time(14, 45)   # 高信賴度開始
START_SELL_TIME = dt_time(14, 45)              # 智能賣出開始
STOP_BUY_TIME = dt_time(15, 5)                 # 禁止買入
FORCE_SELL_TIME = dt_time(15, 15)              # 強制平倉
所有時間均為美東時間，機器人會自動對應。

常見問題與注意事項
Q1: 首次執行出現「UNAUTHORIZED」錯誤？
A: 請檢查手機 Webull App 是否收到授權通知並點擊「同意」。若未收到，可嘗試重新執行腳本，或在 App 內重新產生 App Key/Secret。

Q2: 執行機器人時需要 Futu OpenD 嗎？
A: 不需要。OpenD 僅用於訓練模型。日常運行僅依賴 Webull API。

Q3: 可以修改監控的股票嗎？
A: 可以，直接修改 SYMBOLS 列表即可（請確保有對應的訓練模型）。

Q4: 模型檔案遺失或想要新增股票？
A: 重新執行 train_models.py（需啟動 Futu OpenD）即可重新訓練所有股票。

Q5: 如何調整止盈止損比例？
A: 修改 TAKE_PROFIT_PCT 及 STOP_LOSS_PCT，例如 0.02 表示 2%。

Q6: 小資金適合使用哪個版本？
A: 建議使用 V2，因為它會動態調整持股數量，確保每隻投入至少 15 美元（可調整）。

Q7: 機器人會自動平倉嗎？
A: 會，在強制平倉時間（預設 15:15）到達時，會清空所有持倉。

Q8: 單日虧損超過限制會怎樣？
A: 機器人會停止新的買入，但已持倉的股票仍會依止損/止盈條件賣出，避免繼續虧損。

上傳至 GitHub 的提醒
務必確認已忽略敏感檔案（.gitignore 已包含 conf/token.txt、.env、logs/、models/ 等）。

若想提供預訓練模型，請使用 Git LFS，或讓使用者自行訓練（推薦）。

授權條款
本專案採用 MIT License，使用前請詳閱。

