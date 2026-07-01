# Webull 自動化交易機器人（雙模型融合）

**版本**：V1 基礎版（穩定）/ V2 增強版（動態資金分配）  
**技術棧**：Python 3.10+, Webull OpenAPI, Futu OpenD, XGBoost, TensorFlow (LSTM), Optuna, Pandas, NumPy

---

## 📌 目錄

- [專案簡介](#專案簡介)
- [系統架構與工作原理](#系統架構與工作原理)
- [交易策略詳解](#交易策略詳解)
- [模型訓練方法](#模型訓練方法)
- [安裝與配置（含敏感檔案建立）](#安裝與配置含敏感檔案建立)
- [執行機器人](#執行機器人)
- [自訂參數](#自訂參數)
- [日誌與監控](#日誌與監控)
- [風險提示](#風險提示)
- [授權條款](#授權條款)

---

## 專案簡介

本專案是一個專為美股短線交易設計的自動化機器人，整合了 **機器學習模型** 與 **券商 API**，實現從數據獲取、預測到自動下單的完整流程。

- **核心目標**：利用 15 分鐘 K 線的技術指標，預測未來 5 根 K 線的價格走勢，並在有利時機自動買入／賣出。
- **技術亮點**：
  - XGBoost + LSTM 雙模型融合
  - 70+ 技術指標特徵工程
  - 嚴格的時間控制與風控機制
  - 支援融資買入與動態資金分配（V2）

---

## 系統架構與工作原理

1. 定時循環（每 5 秒）監控美東時間。
2. 判斷階段（正常交易、高信賴度、智能賣出、強制平倉）。
3. 獲取市場數據（即時報價及 300 根 15 分鐘 K 線）。
4. 計算 70+ 技術指標特徵。
5. 載入 XGBoost 與 LSTM 模型，輸出上漲機率（融合權重：0.6 × XGB + 0.4 × LSTM）。
6. 若機率 ≥ 0.70 且持倉未達標，下達市價買單。
7. 若觸及止盈／止損、智能賣出或死叉，下達市價賣單。
8. 在 15:15 強制平倉。
9. 獨立執行緒監控保證金，低於安全值則緊急平倉。

---

## 交易策略詳解

### 雙模型融合預測
- XGBoost 與 LSTM 分別預測上漲機率，最終機率 = 0.6 × XGB + 0.4 × LSTM。
- 預測目標：未來第 5 根 K 線收盤價是否高於當前。

### 技術指標特徵工程（70+ 個）
- 報酬率、移動平均線（SMA）、RSI、ATR、布林帶、MACD、成交量比率、VWAP、K 線形態、波動率、時間特徵、ADX、均線交叉、Williams %R、CCI、OBV、MFI 等。

### 買入條件
- 機率 ≥ 0.70，且不在禁止買入時段，單日虧損未超過上限，持倉未達目標。

### 賣出條件（任一）
- 止盈（≥ 2.0%）
- 止損（≥ 1.0%）
- 智能賣出（14:45 後，若機率 > 60% 標記價格，確認上漲後賣出）
- 死叉（12 SMA 下穿 26 SMA）
- 強制平倉（15:15）

### 時間階段（美東時間）
| 時段 | 階段 | 買入 | 賣出 |
|------|------|------|------|
| 09:30-14:45 | 正常交易 | ✅（機率≥0.70） | ✅（依規則） |
| 14:45-15:05 | 高信賴度 | ✅（機率≥0.75） | ✅（依規則） |
| 14:45-15:15 | 智能賣出 | ❌ | ✅（依規則） |
| ≥15:15 | 強制平倉 | ❌ | ✅（全部） |

### 資金與風險管理
- 總購買力 = 剩餘購買力 + 持倉市值（含融資）。
- 預留 7% 作手續費及緩衝。
- 三隻股票等權重，每隻約佔總購買力 29%，單股上限 35%。
- 單日虧損超過 $5 則停止買入。
- 保證金監控執行緒每 5 秒檢查一次。

---

## 模型訓練方法

### 數據來源
- Futu OpenD（需啟動），K 線類型為 15 分鐘，時間範圍 2020-01-01 至今。

### 特徵與標籤
- 特徵：與交易時使用的 70+ 指標相同。
- 標籤：`(未來第 5 根收盤價 > 當前收盤價)`。

### XGBoost 訓練
- Optuna 超參數最佳化（10 trials），時序交叉驗證（2 折）。
- 儲存為 `models/US.{symbol}_xgb.pkl`。

### LSTM 訓練
- 輸入序列長度 20，雙向 LSTM（64 → 32 units），Dropout(0.3)，Dense(16)，輸出 sigmoid。
- Epochs=30，Batch Size=32，Adam(lr=0.001)，EarlyStopping 與 ReduceLROnPlateau。
- 儲存為 `models/US.{symbol}_lstm.h5`，標準化器為 `US.{symbol}_scaler.pkl`。

### 模型儲存
```
models/
├── US.{symbol}_xgb.pkl
├── US.{symbol}_lstm.h5
├── US.{symbol}_scaler.pkl
└── US.{symbol}_meta.json
```

---

## 安裝與配置（含敏感檔案建立）

**重要**：以下敏感檔案**請自行手動建立，切勿上傳至 GitHub**（已加入 `.gitignore`）。

### 1. 複製專案
```bash
git clone https://github.com/your-username/webull-trading-bot.git
cd webull-trading-bot
```

### 2. 建立虛擬環境
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# 或
venv\Scripts\activate         # Windows
```

### 3. 安裝依賴
```bash
pip install -r requirements.txt
```

主要依賴版本：
```text
webull-openapi-python-sdk>=2.0.10
futu-api>=6.8.0
pandas>=2.0.0
numpy>=1.24.0
xgboost>=1.7.0
tensorflow>=2.12.0
scikit-learn>=1.3.0
optuna>=3.3.0
python-dotenv>=1.0.0
requests>=2.31.0
```

### 4. 取得 Webull API 金鑰
1. 在手機 Webull App 中，進入「帳戶」→「設定」→「OpenAPI 管理」。
2. 點擊「建立應用程式」，取得 `App Key` 與 `App Secret`（Secret 只顯示一次，請立即儲存）。

### 5. 手動建立 `.env` 檔案（根目錄）
建立一個名為 `.env` 的檔案，內容如下（替換為你的實際 Key）：
```env
WEBULL_APP_KEY=你的_App_Key
WEBULL_APP_SECRET=你的_App_Secret
WEBULL_REGION_ID=hk          # 若為美國帳戶請改為 us
WEBULL_ENVIRONMENT=prod      # 正式環境；測試用 uat
```

### 6. 建立 `conf/` 目錄與 Access Token
- 手動建立 `conf/` 目錄：
  ```bash
  mkdir conf
  ```
- 執行機器人，會自動引導手機授權並產生 `conf/token.txt`：
  ```bash
  python trade_v1.py
  ```
  依手機提示點擊「同意」，`conf/token.txt` 將自動產生。

> 若 Token 過期，重複執行上述指令即可重新取得。

---

## 執行機器人

### 訓練模型（首次使用）
需啟動 Futu OpenD（監聽 `127.0.0.1:11111`）：
```bash
python train_models.py
```
訓練完成後可關閉 OpenD。

### 執行 V1 基礎版
```bash
python trade_v1.py
```

### 執行 V2 增強版（小資金動態分配）
```bash
python trade_v2.py
```

---

## 自訂參數

所有參數位於腳本頂部，常見調整如下：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `SYMBOLS` | `['MRVL','INTC','MU']` | 監控股票（不加 `US.`） |
| `TAKE_PROFIT_PCT` | `0.020` | 止盈 2.0% |
| `STOP_LOSS_PCT` | `0.010` | 止損 1.0% |
| `PROB_THRESHOLD_BUY` | `0.70` | 買入門檻 |
| `RESERVE_RATIO` | `0.07` | 預留 7% |
| `MAX_DAILY_LOSS` | `5.00` | 單日最大虧損（美元） |
| `MIN_PER_STOCK_USD`（V2） | `15.0` | 每隻最低投入（低於此則減少持股數） |

時間參數（美東）：
```python
HIGH_CONFIDENCE_START_TIME = dt_time(14, 45)
START_SELL_TIME = dt_time(14, 45)
STOP_BUY_TIME = dt_time(15, 5)
FORCE_SELL_TIME = dt_time(15, 15)
```

---

## 日誌與監控

- 日誌位置：`logs/webull_*.log`（每日切割，保留 90 天）。
- 終端機即時輸出，方便監控。

---

## 風險提示

1. 本工具為輔助交易系統，**不保證獲利**。
2. 使用融資可能放大虧損，請謹慎評估。
3. 建議先以模擬盤或極小資金測試。
4. 開發者不對任何投資損失負責。
5. 請妥善保管 API 金鑰與 Token。

---

## 授權條款

本專案採用 [MIT License](LICENSE)。

---

**如有問題，歡迎提交 Issue 或 Pull Request。**  
**祝交易順利！** 🚀