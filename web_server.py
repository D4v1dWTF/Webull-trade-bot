#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webull 交易機器人 - 完整版（修正 DTypePolicy 載入錯誤）
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
import time
import pickle
import json
import logging
import math
import uuid
import hashlib
import hmac
import base64
import threading
import requests
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request, render_template_string
from dotenv import load_dotenv

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient
from webull.data.data_client import DataClient
from webull.data.common.category import Category
from webull.data.common.timespan import Timespan

load_dotenv()

# ================== 全局配置 ==================
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'AMD', 'TSLA', 'PLTR', 'DELL',
    'INTC', 'BA', 'XOM', 'COIN', 'AVGO',
    'ISRG', 'OUST', 'IBM', 'MU', 'MRVL'
]

MAX_SINGLE_POSITION_PCT = 1.0
HARD_STOP_LOSS_PCT = 0.05
ATR_STOP_MULTIPLIER = 2.0
ATR_PERIOD = 14
BUY_AMOUNT_PCT = 0.20
PROB_THRESHOLD_BUY = 0.65
HIGH_CONFIDENCE_THRESHOLD = 0.75
PROB_THRESHOLD_SELL = 0.60
KLINE_INTERVAL = Timespan.M15.name
CHECK_INTERVAL_SEC = 5
SHORT_SUMMARY_INTERVAL_SEC = 10 * 60
LONG_SUMMARY_INTERVAL_SEC = 30 * 60
HIGH_CONFIDENCE_START_TIME = dt_time(14, 45)
STOP_BUY_TIME = dt_time(15, 0)
START_SELL_TIME = dt_time(15, 0)
FORCE_SELL_TIME = dt_time(15, 15)
RESERVED_FEE_PER_TRADE = 0.02

FEATURES = [
    'return_1d', 'return_5d', 'return_10d', 'return_20d',
    'sma5', 'sma10', 'sma20', 'sma50', 'sma200',
    'sma5_sma20_ratio', 'sma10_sma50_ratio',
    'close_sma20_ratio', 'close_sma50_ratio',
    'rsi_7', 'rsi_14', 'rsi_21',
    'atr', 'atr_pct',
    'bb_width', 'bb_position',
    'macd', 'macd_signal', 'macd_diff',
    'volume_ratio', 'volume_ma_ratio',
    'vwap', 'vwap_ratio',
    'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
    'high_low_ratio', 'open_close_ratio',
    'volatility_10', 'volatility_20',
    'hour_sin', 'hour_cos',
    'adx', 'plus_di', 'minus_di',
    'golden_cross_5_20', 'death_cross_5_20',
    'golden_cross_20_50', 'death_cross_20_50',
    'williams_r', 'cci',
    'obv', 'obv_ratio', 'mfi'
]

# ================== 日誌設定 ==================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_handler = TimedRotatingFileHandler(
    filename=os.path.join(LOG_DIR, "webull_final.log"),
    when="midnight", interval=1, backupCount=90, encoding="utf-8"
)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.addHandler(console)

# 前端日誌列表
ui_log_messages = []

# 警告轉譯字典
WARNING_TRANSLATION = {
    'model missing': '模型檔案缺失',
    'cannot load': '無法載入模型',
    'token expired': '權杖已過期，請重新授權',
    'connection failed': '連線失敗，請檢查網路',
    'market closed': '美股休市中，機器人暫停交易',
    'insufficient funds': '可用資金不足',
    'order failed': '下單失敗',
    'buy amount too small': '買入金額低於5美元',
    'position limit exceeded': '持倉比例超過限制',
}

def translate_warning(msg: str) -> str:
    msg_lower = msg.lower()
    for en, zh in WARNING_TRANSLATION.items():
        if en in msg_lower:
            return zh
    return f"⚠️ {msg}"

def add_ui_log(level: str, msg: str):
    """前端日誌過濾：只排除資產查詢等雜訊，保留所有狀態、交易、錯誤訊息"""
    exclude_keywords = ['美元淨資產', '可用現金']
    
    if any(k in msg for k in exclude_keywords):
        return
    
    if level == 'WARNING':
        msg = translate_warning(msg)
    elif level == 'ERROR':
        msg = f"❌ 錯誤：{msg}"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ui_log_messages.append({"time": timestamp, "level": level, "msg": msg})
    while len(ui_log_messages) > 500:
        ui_log_messages.pop(0)

# 覆蓋 logger
original_info = logger.info
original_error = logger.error
original_warning = logger.warning
original_exception = logger.exception

def info_with_ui(msg, *args, **kwargs):
    original_info(msg, *args, **kwargs)
    add_ui_log("INFO", msg)
def error_with_ui(msg, *args, **kwargs):
    original_error(msg, *args, **kwargs)
    add_ui_log("ERROR", msg)
def warning_with_ui(msg, *args, **kwargs):
    original_warning(msg, *args, **kwargs)
    add_ui_log("WARNING", msg)
def exception_with_ui(msg, *args, **kwargs):
    original_exception(msg, *args, **kwargs)
    add_ui_log("ERROR", f"嚴重異常：{msg}")

logger.info = info_with_ui
logger.error = error_with_ui
logger.warning = warning_with_ui
logger.exception = exception_with_ui

# ================== 全局狀態 ==================
stop_event = threading.Event()
bot_thread = None
thread_lock = threading.Lock()
trade_client = None
data_client = None
models = {}
webull_username = "未知"
account_id = None

# ================== 時間輔助 ==================
def get_current_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))

def is_us_market_open() -> bool:
    now_et = get_current_et()
    if now_et.weekday() >= 5:
        return False
    open_t = dt_time(9, 30)
    close_t = dt_time(16, 0)
    return open_t <= now_et.time() <= close_t

def is_buy_allowed(now_time: dt_time, confidence: float) -> bool:
    if now_time >= STOP_BUY_TIME:
        return False
    if now_time >= HIGH_CONFIDENCE_START_TIME:
        return confidence >= HIGH_CONFIDENCE_THRESHOLD
    return confidence >= PROB_THRESHOLD_BUY

def is_sell_allowed(now_time: dt_time) -> bool:
    return now_time >= START_SELL_TIME

def is_force_sell_time(now_time: dt_time) -> bool:
    return now_time >= FORCE_SELL_TIME

# ================== Webull 客戶端初始化 ==================
def init_webull_clients() -> Tuple[TradeClient, DataClient, str, str]:
    global webull_username, account_id
    logger.info("正在初始化 Webull 客戶端...")
    app_key = os.getenv("WEBULL_APP_KEY")
    app_secret = os.getenv("WEBULL_APP_SECRET")
    region_id = os.getenv("WEBULL_REGION_ID", "hk")
    env = os.getenv("WEBULL_ENVIRONMENT", "prod")
    if not app_key or not app_secret:
        raise ValueError("缺少 WEBULL_APP_KEY 或 WEBULL_APP_SECRET")
    api_client = ApiClient(app_key, app_secret, region_id)
    if env.lower() == "prod":
        api_client.add_endpoint(region_id, "api.webull.hk")
    else:
        api_client.add_endpoint(region_id, "us-openapi-alb.uat.webullbroker.com")
    trade_client = TradeClient(api_client)
    data_client = DataClient(api_client)
    
    resp = trade_client.account_v2.get_account_list()
    accounts = resp.json()
    if not accounts:
        raise ValueError("無法獲取帳戶列表")
    account_id = accounts[0].get("account_id")
    if not account_id:
        raise ValueError("account_id 不存在")
    
    try:
        profile_resp = trade_client.account_v2.get_account_profile(account_id)
        if profile_resp.status_code == 200:
            profile_data = profile_resp.json()
            webull_username = profile_data.get("account_number", "未知")
            logger.info(f"Webull 連線成功，帳戶：{webull_username}")
        else:
            webull_username = "未知"
            logger.warning("無法取得帳戶名稱")
    except Exception as e:
        webull_username = "未知"
        logger.warning(f"取得帳戶名稱失敗: {e}")
    
    return trade_client, data_client, webull_username, account_id

# ================== 行情數據 ==================
def get_market_data(data_client: DataClient, symbol: str, count=300) -> pd.DataFrame:
    try:
        full_symbol = symbol
        res = data_client.market_data.get_history_bar(
            full_symbol,
            Category.US_STOCK.name,
            KLINE_INTERVAL,
            count
        )
        if res.status_code == 200:
            data = res.json()
            bars = None
            if isinstance(data, list):
                for item in data:
                    if item.get("symbol") == full_symbol and "result" in item:
                        bars = item["result"]
                        break
                if bars is None:
                    bars = data
            elif isinstance(data, dict):
                bars = data.get("result", [])
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            if 'time' in df.columns:
                df['time_key'] = pd.to_datetime(df['time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.sort_values('time_key').reset_index(drop=True)
        else:
            logger.warning(f"獲取 {symbol} K線失敗 HTTP {res.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"獲取 {symbol} K線異常: {e}")
        return pd.DataFrame()

def get_real_time_price(data_client: DataClient, symbol: str) -> Optional[float]:
    try:
        full_symbol = symbol
        res = data_client.market_data.get_snapshot(full_symbol, Category.US_STOCK.name)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                return float(data[0].get("price", 0))
        return None
    except Exception:
        return None

# ================== 帳戶與持倉 ==================
def get_account_balance(trade_client: TradeClient) -> Tuple[float, float]:
    try:
        resp = trade_client.account_v2.get_account_list()
        accounts = resp.json()
        if not accounts:
            return 0, 0
        aid = accounts[0].get("account_id")
        bal_resp = trade_client.account_v2.get_account_balance(aid)
        bal_data = bal_resp.json()
        usd_cash = 0.0
        usd_net = 0.0
        if "account_currency_assets" in bal_data:
            for asset in bal_data["account_currency_assets"]:
                if asset.get("currency") == "USD":
                    usd_cash = float(asset.get("cash_balance", 0))
                    usd_net = float(asset.get("net_liquidation_value", 0))
                    break
        available = max(usd_cash - RESERVED_FEE_PER_TRADE, 0)
        logger.debug(f"美元淨資產: ${usd_net:,.2f}, 可用現金: ${available:,.2f}")
        return usd_net, available
    except Exception as e:
        logger.error(f"獲取帳戶餘額失敗: {e}")
        return 0, 0

def get_positions(trade_client: TradeClient) -> pd.DataFrame:
    try:
        resp = trade_client.account_v2.get_account_list()
        accounts = resp.json()
        if not accounts:
            return pd.DataFrame()
        aid = accounts[0].get("account_id")
        pos_resp = trade_client.account_v2.get_account_position(aid)
        positions = pos_resp.json()
        if not positions:
            return pd.DataFrame()
        rows = []
        for pos in positions:
            rows.append({
                'symbol': pos.get('symbol'),
                'qty': float(pos.get('position', 0)),
                'cost_price': float(pos.get('costPrice', 0)),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.debug(f"獲取持倉時臨時錯誤: {e}")
        return pd.DataFrame()

# ================== 下單函數 ==================
def place_buy_order(trade_client: TradeClient, symbol: str, amount_usd: float) -> Tuple[bool, str, Optional[str]]:
    if amount_usd < 5.0:
        return False, f"買入金額 ${amount_usd:.2f} < 5 美元", None
    _, available = get_account_balance(trade_client)
    if available < amount_usd:
        return False, "可用美元資金不足", None

    try:
        token_file = os.path.join("conf", "token.txt")
        with open(token_file, 'r') as f:
            access_token = f.readline().strip()

        order_body = {
            "account_id": account_id,
            "new_orders": [
                {
                    "client_order_id": uuid.uuid4().hex,
                    "combo_type": "NORMAL",
                    "symbol": symbol,
                    "instrument_type": "EQUITY",
                    "market": "US",
                    "order_type": "MARKET",
                    "side": "BUY",
                    "entrust_type": "AMOUNT",
                    "total_cash_amount": str(amount_usd),
                    "time_in_force": "DAY",
                    "support_trading_session": "CORE"
                }
            ]
        }

        uri = "/openapi/trade/order/place"
        app_key = os.getenv("WEBULL_APP_KEY")
        app_secret = os.getenv("WEBULL_APP_SECRET")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nonce = uuid.uuid4().hex
        body_json = json.dumps(order_body, separators=(',', ':'))

        params = {}
        sig_headers = {
            'x-app-key': app_key,
            'x-signature-algorithm': 'HMAC-SHA1',
            'x-signature-version': '1.0',
            'x-signature-nonce': nonce,
            'x-timestamp': timestamp,
            'host': "api.webull.hk"
        }
        params.update(sig_headers)
        sorted_items = sorted(params.items())
        param_string = '&'.join([f"{k}={v}" for k, v in sorted_items])
        body_md5 = hashlib.md5(body_json.encode()).hexdigest().upper()
        sign_str = f"{uri}&{param_string}&{body_md5}"
        encoded = requests.utils.quote(sign_str, safe='')
        secret_key = f"{app_secret}&"
        signature = hmac.new(secret_key.encode(), encoded.encode(), hashlib.sha1).digest()
        signature_b64 = base64.b64encode(signature).decode()

        headers = {
            "x-app-key": app_key,
            "x-timestamp": timestamp,
            "x-signature-version": "1.0",
            "x-signature-algorithm": "HMAC-SHA1",
            "x-signature-nonce": nonce,
            "x-signature": signature_b64,
            "x-version": "v2",
            "x-access-token": access_token,
            "Content-Type": "application/json"
        }

        url = f"https://api.webull.hk{uri}"
        response = requests.post(url, data=body_json, headers=headers)
        if response.status_code == 200:
            result = response.json()
            order_id = result.get('order_id')
            logger.info(f"✅ 買入成功: ${amount_usd:.2f} -> {symbol} (訂單 {order_id})")
            return True, "成功", order_id
        else:
            logger.error(f"買入失敗 HTTP {response.status_code}: {response.text}")
            return False, f"HTTP {response.status_code}", None
    except Exception as e:
        logger.error(f"買入下單異常: {e}")
        return False, str(e), None

def place_sell_order(trade_client: TradeClient, symbol: str, qty: float) -> Tuple[bool, str, Optional[str]]:
    if qty <= 0:
        return False, "數量無效", None

    try:
        token_file = os.path.join("conf", "token.txt")
        with open(token_file, 'r') as f:
            access_token = f.readline().strip()

        order_body = {
            "account_id": account_id,
            "new_orders": [
                {
                    "client_order_id": uuid.uuid4().hex,
                    "combo_type": "NORMAL",
                    "symbol": symbol,
                    "instrument_type": "EQUITY",
                    "market": "US",
                    "order_type": "MARKET",
                    "side": "SELL",
                    "entrust_type": "QTY",
                    "quantity": str(qty),
                    "time_in_force": "DAY",
                    "support_trading_session": "CORE"
                }
            ]
        }

        uri = "/openapi/trade/order/place"
        app_key = os.getenv("WEBULL_APP_KEY")
        app_secret = os.getenv("WEBULL_APP_SECRET")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nonce = uuid.uuid4().hex
        body_json = json.dumps(order_body, separators=(',', ':'))

        params = {}
        sig_headers = {
            'x-app-key': app_key,
            'x-signature-algorithm': 'HMAC-SHA1',
            'x-signature-version': '1.0',
            'x-signature-nonce': nonce,
            'x-timestamp': timestamp,
            'host': "api.webull.hk"
        }
        params.update(sig_headers)
        sorted_items = sorted(params.items())
        param_string = '&'.join([f"{k}={v}" for k, v in sorted_items])
        body_md5 = hashlib.md5(body_json.encode()).hexdigest().upper()
        sign_str = f"{uri}&{param_string}&{body_md5}"
        encoded = requests.utils.quote(sign_str, safe='')
        secret_key = f"{app_secret}&"
        signature = hmac.new(secret_key.encode(), encoded.encode(), hashlib.sha1).digest()
        signature_b64 = base64.b64encode(signature).decode()

        headers = {
            "x-app-key": app_key,
            "x-timestamp": timestamp,
            "x-signature-version": "1.0",
            "x-signature-algorithm": "HMAC-SHA1",
            "x-signature-nonce": nonce,
            "x-signature": signature_b64,
            "x-version": "v2",
            "x-access-token": access_token,
            "Content-Type": "application/json"
        }

        url = f"https://api.webull.hk{uri}"
        response = requests.post(url, data=body_json, headers=headers)
        if response.status_code == 200:
            result = response.json()
            order_id = result.get('order_id')
            logger.info(f"✅ 賣出成功: {qty:.4f} 股 {symbol} (訂單 {order_id})")
            return True, "成功", order_id
        else:
            logger.error(f"賣出失敗 HTTP {response.status_code}: {response.text}")
            return False, f"HTTP {response.status_code}", None
    except Exception as e:
        logger.error(f"賣出下單異常: {e}")
        return False, str(e), None

# ================== 技術指標 ==================
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['return_20d'] = df['close'].pct_change(20)
    for period in [5, 10, 20, 50, 200]:
        df[f'sma{period}'] = df['close'].rolling(period).mean()
    df['sma5_sma20_ratio'] = df['sma5'] / df['sma20'] - 1
    df['sma10_sma50_ratio'] = df['sma10'] / df['sma50'] - 1
    df['close_sma20_ratio'] = df['close'] / df['sma20'] - 1
    df['close_sma50_ratio'] = df['close'] / df['sma50'] - 1
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD).mean()
    df['atr_pct'] = df['atr'] / df['close']
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (2 * df['bb_std']) / df['bb_mid']
    df['bb_position'] = (df['close'] - df['bb_mid']) / (2 * df['bb_std'] + 1e-8)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    df['volume_ma_ratio'] = df['volume_ma'] / (df['volume_ma'].shift(1) + 1e-8) - 1
    df['vwap'] = (df['volume'] * df['close']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)
    df['vwap_ratio'] = df['close'] / df['vwap'] - 1
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['upper_shadow_ratio'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
    df['lower_shadow_ratio'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
    df['volatility_10'] = df['return_1d'].rolling(10).std()
    df['volatility_20'] = df['return_1d'].rolling(20).std()
    df['time_key'] = pd.to_datetime(df['time_key'])
    df['hour'] = df['time_key'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['atr_adx'] = tr.rolling(14).mean()
    df['plus_dm'] = ((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low'])) * (df['high'] - df['high'].shift(1)).clip(lower=0)
    df['minus_dm'] = ((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1))) * (df['low'].shift(1) - df['low']).clip(lower=0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / (df['atr_adx'] + 1e-8))
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / (df['atr_adx'] + 1e-8))
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-8)
    df['adx'] = df['dx'].rolling(14).mean()
    df['golden_cross_5_20'] = ((df['sma5'] > df['sma20']) & (df['sma5'].shift(1) <= df['sma20'].shift(1))).astype(int)
    df['death_cross_5_20'] = ((df['sma5'] < df['sma20']) & (df['sma5'].shift(1) >= df['sma20'].shift(1))).astype(int)
    df['golden_cross_20_50'] = ((df['sma20'] > df['sma50']) & (df['sma20'].shift(1) <= df['sma50'].shift(1))).astype(int)
    df['death_cross_20_50'] = ((df['sma20'] < df['sma50']) & (df['sma20'].shift(1) >= df['sma50'].shift(1))).astype(int)
    df['williams_r'] = -100 * (df['high'].rolling(14).max() - df['close']) / (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-8)
    df['cci'] = (df['close'] - df['close'].rolling(20).mean()) / (0.015 * df['close'].rolling(20).std() + 1e-8)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_ratio'] = df['obv'] / (df['obv'].abs().max() + 1e-8)
    tp = (df['high'] + df['low'] + df['close']) / 3
    money_flow = df['volume'] * tp
    pos_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(14).sum()
    neg_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(14).sum()
    df['mfi'] = 100 - 100 / (1 + pos_flow / (neg_flow + 1e-8))
    df.drop(columns=['hour', 'atr_adx', 'dx'], errors='ignore', inplace=True)
    return df.dropna().reset_index(drop=True)

# ================== 模型載入（相容舊版） ==================
class CompatibleInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

class DummyDTypePolicy:
    """模擬舊版 dtype policy，提供必要的 name 屬性"""
    def __init__(self, *args, **kwargs):
        pass
    @property
    def name(self):
        return "float32"

def load_models() -> Dict[str, Tuple]:
    models_dict = {}
    logger.info("開始載入模型...")
    for symbol in SYMBOLS:
        xgb_path = f"models/US.{symbol}_xgb.pkl"
        lstm_path = f"models/US.{symbol}_lstm.h5"
        scaler_path = f"models/US.{symbol}_scaler.pkl"
        if not (os.path.exists(xgb_path) and os.path.exists(lstm_path) and os.path.exists(scaler_path)):
            logger.warning(f"模型缺失: {symbol}，跳過")
            continue
        with open(xgb_path, 'rb') as f:
            xgb_model = pickle.load(f)
        
        lstm_model = None
        try:
            lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
        except Exception as e:
            logger.warning(f"載入 {symbol} LSTM 遇到錯誤，嘗試使用自定義物件")
            try:
                custom_objects = {
                    'InputLayer': CompatibleInputLayer,
                    'DTypePolicy': DummyDTypePolicy,
                }
                lstm_model = tf.keras.models.load_model(
                    lstm_path,
                    compile=False,
                    custom_objects=custom_objects
                )
            except Exception as e2:
                logger.error(f"仍無法載入 {symbol} LSTM: {e2}，跳過此股票")
                continue
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        models_dict[symbol] = (xgb_model, lstm_model, scaler)
        logger.info(f"成功載入模型: {symbol}")
    logger.info(f"模型載入完成，共 {len(models_dict)} 支股票")
    return models_dict

def predict_probability(data_client: DataClient, symbol: str, xgb_model, lstm_model, scaler) -> float:
    df = get_market_data(data_client, symbol, 300)
    if df.empty or len(df) < 21:
        return 0.5
    df_feat = compute_technical_features(df)
    if df_feat.empty:
        return 0.5
    X = df_feat[FEATURES].iloc[-1:].values
    xgb_prob = xgb_model.predict_proba(X)[0][1]
    if len(df_feat) >= 21:
        last_20 = df_feat[FEATURES].iloc[-20:].values
        last_20_scaled = scaler.transform(last_20)
        X_lstm = last_20_scaled.reshape(1, 20, -1)
        lstm_prob = lstm_model.predict(X_lstm, verbose=0)[0][0]
    else:
        lstm_prob = 0.5
    return 0.6 * xgb_prob + 0.4 * lstm_prob

# ================== 持倉追蹤器 ==================
class PositionTracker:
    def __init__(self):
        self.positions = {}
    def add(self, symbol: str, entry_price: float, entry_atr: float):
        self.positions[symbol] = {'entry_price': entry_price, 'entry_atr': entry_atr, 'highest_price': entry_price}
    def remove(self, symbol: str):
        self.positions.pop(symbol, None)
    def update_high(self, symbol: str, current_price: float):
        if symbol in self.positions and current_price > self.positions[symbol]['highest_price']:
            self.positions[symbol]['highest_price'] = current_price
    def get_info(self, symbol: str):
        return self.positions.get(symbol)

def check_exit_conditions(tracker: PositionTracker, trade_client: TradeClient, data_client: DataClient,
                          symbol: str, current_price: float, df: pd.DataFrame, sell_prob: float, now_time: dt_time) -> Tuple[bool, str, Optional[float]]:
    info = tracker.get_info(symbol)
    if info is None:
        return False, "無持倉", None
    entry_price = info['entry_price']
    entry_atr = info.get('entry_atr', 0)
    highest_price = info['highest_price']
    profit_loss_pct = (current_price - entry_price) / entry_price

    pos_df = get_positions(trade_client)
    row = pos_df[pos_df['symbol'] == symbol]
    if row.empty:
        return False, "無持倉", None
    qty = row.iloc[0]['qty']

    if profit_loss_pct <= -HARD_STOP_LOSS_PCT:
        return True, f"硬止損 (虧損 {profit_loss_pct*100:.2f}%)", qty
    if entry_atr > 0 and highest_price > entry_price:
        stop_price = highest_price - ATR_STOP_MULTIPLIER * entry_atr
        if current_price <= stop_price:
            return True, "ATR移動止損", qty
    if is_sell_allowed(now_time) and sell_prob >= PROB_THRESHOLD_SELL:
        return True, f"智能賣出 (預測下跌 {sell_prob*100:.1f}%)", qty
    if sell_prob >= PROB_THRESHOLD_SELL and profit_loss_pct > 0:
        return True, f"ML止盈 (預測下跌 {sell_prob*100:.1f}%)", qty
    if len(df) >= 50:
        sma_short = df['close'].rolling(12).mean().iloc[-1]
        sma_long = df['close'].rolling(26).mean().iloc[-1]
        prev_short = df['close'].rolling(12).mean().iloc[-2]
        prev_long = df['close'].rolling(26).mean().iloc[-2]
        if prev_short >= prev_long and sma_short < sma_long:
            return True, "死叉賣出", qty
    tracker.update_high(symbol, current_price)
    return False, None, None

def force_close_all(trade_client: TradeClient):
    pos_df = get_positions(trade_client)
    for _, row in pos_df.iterrows():
        symbol = row['symbol']
        qty = row['qty']
        logger.info(f"強制平倉: {symbol} {qty:.4f} 股")
        place_sell_order(trade_client, symbol, qty)

# ================== 機器人主循環（含狀態提示） ==================
def bot_loop():
    global stop_event, trade_client, data_client, models
    logger.info("機器人執行緒啟動，開始交易循環")
    tracker = PositionTracker()
    
    last_waiting_msg_time = 0
    last_high_conf_mode = False
    last_stop_buy_mode = False
    last_start_sell_mode = False
    last_force_sell_mode = False
    
    try:
        pos_df = get_positions(trade_client)
        for _, row in pos_df.iterrows():
            symbol = row['symbol']
            if symbol in models:
                cost = row['cost_price']
                df = get_market_data(data_client, symbol, 200)
                atr_val = df['atr'].iloc[-1] if not df.empty and 'atr' in df.columns else 0
                tracker.add(symbol, cost, atr_val)
                logger.info(f"同步現有持倉 {symbol} 成本 {cost:.2f}")
    except Exception as e:
        logger.exception(f"同步持倉失敗: {e}")
        stop_event.set()
        return

    try:
        while not stop_event.is_set():
            try:
                now_et = get_current_et()
                now_time = now_et.time()
                
                high_conf_active = now_time >= HIGH_CONFIDENCE_START_TIME
                stop_buy_active = now_time >= STOP_BUY_TIME
                start_sell_active = now_time >= START_SELL_TIME
                force_sell_active = now_time >= FORCE_SELL_TIME
                
                if high_conf_active and not last_high_conf_mode:
                    logger.info("🔔 已進入高信賴度買入模式 (14:45後，買入需 ≥75% 信心)")
                elif not high_conf_active and last_high_conf_mode and now_time.hour < 14:
                    logger.info("🔔 已離開高信賴度買入模式")
                last_high_conf_mode = high_conf_active
                
                if stop_buy_active and not last_stop_buy_mode:
                    logger.info("⛔ 已到達停止買入時間 15:00，禁止新買入")
                last_stop_buy_mode = stop_buy_active
                
                if start_sell_active and not last_start_sell_mode:
                    logger.info("💰 已到達開始賣出時間 15:00，啟用智能賣出")
                last_start_sell_mode = start_sell_active
                
                if force_sell_active and not last_force_sell_mode:
                    logger.info("⚠️ 已到達強制平倉時間 15:15，將平倉所有持倉")
                last_force_sell_mode = force_sell_active
                
                market_open = is_us_market_open()
                if not market_open:
                    if time.time() - last_waiting_msg_time > 60:
                        logger.info("⏳ 美股尚未開盤（美東時間 9:30-16:00），機器人等待中...")
                        last_waiting_msg_time = time.time()
                    for _ in range(CHECK_INTERVAL_SEC):
                        if stop_event.is_set():
                            break
                        time.sleep(1)
                    continue
                else:
                    last_waiting_msg_time = 0
                    if not hasattr(bot_loop, "_market_open_just_entered"):
                        logger.info("📈 美股已開盤，機器人開始掃描標的")
                        bot_loop._market_open_just_entered = True
                
                if is_force_sell_time(now_time):
                    logger.info("到達強制平倉時間 15:15，平倉所有持倉")
                    force_close_all(trade_client)
                    stop_event.set()
                    break

                total, available = get_account_balance(trade_client)
                pos_df = get_positions(trade_client)
                current_holdings = {row['symbol']: row['qty'] for _, row in pos_df.iterrows()}

                for symbol in SYMBOLS:
                    if stop_event.is_set():
                        break
                    if symbol not in models:
                        continue
                    logger.info(f"分析 {symbol}")
                    df = get_market_data(data_client, symbol, 300)
                    if df.empty:
                        continue
                    current_price = df['close'].iloc[-1]
                    xgb_model, lstm_model, scaler = models[symbol]
                    buy_prob = predict_probability(data_client, symbol, xgb_model, lstm_model, scaler)
                    sell_prob = 1 - buy_prob
                    logger.info(f"  {symbol} 上漲概率: {buy_prob:.3f}")

                    holding_qty = current_holdings.get(symbol, 0)

                    if holding_qty > 0:
                        should_sell, reason, qty = check_exit_conditions(
                            tracker, trade_client, data_client, symbol, current_price, df, sell_prob, now_time
                        )
                        if should_sell:
                            logger.info(f"賣出 {symbol}: {reason}")
                            success, msg, oid = place_sell_order(trade_client, symbol, qty)
                            if success:
                                tracker.remove(symbol)
                                time.sleep(0.5)
                            else:
                                logger.error(f"賣出失敗: {msg}")
                            continue

                    if holding_qty == 0:
                        if is_buy_allowed(now_time, buy_prob):
                            if available <= 0:
                                logger.info(f"  {symbol} 買入信號，但美元現金為0")
                                continue
                            buy_amount = available * BUY_AMOUNT_PCT
                            if buy_amount < 5.0:
                                logger.info(f"  {symbol} 買入信號，但資金不足（需要至少 $5）")
                                continue
                            effective_total = total if total > 0 else available
                            new_ratio = buy_amount / (effective_total + buy_amount)
                            if new_ratio > MAX_SINGLE_POSITION_PCT:
                                logger.info(f"  {symbol} 買入會導致倉位超限 {new_ratio:.1%} > {MAX_SINGLE_POSITION_PCT:.0%}")
                                continue
                            logger.info(f"🎯 買入信號 {symbol} 價格 {current_price:.2f} 概率 {buy_prob:.3f} 金額 ${buy_amount:.2f}")
                            success, msg, oid = place_buy_order(trade_client, symbol, buy_amount)
                            if success:
                                atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                                tracker.add(symbol, current_price, atr_val)
                                available -= buy_amount
                            else:
                                logger.error(f"買入失敗: {msg}")
                    time.sleep(0.8)

                logger.info("完成一輪掃描，等待5秒")
                for _ in range(CHECK_INTERVAL_SEC):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
            except Exception as inner_e:
                logger.exception(f"交易循環內部發生嚴重錯誤: {inner_e}")
                logger.error("因嚴重錯誤，機器人將自動停止以避免潛在虧損")
                stop_event.set()
                break
    except Exception as e:
        logger.exception(f"機器人主循環發生嚴重錯誤: {e}")
        logger.error("因嚴重錯誤，機器人已自動停止")
        stop_event.set()
    finally:
        logger.info("機器人執行緒已停止")

# ================== Flask Web 服務（內嵌 HTML） ==================
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>TradeBoy</title>
    <style>
        * { box-sizing: border-box; user-select: none; }
        body {
            background: #8b8b8b;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: 'Courier New', 'VT323', monospace;
            margin: 0;
            padding: 20px;
        }
        .gameboy {
            background: #b5b5b5;
            border-radius: 20px;
            box-shadow: 0 10px 0 #5a5a5a;
            padding: 20px 20px 30px;
            max-width: 620px;
            width: 100%;
        }
        .screen {
            background: #9bbc0f;
            border: 5px solid #306230;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 15px;
            font-family: monospace;
            color: #0f380f;
            font-weight: bold;
            font-size: 13px;
            box-shadow: inset 0 0 5px #306230;
        }
        .balance {
            background: #0f380f;
            color: #9bbc0f;
            padding: 8px;
            border-radius: 8px;
            margin-bottom: 10px;
            text-align: center;
            font-size: 15px;
        }
        .positions {
            background: #e0f0d0;
            padding: 8px;
            border-radius: 8px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .positions table {
            width: 100%;
            border-collapse: collapse;
        }
        .positions th, .positions td {
            text-align: left;
            padding: 4px 2px;
            border-bottom: 1px solid #7c8c5e;
        }
        .log-area {
            background: #0f380f;
            color: #9bbc0f;
            padding: 6px;
            border-radius: 8px;
            height: 150px;
            overflow-y: auto;
            font-size: 11px;
            font-family: monospace;
            margin-bottom: 15px;
            text-align: left;
        }
        .log-area p {
            margin: 2px 0;
            border-bottom: 1px dotted #306230;
        }
        .controls {
            display: flex;
            justify-content: space-around;
            gap: 15px;
            margin-top: 5px;
        }
        button {
            background: #4b4b4b;
            border: none;
            color: white;
            font-family: inherit;
            font-size: 18px;
            font-weight: bold;
            padding: 12px 16px;
            border-radius: 50px;
            box-shadow: 0 5px 0 #2a2a2a;
            cursor: pointer;
            transition: 0.05s linear;
            flex: 1;
            letter-spacing: 2px;
        }
        button:active {
            transform: translateY(2px);
            box-shadow: 0 2px 0 #2a2a2a;
        }
        button.start { background: #3a7a3a; box-shadow: 0 5px 0 #1e4a1e; }
        button.stop { background: #aa4a4a; box-shadow: 0 5px 0 #6a2a2a; }
        button.force { background: #aa6a2a; box-shadow: 0 5px 0 #6a3a0a; }
        .status {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #333;
            margin-right: 6px;
        }
        .status.running { background: #2ecc2e; box-shadow: 0 0 5px #2ecc2e; }
        .status.stopped { background: #e74c3c; }
        .refresh-note {
            text-align: center;
            font-size: 10px;
            margin-top: 10px;
            color: #306230;
        }
        .disclaimer {
            font-size: 10px;
            text-align: center;
            margin-top: 15px;
            color: #306230;
            border-top: 1px solid #306230;
            padding-top: 8px;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.6);
            align-items: center;
            justify-content: center;
        }
        .modal-content {
            background-color: #b5b5b5;
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            max-width: 280px;
            font-family: monospace;
            border: 3px solid #306230;
        }
        .modal-buttons {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }
        .modal-buttons button {
            font-size: 16px;
            padding: 8px 12px;
            width: 80px;
        }
    </style>
</head>
<body>
<div class="gameboy">
    <div class="screen">
        <div class="balance" id="balanceArea">💰 載入中...</div>
        <div class="positions" id="positionsArea">📋 持倉列表</div>
        <div class="log-area" id="logArea">📜 日誌區域</div>
    </div>
    <div class="controls">
        <button class="start" id="btnStart">▶ 開始</button>
        <button class="stop" id="btnStop">⏹ 停止</button>
        <button class="force" id="btnForce">⛔ 平倉並停止</button>
    </div>
    <div class="refresh-note">
        <span id="statusLed" class="status stopped"></span> 機器人狀態：<span id="runStatus">已停止</span>
    </div>
    <div class="disclaimer">
        ⚠️ 免責聲明：本機器人為輔助交易工具，不保證獲利。任何因使用本程式造成的投資損失，開發者與 AI 協助者概不負責。請謹慎使用，風險自負。
    </div>
</div>

<div id="confirmModal" class="modal">
    <div class="modal-content">
        <p id="modalMessage">確定要執行此操作嗎？</p>
        <div class="modal-buttons">
            <button id="modalConfirm">確認</button>
            <button id="modalCancel">取消</button>
        </div>
    </div>
</div>

<script>
    let pendingAction = null;

    function showModal(message, onConfirm) {
        const modal = document.getElementById('confirmModal');
        const msgSpan = document.getElementById('modalMessage');
        msgSpan.innerText = message;
        modal.style.display = 'flex';
        const confirmBtn = document.getElementById('modalConfirm');
        const cancelBtn = document.getElementById('modalCancel');
        const handler = () => {
            modal.style.display = 'none';
            confirmBtn.removeEventListener('click', handler);
            cancelBtn.removeEventListener('click', cancelHandler);
            onConfirm();
        };
        const cancelHandler = () => {
            modal.style.display = 'none';
            confirmBtn.removeEventListener('click', handler);
            cancelBtn.removeEventListener('click', cancelHandler);
        };
        confirmBtn.addEventListener('click', handler);
        cancelBtn.addEventListener('click', cancelHandler);
    }

    function fetchStatus() {
        fetch('/status')
            .then(res => res.json())
            .then(data => {
                const isRunning = data.running;
                const led = document.getElementById('statusLed');
                const statusText = document.getElementById('runStatus');
                if (isRunning) {
                    led.className = 'status running';
                    statusText.innerText = '運行中';
                } else {
                    led.className = 'status stopped';
                    statusText.innerText = '已停止';
                }
                const username = data.username || '未知';
                const total = data.total || 0;
                const available = data.available || 0;
                document.getElementById('balanceArea').innerHTML = `
                    👤 ${username} &nbsp;| 💰 淨資產: $${total.toFixed(2)} &nbsp;| 可用: $${available.toFixed(2)}
                `;
                const positions = data.positions || [];
                if (positions.length === 0) {
                    document.getElementById('positionsArea').innerHTML = '📋 暫無持倉';
                } else {
                    let html = `<table><th>股票</th><th>股數</th><th>成本</th><th>現價</th><th>盈虧</th></tr>`;
                    positions.forEach(p => {
                        let priceStr = p.price ? p.price.toFixed(2) : '--';
                        let pnlStr = p.pnl ? p.pnl.toFixed(2) : '0.00';
                        let pnlColor = p.pnl >= 0 ? '#2a6b2a' : '#aa4a4a';
                        html += `<tr>
                            <td>${p.symbol}</td>
                            <td>${p.qty.toFixed(4)}</td>
                            <td>${p.cost.toFixed(2)}</td>
                            <td>${priceStr}</td>
                            <td style="color:${pnlColor}">$${pnlStr}</td>
                        </tr>`;
                    });
                    html += `</table>`;
                    document.getElementById('positionsArea').innerHTML = html;
                }
                const logs = data.logs || [];
                const logDiv = document.getElementById('logArea');
                if (logs.length === 0) {
                    logDiv.innerHTML = '📜 尚無日誌';
                } else {
                    let logHtml = '';
                    logs.slice().reverse().forEach(log => {
                        logHtml += `<p>[${log.time}] ${log.level}: ${log.msg}</p>`;
                    });
                    logDiv.innerHTML = logHtml;
                }
            })
            .catch(err => console.error('獲取狀態失敗', err));
    }

    function sendCommand(endpoint, confirmMsg) {
        if (confirmMsg) {
            showModal(confirmMsg, () => {
                fetch(endpoint, { method: 'POST' })
                    .then(res => res.json())
                    .then(data => {
                        console.log(data);
                        fetchStatus();
                    })
                    .catch(err => alert('命令發送失敗: ' + err));
            });
        } else {
            fetch(endpoint, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    console.log(data);
                    fetchStatus();
                })
                .catch(err => alert('命令發送失敗: ' + err));
        }
    }

    document.getElementById('btnStart').addEventListener('click', () => {
        sendCommand('/start', '確定要啟動機器人嗎？');
    });
    document.getElementById('btnStop').addEventListener('click', () => {
        sendCommand('/stop', '確定要停止機器人嗎？停止不會賣出任何股票，機器人將暫停交易。');
    });
    document.getElementById('btnForce').addEventListener('click', () => {
        sendCommand('/force_stop', '⚠️ 確定要強制平倉所有持倉並停止機器人？此操作不可逆！');
    });

    fetchStatus();
    setInterval(fetchStatus, 3000);
</script>
</body>
</html>
'''

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    is_running = not stop_event.is_set() and bot_thread is not None and bot_thread.is_alive()
    if trade_client is None:
        return jsonify({
            "running": is_running,
            "total": 0,
            "available": 0,
            "positions": [],
            "username": webull_username,
            "logs": ui_log_messages[-50:]
        })
    try:
        total, available = get_account_balance(trade_client)
        pos_df = get_positions(trade_client)
        positions = []
        for _, row in pos_df.iterrows():
            symbol = row['symbol']
            qty = row['qty']
            cost = row['cost_price']
            price = get_real_time_price(data_client, symbol)
            pnl = (price - cost) * qty if price else 0
            positions.append({
                'symbol': symbol,
                'qty': qty,
                'cost': cost,
                'price': price,
                'pnl': pnl
            })
        return jsonify({
            "running": is_running,
            "total": total,
            "available": available,
            "positions": positions,
            "username": webull_username,
            "logs": ui_log_messages[-50:]
        })
    except Exception as e:
        logger.exception("獲取狀態失敗")
        return jsonify({"error": str(e)}), 500

@app.route('/start', methods=['POST'])
def start_bot():
    global bot_thread, stop_event, trade_client, data_client, models, webull_username, account_id
    with thread_lock:
        if bot_thread is not None and bot_thread.is_alive() and not stop_event.is_set():
            return jsonify({"status": "already running"})
        if trade_client is None:
            try:
                trade_client, data_client, webull_username, account_id = init_webull_clients()
                models = load_models()
                if not models:
                    raise ValueError("無法載入任何模型，請檢查 models 目錄")
            except Exception as e:
                logger.exception("初始化失敗")
                return jsonify({"status": "init failed", "error": str(e)}), 500
        stop_event.clear()
        bot_thread = threading.Thread(target=bot_loop, daemon=True)
        bot_thread.start()
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_bot():
    global stop_event
    with thread_lock:
        stop_event.set()
    return jsonify({"status": "stopping"})

@app.route('/force_stop', methods=['POST'])
def force_stop():
    global trade_client, stop_event
    if trade_client is None:
        return jsonify({"status": "trade client not ready"})
    force_close_all(trade_client)
    with thread_lock:
        stop_event.set()
    return jsonify({"status": "force stopped and liquidated"})

if __name__ == '__main__':
    os.makedirs("conf", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)