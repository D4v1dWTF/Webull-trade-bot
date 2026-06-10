#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Webull 交易机器人 - 完整 Web 版（与 trade.py 策略完全一致）
- 支持手机访问
- 实时状态查看（持仓、余额、日志）
- 远程启动/停止/强制平仓
- 时间参数：高置信度15:00，禁止买入15:15，智能卖出15:15，强制平仓15:30
=======
Webull 交易機器人 - 終極相容版（自動修復舊版 H5 模型）
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import pickle
import json
import logging
import uuid
import hashlib
import hmac
import base64
import threading
import requests
import tempfile
import h5py
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Tuple, Optional, List
from functools import wraps

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

# ================== 全局配置（与 trade.py 保持一致） ==================
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'AMD', 'TSLA', 'PLTR', 'DELL',
    'INTC', 'BA', 'XOM', 'COIN', 'AVGO',
    'ISRG', 'OUST', 'IBM', 'MU', 'MRVL'
]

MAX_SINGLE_POSITION_PCT = 0.35
USE_MARGIN = True
ALLOW_ADDING_TO_EXISTING = False
BUY_AMOUNT_PCT = 0.95
SAFE_MARGIN = 1.0

TAKE_PROFIT_PCT = 0.005
STOP_LOSS_PCT = 0.005
PROB_THRESHOLD_RISE_MARK = 0.60

ATR_PERIOD = 14
PROB_THRESHOLD_BUY = 0.65
HIGH_CONFIDENCE_THRESHOLD = 0.75
PROB_THRESHOLD_SELL = 0.60
KLINE_INTERVAL = Timespan.M15.name
CHECK_INTERVAL_SEC = 5
SHORT_SUMMARY_INTERVAL_SEC = 10 * 60
LONG_SUMMARY_INTERVAL_SEC = 30 * 60

# 时间参数（与 trade.py 一致）
HIGH_CONFIDENCE_START_TIME = dt_time(15, 0)   # 15:00 后高置信度
STOP_BUY_TIME = dt_time(15, 15)               # 15:15 后禁止买入
START_SELL_TIME = dt_time(15, 15)             # 15:15 后智能卖出
FORCE_SELL_TIME = dt_time(15, 30)             # 15:30 强制平仓

RESERVED_FEE_PER_TRADE = 0.02
MODEL_MAX_AGE_DAYS = 7

RISK_LEVEL_SAFE_PCT = 0.35
RISK_LEVEL_CAUTION_PCT = 0.15
RISK_LEVEL_WARNING_PCT = 0.0
RISK_LEVEL_CRITICAL = -0.15
RISK_CHECK_INTERVAL_SEC = 5
maintenance_margin_pct = 0.25

# 特征列表（与 trade.py 完全相同）
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

# ================== 日志系统（支持前端显示） ==================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_handler = TimedRotatingFileHandler(
    filename=os.path.join(LOG_DIR, "webull_web.log"),
    when="midnight", interval=1, backupCount=90, encoding="utf-8"
)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.addHandler(console)

# 前端日志队列
ui_log_messages = []
<<<<<<< HEAD
def add_ui_log(level: str, msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
=======

def add_ui_log(level: str, msg: str):
    # 排除資產查詢等雜訊
    exclude_keywords = ['美元淨資產', '可用現金']
    if any(k in msg for k in exclude_keywords):
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599
    ui_log_messages.append({"time": timestamp, "level": level, "msg": msg})
    while len(ui_log_messages) > 200:
        ui_log_messages.pop(0)

# 包装 logger
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
    add_ui_log("ERROR", f"异常: {msg}")

logger.info = info_with_ui
logger.error = error_with_ui
logger.warning = warning_with_ui
logger.exception = exception_with_ui

# ================== 全局变量 ==================
stop_event = threading.Event()
bot_thread = None
thread_lock = threading.Lock()
trade_client = None
data_client = None
models = {}
webull_username = "未知"
account_id = None
position_tracker = None  # 将在 bot_loop 中初始化

# ================== 时间辅助函数 ==================
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

<<<<<<< HEAD
# ================== Webull 客户端初始化 ==================
=======
# ================== Webull 客戶端初始化 ==================
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599
def init_webull_clients() -> Tuple[TradeClient, DataClient, str, str]:
    global webull_username, account_id
    logger.info("正在初始化 Webull 客户端...")
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
    
    # 获取账户ID
    resp = trade_client.account_v2.get_account_list()
    accounts = resp.json()
    if not accounts:
        raise ValueError("无法获取账户列表")
    account_id = accounts[0].get("account_id")
    if not account_id:
        raise ValueError("account_id 不存在")
    logger.info(f"获取到 account_id: {account_id}")
    
<<<<<<< HEAD
    # 获取用户名（尝试多种方式）
=======
    # 獲取用戶名
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599
    try:
        webull_username = accounts[0].get("account_name", "未知")
        if webull_username == "未知":
            try:
                profile_resp = trade_client.account_v2.get_account_profile(account_id)
                if profile_resp.status_code == 200:
                    profile_data = profile_resp.json()
                    webull_username = profile_data.get("account_number", "未知")
            except:
                pass
        logger.info(f"Webull 连接成功，账户：{webull_username}")
    except Exception as e:
        webull_username = "未知"
        logger.warning(f"获取账户名称失败: {e}")
    
    return trade_client, data_client, webull_username, account_id

# ================== 市场数据函数（完全复制自 trade.py） ==================
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
            logger.warning(f"获取 {symbol} K线失败 HTTP {res.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"获取 {symbol} K线异常: {e}")
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

def get_account_balance(trade_client: TradeClient) -> Tuple[float, float, float]:
    """返回 (净资产, 现金, 购买力)"""
    try:
        resp = trade_client.account_v2.get_account_list()
        accounts = resp.json()
        if not accounts:
            return 0, 0, 0
        account_id = accounts[0].get("account_id")
        bal_resp = trade_client.account_v2.get_account_balance(account_id)
        bal_data = bal_resp.json()

        usd_cash = 0.0
        usd_net = 0.0
        buying_power = 0.0

        if isinstance(bal_data, dict):
            usd_net = float(bal_data.get("total_net_liquidation_value", 0))
            if "account_currency_assets" in bal_data:
                for asset in bal_data["account_currency_assets"]:
                    if asset.get("currency") == "USD":
                        usd_cash = float(asset.get("cash_balance", 0))
                        buying_power = float(asset.get("buying_power", 0))
                        break
        effective_buying_power = max(0, buying_power - SAFE_MARGIN) if USE_MARGIN else usd_cash
        available_cash = max(usd_cash - RESERVED_FEE_PER_TRADE, 0)
        return usd_net, available_cash, effective_buying_power
    except Exception as e:
        logger.error(f"获取账户余额失败: {e}")
        return 0, 0, 0

def get_positions(trade_client: TradeClient) -> pd.DataFrame:
    """完整持仓解析（支持碎股）"""
    try:
        resp = trade_client.account_v2.get_account_list()
        accounts = resp.json()
        if not accounts:
            return pd.DataFrame()
        account_id = accounts[0].get("account_id")
        pos_resp = trade_client.account_v2.get_account_position(account_id)
        positions = pos_resp.json()
        if not positions:
            return pd.DataFrame()
        
        rows = []
        for pos in positions:
            qty = 0.0
            # 尝试多种字段
            for key in ['position', 'quantity', 'qty', 'shares', 'totalQuantity', 'currentQty', 'fractionalQty', 'oddLotQuantity']:
                if key in pos:
                    try:
                        val = pos[key]
                        if val is None:
                            continue
                        qty = float(val)
                        if qty != 0:
                            break
                    except:
                        continue
            if qty == 0:
                continue
            cost = 0.0
            for key in ['costPrice', 'cost_price', 'avgPrice', 'averagePrice']:
                if key in pos:
                    try:
                        cost = float(pos[key])
                        break
                    except:
                        continue
            symbol = pos.get('symbol', '')
            if symbol:
                rows.append({'symbol': symbol, 'qty': qty, 'cost_price': cost})
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"获取持仓异常: {e}")
        return pd.DataFrame()

# ================== 下单函数（与 trade.py 一致） ==================
def place_buy_order(trade_client: TradeClient, symbol: str, amount_usd: float) -> Tuple[bool, str, Optional[str]]:
    if amount_usd < 1.0:
        return False, f"金额 ${amount_usd:.2f} < 1 美元", None
    _, _, buying_power = get_account_balance(trade_client)
    if buying_power < amount_usd:
        return False, f"购买力不足：需要 ${amount_usd:.2f}，可用 ${buying_power:.2f}", None

    logger.info(f"📈 融资买入: ${amount_usd:.2f} (可用 ${buying_power:.2f})")

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
            logger.info(f"✅ 买入成功: ${amount_usd:.2f} -> {symbol} (订单 {order_id})")
            return True, "成功", order_id
        else:
            logger.error(f"买入失败 HTTP {response.status_code}: {response.text}")
            return False, f"HTTP {response.status_code}", None
    except Exception as e:
        logger.error(f"买入异常: {e}")
        return False, str(e), None

def place_sell_order(trade_client: TradeClient, symbol: str, qty: float) -> Tuple[bool, str, Optional[str]]:
    if qty <= 1e-8:
        return False, f"数量 {qty:.10f} 过小，视为无效", None

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
            logger.info(f"✅ 卖出成功: {qty:.6f} 股 {symbol} (订单 {order_id})")
            return True, "成功", order_id
        else:
            error_text = response.text
            logger.error(f"卖出失败 HTTP {response.status_code}: {error_text}")
            if ("ORDER_ODD_LOT_SELL_QUANTITY_INVALID" in error_text or
                "CAN_SELL_QTY_NOT_ENOUGH" in error_text):
                logger.warning(f"⚠️ 碎股卖出失败，尝试卖出全部持仓 {symbol}")
                pos_df = get_positions(trade_client)
                row = pos_df[pos_df['symbol'] == symbol]
                if not row.empty:
                    fallback_qty = row.iloc[0]['qty']
                    if fallback_qty > 1e-8:
                        logger.info(f"🔄 兜底卖出: {symbol} 全部 {fallback_qty:.6f} 股")
                        return place_sell_order(trade_client, symbol, fallback_qty)
            return False, f"HTTP {response.status_code}", None
    except Exception as e:
        logger.error(f"卖出异常: {e}")
        return False, str(e), None

# ================== 技术指标计算（完整版） ==================
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

<<<<<<< HEAD
# ================== 模型加载与预测 ==================
class CompatibleInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

class DummyDTypePolicy:
    def __init__(self, name='float32'):
        self._name = name
    @property
    def name(self):
        return self._name
    @property
    def compute_dtype(self):
        return tf.float32
    @property
    def variable_dtype(self):
        return tf.float32
    def __repr__(self):
        return f'<DummyDTypePolicy name="{self._name}">'
=======
# ================== 模型載入（修復 H5 檔案中的舊參數） ==================
def fix_h5_model(path):
    """複製並修復 H5 檔案中 InputLayer 的 optional 參數"""
    # 建立臨時檔案
    fd, temp_path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    
    # 複製原始檔案到臨時檔案
    with open(path, 'rb') as src, open(temp_path, 'wb') as dst:
        dst.write(src.read())
    
    # 使用 h5py 修改臨時檔案
    try:
        with h5py.File(temp_path, 'r+') as f:
            # 遞迴查找所有 config 屬性
            def fix_config(name, obj):
                if isinstance(obj, h5py.Dataset) and 'config' in name:
                    config_str = obj[()].decode('utf-8')
                    # 替換 'optional': True/False 為空
                    import re
                    fixed = re.sub(r',?\s*"optional":\s*(true|false)', '', config_str)
                    if fixed != config_str:
                        obj[()] = fixed.encode('utf-8')
                        logger.debug(f"Fixed config in {name}")
            f.visititems(fix_config)
        return temp_path
    except Exception as e:
        logger.warning(f"修復模型失敗: {e}, 使用原始檔案")
        os.unlink(temp_path)
        return path
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599

def load_models() -> Dict[str, Tuple]:
    models_dict = {}
    logger.info("开始加载模型...")
    for symbol in SYMBOLS:
        xgb_path = f"models/US.{symbol}_xgb.pkl"
        lstm_path = f"models/US.{symbol}_lstm.h5"
        scaler_path = f"models/US.{symbol}_scaler.pkl"
        if not (os.path.exists(xgb_path) and os.path.exists(lstm_path) and os.path.exists(scaler_path)):
            logger.warning(f"模型缺失: {symbol}，跳过")
            continue
        with open(xgb_path, 'rb') as f:
            xgb_model = pickle.load(f)
        
        # 修復 LSTM 模型
        fixed_path = fix_h5_model(lstm_path)
        try:
<<<<<<< HEAD
            lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
            logger.info(f"直接加载 {symbol} LSTM 成功")
        except Exception as e:
            logger.warning(f"加载 {symbol} LSTM 遇到错误: {e}，尝试使用自定义对象")
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
                logger.info(f"使用自定义对象加载 {symbol} LSTM 成功")
            except Exception as e2:
                logger.error(f"仍无法加载 {symbol} LSTM: {e2}，跳过此股票")
                continue
=======
            lstm_model = tf.keras.models.load_model(fixed_path, compile=False)
            logger.info(f"成功載入 {symbol} LSTM")
        except Exception as e:
            logger.error(f"無法載入 {symbol} LSTM: {e}，跳過此股票")
            continue
        finally:
            if fixed_path != lstm_path and os.path.exists(fixed_path):
                os.unlink(fixed_path)
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        models_dict[symbol] = (xgb_model, lstm_model, scaler)
        logger.info(f"成功加载模型: {symbol}")
    logger.info(f"模型加载完成，共 {len(models_dict)} 支股票")
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

# ================== 持仓追踪器 ==================
class PositionTracker:
    def __init__(self):
        self.positions = {}
        self.pending_sell = {}

    def add(self, symbol: str, entry_price: float, entry_atr: float):
        self.positions[symbol] = {'entry_price': entry_price, 'entry_atr': entry_atr, 'highest_price': entry_price}
        self.pending_sell.pop(symbol, None)

    def remove(self, symbol: str):
        self.positions.pop(symbol, None)
        self.pending_sell.pop(symbol, None)

    def update_high(self, symbol: str, current_price: float):
        if symbol in self.positions and current_price > self.positions[symbol]['highest_price']:
            self.positions[symbol]['highest_price'] = current_price

    def get_info(self, symbol: str):
        return self.positions.get(symbol)

    def set_pending_sell(self, symbol: str, mark_price: float):
        self.pending_sell[symbol] = {'mark_price': mark_price, 'mark_time': time.time()}

    def get_pending_sell(self, symbol: str):
        return self.pending_sell.get(symbol)

    def clear_pending_sell(self, symbol: str):
        self.pending_sell.pop(symbol, None)

def check_exit_conditions(tracker: PositionTracker, trade_client: TradeClient, data_client: DataClient,
                          symbol: str, current_price: float, df: pd.DataFrame, buy_prob: float, now_time: dt_time) -> Tuple[bool, str, Optional[float]]:
    info = tracker.get_info(symbol)
    if info is None:
        return False, "无持仓", None
    entry_price = info['entry_price']
    profit_loss_pct = (current_price - entry_price) / entry_price

    pos_df = get_positions(trade_client)
    row = pos_df[pos_df['symbol'] == symbol]
    if row.empty:
        return False, "无持仓", None
    qty = row.iloc[0]['qty']

    # 固定止盈止损
    if profit_loss_pct >= TAKE_PROFIT_PCT:
        tracker.clear_pending_sell(symbol)
        return True, f"止盈 (盈利 {profit_loss_pct*100:.2f}%)", qty

    if profit_loss_pct <= -STOP_LOSS_PCT:
        tracker.clear_pending_sell(symbol)
        return True, f"止损 (亏损 {profit_loss_pct*100:.2f}%)", qty

    # 智能卖出（标记确认）
    if is_sell_allowed(now_time):
        pending = tracker.get_pending_sell(symbol)
        if buy_prob > PROB_THRESHOLD_RISE_MARK and pending is None:
            tracker.set_pending_sell(symbol, current_price)
            logger.info(f"  {symbol} 预测上涨 {buy_prob*100:.1f}% > 60%，标记待确认 (标记价 {current_price:.2f})")
            return False, None, None
        elif pending is not None:
            if current_price > pending['mark_price']:
                rise_pct = (current_price - pending['mark_price']) / pending['mark_price']
                tracker.clear_pending_sell(symbol)
                return True, f"智能卖出 (确认上涨，涨 {rise_pct*100:.2f}%)", qty
            elif buy_prob <= PROB_THRESHOLD_RISE_MARK:
                tracker.clear_pending_sell(symbol)
                logger.info(f"  {symbol} 预测概率回落至 {buy_prob*100:.1f}%，取消标记")
                return False, None, None

    # 预测下跌卖出
    sell_prob = 1 - buy_prob
    if is_sell_allowed(now_time) and sell_prob >= PROB_THRESHOLD_SELL:
        tracker.clear_pending_sell(symbol)
        return True, f"智能卖出 (预测下跌 {sell_prob*100:.1f}%)", qty

    # 死叉卖出
    if len(df) >= 50:
        sma_short = df['close'].rolling(12).mean().iloc[-1]
        sma_long = df['close'].rolling(26).mean().iloc[-1]
        prev_short = df['close'].rolling(12).mean().iloc[-2]
        prev_long = df['close'].rolling(26).mean().iloc[-2]
        if prev_short >= prev_long and sma_short < sma_long:
            tracker.clear_pending_sell(symbol)
            return True, "死叉卖出", qty

    tracker.update_high(symbol, current_price)
    return False, None, None

def force_close_all(trade_client: TradeClient, tracker: PositionTracker) -> int:
    pos_df = get_positions(trade_client)
    if pos_df.empty:
        logger.info("无持仓需要平仓")
        return 0
    closed = 0
    for _, row in pos_df.iterrows():
        symbol = row['symbol']
        qty = float(row['qty'])
        if qty <= 1e-8:
            logger.warning(f"跳过 {symbol}: 数量 {qty:.10f} 过小")
            continue
        logger.warning(f"🏳️ 强制平仓: 尝试卖出 {symbol} {qty:.6f} 股")
        success, msg, oid = place_sell_order(trade_client, symbol, qty)
        if success:
            tracker.remove(symbol)
            closed += 1
            logger.info(f"✅ 已卖出 {symbol} {qty:.6f} 股")
        else:
            logger.error(f"❌ 平仓失败 {symbol}: {msg}")
    logger.warning(f"强制平仓完成，共平仓 {closed} 个持仓")
    return closed

<<<<<<< HEAD
# ================== 机器人主循环 ==================
=======
# ================== 機器人主循環 ==================
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599
def bot_loop():
    global stop_event, trade_client, data_client, models, position_tracker
    logger.info("机器人线程启动，开始交易循环")
    tracker = PositionTracker()
<<<<<<< HEAD
    position_tracker = tracker  # 供强制平仓调用

    # 同步现有持仓
=======
    
    last_waiting_msg_time = 0
    last_high_conf_mode = False
    last_stop_buy_mode = False
    last_start_sell_mode = False
    last_force_sell_mode = False
    
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599
    try:
        pos_df = get_positions(trade_client)
        for _, row in pos_df.iterrows():
            symbol = row['symbol']
            if symbol in models:
                cost = row['cost_price']
                if cost == 0:
                    price = get_real_time_price(data_client, symbol)
                    if price:
                        cost = price
                df = get_market_data(data_client, symbol, 200)
                atr_val = df['atr'].iloc[-1] if not df.empty and 'atr' in df.columns else 0
                tracker.add(symbol, cost, atr_val)
                logger.info(f"同步现有持仓 {symbol} 成本 {cost:.2f} 数量 {row['qty']:.6f}")
    except Exception as e:
        logger.exception(f"同步持仓失败: {e}")
        stop_event.set()
        return

<<<<<<< HEAD
    last_short = time.time()
    last_long = time.time()
    
    while not stop_event.is_set():
        try:
            now_ts = time.time()
            if not is_us_market_open():
                time.sleep(CHECK_INTERVAL_SEC)
                continue
=======
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
>>>>>>> 449daf10083799a14f565e7ce27c1c7ad6b26599

            now_et = get_current_et()
            now_time = now_et.time()

            if is_force_sell_time(now_time):
                logger.info("⏰ 15:30 强制平仓")
                force_close_all(trade_client, tracker)
                stop_event.set()
                break

            total, cash, buying_power = get_account_balance(trade_client)

            if now_ts - last_long >= LONG_SUMMARY_INTERVAL_SEC:
                # 完整摘要
                pos_df = get_positions(trade_client)
                logger.info(f"📊 详细摘要 | 净资产: ${total:.2f} | 持仓 {len(pos_df)} 只")
                last_long, last_short = now_ts, now_ts
            elif now_ts - last_short >= SHORT_SUMMARY_INTERVAL_SEC:
                pos_df = get_positions(trade_client)
                logger.info(f"📊 简短摘要 | 净资产: ${total:.2f} | 持仓 {len(pos_df)} 只")
                last_short = now_ts

            pos_df = get_positions(trade_client)
            current_holdings = {row['symbol']: row['qty'] for _, row in pos_df.iterrows()}

            for symbol in SYMBOLS:
                if stop_event.is_set():
                    break
                if symbol not in models:
                    continue
                logger.info(f"🔍 分析 {symbol}")
                df = get_market_data(data_client, symbol, 300)
                if df.empty:
                    continue
                current_price = df['close'].iloc[-1]
                xgb, lstm, scaler = models[symbol]
                buy_prob = predict_probability(data_client, symbol, xgb, lstm, scaler)
                logger.info(f"  {symbol} 上涨概率: {buy_prob:.3f}")

                current_qty = current_holdings.get(symbol, 0)
                current_market_value = current_qty * current_price
                current_ratio = current_market_value / buying_power if buying_power > 0 else 0.0

                # 更新或初始化持仓信息
                if current_qty > 0 and symbol not in tracker.positions:
                    cost = pos_df[pos_df['symbol'] == symbol]['cost_price'].values[0]
                    if cost == 0:
                        cost = current_price
                    atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                    tracker.add(symbol, cost, atr_val)
                    logger.info(f"初始化持仓信息 {symbol} 成本 {cost:.2f} 数量 {current_qty:.6f}")

                # 卖出逻辑
                if current_qty > 0:
                    should_sell, reason, qty_to_sell = check_exit_conditions(
                        tracker, trade_client, data_client, symbol, current_price, df, buy_prob, now_time
                    )
                    if should_sell:
                        logger.info(f"卖出 {symbol}: {reason}")
                        success, msg, oid = place_sell_order(trade_client, symbol, qty_to_sell)
                        if success:
                            tracker.remove(symbol)
                            time.sleep(0.5)
                            total, cash, buying_power = get_account_balance(trade_client)
                        continue

                # 买入逻辑
                if is_buy_allowed(now_time, buy_prob):
                    if not ALLOW_ADDING_TO_EXISTING and tracker.get_info(symbol) is not None:
                        logger.info(f"  {symbol} 已有持仓，禁止加仓")
                        continue

                    if current_ratio >= MAX_SINGLE_POSITION_PCT:
                        logger.info(f"  {symbol} 已达仓位上限 {current_ratio*100:.1f}%")
                        continue

                    max_allowed_value = buying_power * MAX_SINGLE_POSITION_PCT
                    remaining = max_allowed_value - current_market_value
                    if remaining <= 0:
                        logger.info(f"  {symbol} 剩余可用仓位 ${remaining:.2f} <= 0")
                        continue

                    proposed = remaining * BUY_AMOUNT_PCT
                    if proposed < 1.0:
                        logger.info(f"  {symbol} 拟投金额 ${proposed:.2f} < 1 美元")
                        continue

                    logger.info(f"🎯 买入 {symbol} | 价格 {current_price:.2f} | 概率 {buy_prob:.3f} | "
                                f"当前仓位 {current_ratio*100:.1f}% | 拟投 ${proposed:.2f}")
                    success, msg, oid = place_buy_order(trade_client, symbol, proposed)
                    if success:
                        time.sleep(0.5)
                        new_pos = get_positions(trade_client)
                        new_row = new_pos[new_pos['symbol'] == symbol]
                        if not new_row.empty:
                            new_cost = new_row.iloc[0]['cost_price']
                            new_qty = new_row.iloc[0]['qty']
                            if symbol not in tracker.positions:
                                atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                                tracker.add(symbol, new_cost, atr_val)
                            logger.info(f"买入成功，{symbol} 成本 ${new_cost:.2f} 持仓 {new_qty:.6f} 股")
                            total, cash, buying_power = get_account_balance(trade_client)
                        else:
                            logger.warning(f"买入后未找到持仓 {symbol}")
                    else:
                        logger.error(f"买入失败: {msg}")
                else:
                    if buy_prob >= PROB_THRESHOLD_BUY:
                        logger.info(f"  {symbol} 买入被拒绝：时间或置信度条件未满足")

                time.sleep(0.8)

            logger.info(f"⏳ 等待 {CHECK_INTERVAL_SEC} 秒...")
            for _ in range(CHECK_INTERVAL_SEC):
                if stop_event.is_set():
                    break
                time.sleep(1)

        except Exception as e:
            logger.exception(f"交易循环内部错误: {e}")
            time.sleep(CHECK_INTERVAL_SEC)

    logger.info("机器人线程已停止")

# ================== Flask Web 服务 ==================
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>Webull Bot</title>
    <style>
        * { box-sizing: border-box; }
        body {
            background: #1a1a2e;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            color: #eee;
        }
        .container {
            max-width: 700px;
            margin: 0 auto;
            background: #16213e;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            font-size: 1.8rem;
            margin: 0 0 15px 0;
            color: #e94560;
        }
        .status-bar {
            background: #0f3460;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        .status-led {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #e74c3c;
            box-shadow: 0 0 5px #e74c3c;
            margin-right: 8px;
        }
        .status-led.running {
            background: #2ecc71;
            box-shadow: 0 0 5px #2ecc71;
        }
        .balance {
            background: #0f3460;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 0.9rem;
            text-align: center;
        }
        .positions {
            background: #0f3460;
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 15px;
            max-height: 250px;
            overflow-y: auto;
        }
        .positions table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }
        .positions th, .positions td {
            padding: 6px 4px;
            text-align: left;
            border-bottom: 1px solid #2a4a7a;
        }
        .log-area {
            background: #0a1a2a;
            border-radius: 12px;
            padding: 10px;
            height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.75rem;
            margin-bottom: 15px;
        }
        .log-entry {
            border-bottom: 1px solid #1a3a5a;
            padding: 3px 0;
        }
        .log-info { color: #88c0ff; }
        .log-warning { color: #ffaa44; }
        .log-error { color: #ff6666; }
        .buttons {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
        }
        button {
            background: #e94560;
            border: none;
            color: white;
            font-size: 1rem;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 40px;
            cursor: pointer;
            transition: 0.1s linear;
            flex: 1;
            min-width: 100px;
        }
        button:active {
            transform: scale(0.97);
        }
        .btn-start { background: #2ecc71; }
        .btn-stop { background: #e74c3c; }
        .btn-force { background: #f39c12; }
        .refresh-note {
            text-align: center;
            font-size: 0.7rem;
            margin-top: 12px;
            color: #888;
        }
        .disclaimer {
            font-size: 0.7rem;
            text-align: center;
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid #2a4a7a;
            color: #aaa;
        }
        @media (max-width: 500px) {
            .container { padding: 15px; }
            button { padding: 8px 12px; font-size: 0.9rem; }
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🤖 Webull Bot</h1>
    <div class="status-bar">
        <span><span id="statusLed" class="status-led"></span> <span id="statusText">加载中...</span></span>
        <span id="timeDisplay"></span>
    </div>
    <div class="balance" id="balanceArea">💰 加载账户信息...</div>
    <div class="positions" id="positionsArea">📋 持仓列表</div>
    <div class="log-area" id="logArea">📜 日志区域</div>
    <div class="buttons">
        <button class="btn-start" id="btnStart">▶ 启动</button>
        <button class="btn-stop" id="btnStop">⏹ 停止</button>
        <button class="btn-force" id="btnForce">⚠️ 平仓并停止</button>
    </div>
    <div class="refresh-note">数据自动刷新 (3秒)</div>
    <div class="disclaimer">
        ⚠️ 免责声明：本工具为辅助交易系统，不保证盈利。使用风险自负。
    </div>
</div>

<script>
    function fetchStatus() {
        fetch('/status')
            .then(res => res.json())
            .then(data => {
                const running = data.running;
                const led = document.getElementById('statusLed');
                const statusSpan = document.getElementById('statusText');
                if (running) {
                    led.className = 'status-led running';
                    statusSpan.innerText = '运行中';
                } else {
                    led.className = 'status-led';
                    statusSpan.innerText = '已停止';
                }
                document.getElementById('balanceArea').innerHTML = `👤 ${data.username || '未知'} &nbsp;| 💰 净资产: $${data.total.toFixed(2)} &nbsp;| 可用: $${data.available.toFixed(2)}`;
                const positions = data.positions || [];
                if (positions.length === 0) {
                    document.getElementById('positionsArea').innerHTML = '📋 暂无持仓';
                } else {
                    let html = `<table><th>股票</th><th>股数</th><th>成本</th><th>现价</th><th>盈亏</th></tr>`;
                    positions.forEach(p => {
                        const pnl = p.pnl || 0;
                        const color = pnl >= 0 ? '#2ecc71' : '#e74c3c';
                        html += `<tr>
                            <td>${p.symbol}</td>
                            <td>${p.qty.toFixed(4)}</td>
                            <td>$${p.cost.toFixed(2)}</td>
                            <td>${p.price ? '$'+p.price.toFixed(2) : '--'}</td>
                            <td style="color:${color}">$${pnl.toFixed(2)}</td>
                        </tr>`;
                    });
                    html += `</table>`;
                    document.getElementById('positionsArea').innerHTML = html;
                }
                const logs = data.logs || [];
                const logDiv = document.getElementById('logArea');
                if (logs.length === 0) {
                    logDiv.innerHTML = '📜 暂无日志';
                } else {
                    let logHtml = '';
                    logs.slice().reverse().forEach(log => {
                        let levelClass = 'log-info';
                        if (log.level === 'WARNING') levelClass = 'log-warning';
                        if (log.level === 'ERROR') levelClass = 'log-error';
                        logHtml += `<div class="log-entry ${levelClass}">[${log.time}] ${log.level}: ${log.msg}</div>`;
                    });
                    logDiv.innerHTML = logHtml;
                }
                document.getElementById('timeDisplay').innerHTML = new Date().toLocaleTimeString();
            })
            .catch(err => console.error(err));
    }

    function sendCommand(endpoint, confirmMsg) {
        if (confirmMsg && !confirm(confirmMsg)) return;
        fetch(endpoint, { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                console.log(data);
                fetchStatus();
            })
            .catch(err => alert('命令失败: ' + err));
    }

    document.getElementById('btnStart').onclick = () => sendCommand('/start', '确定启动机器人？');
    document.getElementById('btnStop').onclick = () => sendCommand('/stop', '确定停止机器人？不会卖出持仓。');
    document.getElementById('btnForce').onclick = () => sendCommand('/force_stop', '⚠️ 确定强制平仓并停止？此操作不可逆！');

    fetchStatus();
    setInterval(fetchStatus, 3000);
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    running = (bot_thread is not None and bot_thread.is_alive() and not stop_event.is_set())
    if trade_client is None:
        return jsonify({
            "running": False,
            "total": 0,
            "available": 0,
            "positions": [],
            "username": webull_username,
            "logs": ui_log_messages[-50:]
        })
    try:
        total, available, _ = get_account_balance(trade_client)
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
            "running": running,
            "total": total,
            "available": available,
            "positions": positions,
            "username": webull_username,
            "logs": ui_log_messages[-50:]
        })
    except Exception as e:
        logger.exception("获取状态失败")
        return jsonify({"error": str(e)}), 500

@app.route('/start', methods=['POST'])
def start_bot():
    global bot_thread, stop_event, trade_client, data_client, models, webull_username, account_id, position_tracker
    with thread_lock:
        if bot_thread is not None and bot_thread.is_alive() and not stop_event.is_set():
            return jsonify({"status": "already running"})
        if trade_client is None or len(models) == 0:
            try:
                trade_client, data_client, webull_username, account_id = init_webull_clients()
                models = load_models()
                if not models:
                    raise ValueError("无法加载任何模型")
            except Exception as e:
                logger.exception("初始化失败")
                return jsonify({"status": "init failed", "error": str(e)}), 500
        stop_event.clear()
        bot_thread = threading.Thread(target=bot_loop, daemon=True)
        bot_thread.start()
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_bot():
    global stop_event
    stop_event.set()
    return jsonify({"status": "stopping"})

@app.route('/force_stop', methods=['POST'])
def force_stop():
    global trade_client, stop_event, position_tracker
    if trade_client is not None and position_tracker is not None:
        force_close_all(trade_client, position_tracker)
    stop_event.set()
    return jsonify({"status": "force stopped and liquidated"})

if __name__ == '__main__':
    os.makedirs("conf", exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)