#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Webull 短线交易机器人 - 动态资金分配版
- 总购买力扣除7%后，按每只至少10美元分配股票数量（3→2→1）
- 每次账户余额变化时重新计算目标
- 止盈0.8% | 止损0.4%
- 时间模式：高置信度/智能卖出 14:45，禁止买入 15:05，强制平仓 15:15
"""

import os
import sys
import time
import pickle
import json
import logging
import uuid
import hashlib
import hmac
import base64
import requests
import threading
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv

from webull.core.client import ApiClient
from webull.trade.trade_client import TradeClient
from webull.data.data_client import DataClient
from webull.data.common.category import Category
from webull.data.common.timespan import Timespan
from webull.core.exception.exceptions import ServerException

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
load_dotenv()

# 设置全局请求超时
original_request = requests.Session.request
def timeout_request(self, method, url, **kwargs):
    kwargs.setdefault('timeout', (10, 15))
    return original_request(self, method, url, **kwargs)
requests.Session.request = timeout_request

# ================== 核心参数 ==================
SYMBOLS = ['NVDA', 'TSLA', 'MU']
NUM_STOCKS = len(SYMBOLS)
RESERVE_RATIO = 0.07                        # 预留7%总购买力
MIN_PER_STOCK_USD = 15.0                    # 每只股票最低投入金额
MAX_SINGLE_POSITION_PCT = 0.35              # 单股上限35%
USE_MARGIN = True
ALLOW_ADDING_TO_EXISTING = False
SAFE_MARGIN = 1.0
MIN_BUY_AMOUNT = 5.0

# 止盈止损
TAKE_PROFIT_PCT = 0.008      # 0.8%
STOP_LOSS_PCT = 0.004        # 0.4%

# 买入阈值
PROB_THRESHOLD_BUY = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.75
PROB_THRESHOLD_RISE_MARK = 0.60
PROB_THRESHOLD_SELL = 0.60

# 单日最大亏损限制（美元）
MAX_DAILY_LOSS = 1.00

ATR_PERIOD = 14
KLINE_INTERVAL = Timespan.M15.name
CHECK_INTERVAL_SEC = 5
SHORT_SUMMARY_INTERVAL_SEC = 10 * 60
LONG_SUMMARY_INTERVAL_SEC = 30 * 60

# 时间参数（美东）
HIGH_CONFIDENCE_START_TIME = dt_time(14, 45)
START_SELL_TIME = dt_time(14, 45)
STOP_BUY_TIME = dt_time(15, 5)
FORCE_SELL_TIME = dt_time(15, 15)

RESERVED_FEE_PER_TRADE = 0.02
MODEL_MAX_AGE_DAYS = 7

RISK_LEVEL_SAFE_PCT = 0.35
RISK_LEVEL_CAUTION_PCT = 0.15
RISK_LEVEL_WARNING_PCT = 0.0
RISK_LEVEL_CRITICAL = -0.15
RISK_CHECK_INTERVAL_SEC = 5
maintenance_margin_pct = 0.25

risk_force_sell = False
risk_force_sell_lock = threading.Lock()
force_closed_today = False
force_closed_lock = threading.Lock()

daily_loss = 0.0
daily_loss_lock = threading.Lock()
last_reset_day = None

# 特征列表
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

# ================== 日志 ==================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_handler = TimedRotatingFileHandler(
    filename=os.path.join(LOG_DIR, "webull_3stocks.log"),
    when="midnight", interval=1, backupCount=90, encoding="utf-8"
)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.addHandler(console)

_api_zero_warning_shown = False

# ================== 辅助函数 ==================
def get_current_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))

def is_us_market_open() -> bool:
    now_et = get_current_et()
    if now_et.weekday() >= 5:
        return False
    open_t = dt_time(9, 30)
    close_t = dt_time(16, 0)
    return open_t <= now_et.time() <= close_t

def get_time_phase_name(now_time: dt_time) -> str:
    if now_time >= FORCE_SELL_TIME:
        return "🔴 强制平仓阶段 (≥15:15)"
    elif now_time >= START_SELL_TIME:
        return "🟠 智能卖出阶段 (14:45-15:15，禁止买入)"
    elif now_time >= HIGH_CONFIDENCE_START_TIME:
        return "🟡 高置信度阶段 (14:45-15:05，仅买入≥75%信号)"
    else:
        return "🟢 正常交易阶段 (9:30-14:45)"

def is_buy_allowed_with_log(now_time: dt_time, confidence: float) -> Tuple[bool, str]:
    if now_time >= STOP_BUY_TIME:
        return False, "禁止买入 (≥15:05)"
    if now_time >= START_SELL_TIME:
        return False, "禁止买入 (智能卖出阶段)"
    if now_time >= HIGH_CONFIDENCE_START_TIME:
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return True, "高置信度买入"
        else:
            return False, f"需≥{HIGH_CONFIDENCE_THRESHOLD*100:.0f}%"
    else:
        if confidence >= PROB_THRESHOLD_BUY:
            return True, "正常买入"
        else:
            return False, f"需≥{PROB_THRESHOLD_BUY*100:.0f}%"

def is_buy_allowed(now_time: dt_time, confidence: float) -> bool:
    allowed, _ = is_buy_allowed_with_log(now_time, confidence)
    return allowed

def is_sell_allowed(now_time: dt_time) -> bool:
    return now_time >= START_SELL_TIME

def is_force_sell_time(now_time: dt_time) -> bool:
    return now_time >= FORCE_SELL_TIME

def set_risk_force_sell(flag: bool):
    global risk_force_sell
    with risk_force_sell_lock:
        risk_force_sell = flag

def get_risk_force_sell() -> bool:
    with risk_force_sell_lock:
        return risk_force_sell

def set_force_closed_today(flag: bool):
    global force_closed_today
    with force_closed_lock:
        force_closed_today = flag

def get_force_closed_today() -> bool:
    with force_closed_lock:
        return force_closed_today

def update_daily_loss(profit_usd: float):
    global daily_loss, last_reset_day
    now = get_current_et()
    today = now.date()
    with daily_loss_lock:
        if last_reset_day != today:
            daily_loss = 0.0
            last_reset_day = today
        if profit_usd < 0:
            daily_loss += abs(profit_usd)
            logger.info(f"今日累计亏损: ${daily_loss:.2f} (限制 ${MAX_DAILY_LOSS})")
    return daily_loss >= MAX_DAILY_LOSS

# ================== Webull 客户端初始化 ==================
def init_webull_clients() -> Tuple[ApiClient, TradeClient, DataClient]:
    app_key = os.getenv("WEBULL_APP_KEY")
    app_secret = os.getenv("WEBULL_APP_SECRET")
    region_id = os.getenv("WEBULL_REGION_ID", "hk")
    env = os.getenv("WEBULL_ENVIRONMENT", "prod")

    if not app_key or not app_secret:
        raise ValueError("请在 .env 中配置 WEBULL_APP_KEY 和 WEBULL_APP_SECRET")

    logger.info(f"初始化 Webull 客户端，环境: {env.upper()}, 区域: {region_id.upper()}")
    api_client = ApiClient(app_key, app_secret, region_id)

    if env.lower() == "prod":
        api_client.add_endpoint(region_id, "api.webull.hk")
        logger.info("交易接口使用: api.webull.hk")
    else:
        api_client.add_endpoint(region_id, "us-openapi-alb.uat.webullbroker.com")
        logger.warning("使用沙盒环境")

    try:
        trade_client = TradeClient(api_client)
        data_client = DataClient(api_client)
        logger.info("✅ Webull 客户端授权成功")
    except ServerException as e:
        if "UNAUTHORIZED" in str(e):
            logger.error("=" * 60)
            logger.error("❌ 授权失败，请检查手机 App 中的 OpenAPI 通知是否已确认。")
            logger.error("=" * 60)
            raise
        else:
            raise
    return api_client, trade_client, data_client

# ================== 行情与账户 ==================
def get_market_data_with_retry(data_client: DataClient, symbol: str, count=300, max_retries=2):
    for attempt in range(max_retries):
        try:
            df = get_market_data(data_client, symbol, count)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"获取 {symbol} K线失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    logger.error(f"获取 {symbol} K线最终失败")
    return pd.DataFrame()

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
            logger.error(f"获取 {symbol} K线失败 HTTP {res.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"获取 {symbol} K线异常: {e}")
        raise

def get_account_balance_with_retry(trade_client: TradeClient, max_retries=2):
    for attempt in range(max_retries):
        try:
            return get_account_balance(trade_client)
        except Exception as e:
            logger.warning(f"获取账户余额失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    logger.error("获取账户余额最终失败")
    return 0, 0, 0, 0, 0, 0, 0, []

def get_account_balance(trade_client: TradeClient) -> Tuple[float, float, float, float, float, float, float, List]:
    global _api_zero_warning_shown
    try:
        resp = trade_client.account_v2.get_account_list()
        accounts = resp.json()
        if not accounts:
            return 0, 0, 0, 0, 0, 0, 0, []
        account_id = accounts[0].get("account_id")
        bal_resp = trade_client.account_v2.get_account_balance(account_id)
        bal_data = bal_resp.json()

        usd_cash = 0.0
        usd_net = 0.0
        total_market_value = 0.0
        maintenance_margin = 0.0
        unrealized_pnl = 0.0
        buying_power = 0.0
        margin_calls = []

        if isinstance(bal_data, dict):
            usd_net = float(bal_data.get("total_net_liquidation_value", 0))
            total_market_value = float(bal_data.get("total_market_value", 0))
            maintenance_margin = float(bal_data.get("maintenance_margin", 0))
            unrealized_pnl = float(bal_data.get("total_unrealized_profit_loss", 0))
            margin_calls = bal_data.get("open_margin_calls", []) or []

            if "account_currency_assets" in bal_data:
                for asset in bal_data["account_currency_assets"]:
                    if asset.get("currency") == "USD":
                        usd_cash = float(asset.get("cash_balance", 0))
                        buying_power = float(asset.get("buying_power", 0))
                        break

        if usd_net == 0 and total_market_value > 0:
            if not _api_zero_warning_shown:
                logger.info(f"估算净资产 = ${usd_cash + total_market_value:.2f} (API净值为0)")
                _api_zero_warning_shown = True
            usd_net = usd_cash + total_market_value

        if USE_MARGIN:
            effective_buying_power = max(0, buying_power - SAFE_MARGIN)
        else:
            effective_buying_power = usd_cash

        available_cash = max(usd_cash - RESERVED_FEE_PER_TRADE, 0)
        return (usd_net, available_cash, effective_buying_power, maintenance_margin,
                total_market_value, unrealized_pnl, 0, margin_calls)
    except Exception as e:
        logger.error(f"获取账户余额失败: {e}")
        raise

def get_positions(trade_client: TradeClient) -> pd.DataFrame:
    try:
        resp = trade_client.account_v2.get_account_list()
        accounts = resp.json()
        if not accounts:
            return pd.DataFrame(columns=['symbol', 'qty', 'cost_price'])
        account_id = accounts[0].get("account_id")
        pos_resp = trade_client.account_v2.get_account_position(account_id)
        positions = pos_resp.json()
        if not positions:
            return pd.DataFrame(columns=['symbol', 'qty', 'cost_price'])

        rows = []
        for pos in positions:
            qty = 0.0
            for key in ['position', 'quantity', 'qty', 'shares', 'totalQuantity',
                        'currentQty', 'fractionalQty', 'oddLotQuantity']:
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
        if not rows:
            return pd.DataFrame(columns=['symbol', 'qty', 'cost_price'])
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"获取持仓异常: {e}")
        return pd.DataFrame(columns=['symbol', 'qty', 'cost_price'])

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

# ================== 下单函数 ==================
def place_buy_order(trade_client: TradeClient, symbol: str, amount_usd: float) -> Tuple[bool, str, Optional[str]]:
    if amount_usd < MIN_BUY_AMOUNT:
        return False, f"金额 ${amount_usd:.2f} < {MIN_BUY_AMOUNT}", None

    _, _, buying_power, _, _, _, _, _ = get_account_balance(trade_client)
    if buying_power < amount_usd:
        return False, f"购买力不足", None

    logger.info(f"🟢 买入 {symbol} ${amount_usd:.2f}")

    try:
        resp = trade_client.account_v2.get_account_list()
        accounts = resp.json()
        if not accounts:
            return False, "无账户ID", None
        account_id = accounts[0].get("account_id")

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
            logger.info(f"✅ 买入成功 {symbol} ${amount_usd:.2f} (订单 {order_id})")
            return True, "成功", order_id
        else:
            logger.error(f"买入失败 {symbol} HTTP {response.status_code}")
            return False, f"HTTP {response.status_code}", None
    except Exception as e:
        logger.error(f"买入异常 {symbol}: {e}")
        return False, str(e), None

def place_sell_order(trade_client: TradeClient, symbol: str, qty: float, reason: str, profit_usd: float = None) -> Tuple[bool, str, Optional[str]]:
    if qty <= 1e-8:
        return False, "数量过小", None

    if profit_usd is not None:
        logger.info(f"🔴 卖出 {symbol} {qty:.4f}股 | 原因: {reason} | 盈亏: ${profit_usd:.4f}")
    else:
        logger.info(f"🔴 卖出 {symbol} {qty:.4f}股 | 原因: {reason}")

    try:
        resp = trade_client.account_v2.get_account_list()
        accounts = resp.json()
        if not accounts:
            return False, "无账户ID", None
        account_id = accounts[0].get("account_id")

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
            logger.info(f"✅ 卖出成功 {symbol} 订单 {order_id}")
            return True, "成功", order_id
        else:
            logger.error(f"卖出失败 {symbol} HTTP {response.status_code}")
            return False, f"HTTP {response.status_code}", None
    except Exception as e:
        logger.error(f"卖出异常 {symbol}: {e}")
        return False, str(e), None

# ================== 技术指标计算 ==================
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

# ================== 模型加载与预测 ==================
def load_models() -> Dict[str, Tuple]:
    models = {}
    for symbol in SYMBOLS:
        xgb_path = f"models/US.{symbol}_xgb.pkl"
        lstm_path = f"models/US.{symbol}_lstm.h5"
        scaler_path = f"models/US.{symbol}_scaler.pkl"
        if not (os.path.exists(xgb_path) and os.path.exists(lstm_path) and os.path.exists(scaler_path)):
            logger.warning(f"模型缺失: {symbol}，跳过")
            continue
        with open(xgb_path, 'rb') as f:
            xgb_model = pickle.load(f)
        lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        models[symbol] = (xgb_model, lstm_model, scaler)
        logger.info(f"加载模型: {symbol}")
    return models

def predict_probability(data_client: DataClient, symbol: str, xgb_model, lstm_model, scaler) -> float:
    df = get_market_data_with_retry(data_client, symbol, 300)
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

position_tracker = PositionTracker()

# ================== 退出条件检查 ==================
def check_exit_conditions(trade_client: TradeClient, data_client: DataClient, symbol: str,
                          current_price: float, df: pd.DataFrame, buy_prob: float, now_time: dt_time) -> Tuple[bool, str, Optional[float], Optional[float]]:
    info = position_tracker.get_info(symbol)
    if info is None:
        return False, "无持仓", None, None
    entry_price = info['entry_price']
    profit_loss_pct = (current_price - entry_price) / entry_price

    pos_df = get_positions(trade_client)
    if pos_df.empty or 'symbol' not in pos_df.columns:
        return False, "无持仓", None, None
    row = pos_df[pos_df['symbol'] == symbol]
    if row.empty:
        return False, "无持仓", None, None
    qty = row.iloc[0]['qty']
    profit_usd = (current_price - entry_price) * qty

    if profit_loss_pct >= TAKE_PROFIT_PCT:
        position_tracker.clear_pending_sell(symbol)
        return True, f"止盈 (+{profit_loss_pct*100:.2f}%)", qty, profit_usd

    if profit_loss_pct <= -STOP_LOSS_PCT:
        position_tracker.clear_pending_sell(symbol)
        return True, f"止损 ({profit_loss_pct*100:.2f}%)", qty, profit_usd

    if is_sell_allowed(now_time):
        pending = position_tracker.get_pending_sell(symbol)
        if buy_prob > PROB_THRESHOLD_RISE_MARK and pending is None:
            position_tracker.set_pending_sell(symbol, current_price)
            return False, None, None, None
        elif pending is not None:
            if current_price > pending['mark_price']:
                rise_pct = (current_price - pending['mark_price']) / pending['mark_price']
                position_tracker.clear_pending_sell(symbol)
                return True, f"智能卖出 (确认上涨 {rise_pct*100:.2f}%)", qty, profit_usd
            elif buy_prob <= PROB_THRESHOLD_RISE_MARK:
                position_tracker.clear_pending_sell(symbol)
                return False, None, None, None

    sell_prob = 1 - buy_prob
    if is_sell_allowed(now_time) and sell_prob >= PROB_THRESHOLD_SELL and profit_loss_pct > 0:
        position_tracker.clear_pending_sell(symbol)
        return True, f"智能卖出 (预测下跌 {sell_prob*100:.1f}%)", qty, profit_usd

    if len(df) >= 50:
        sma_short = df['close'].rolling(12).mean().iloc[-1]
        sma_long = df['close'].rolling(26).mean().iloc[-1]
        prev_short = df['close'].rolling(12).mean().iloc[-2]
        prev_long = df['close'].rolling(26).mean().iloc[-2]
        if prev_short >= prev_long and sma_short < sma_long:
            position_tracker.clear_pending_sell(symbol)
            return True, "死叉卖出", qty, profit_usd

    position_tracker.update_high(symbol, current_price)
    return False, None, None, None

# ================== 强制平仓 ==================
def force_close_all(trade_client: TradeClient, reason: str = "风控平仓") -> int:
    pos_df = get_positions(trade_client)
    if pos_df.empty:
        logger.info(f"无持仓需要平仓 ({reason})")
        return 0
    closed = 0
    for _, row in pos_df.iterrows():
        symbol = row['symbol']
        qty = float(row['qty'])
        if qty <= 1e-8:
            continue
        success, msg, oid = place_sell_order(trade_client, symbol, qty, reason, None)
        if success:
            position_tracker.remove(symbol)
            closed += 1
        else:
            logger.error(f"平仓失败 {symbol}: {msg}")
    return closed

# ================== 启动摘要 ==================
def print_startup_summary(trade_client: TradeClient, data_client: DataClient):
    total, cash, buying_power, _, _, _, _, _ = get_account_balance_with_retry(trade_client)
    logger.info(f"💰 账户现金: ${cash:.2f} | 购买力: ${buying_power:.2f} | 净资产: ${total:.2f}")
    pos_df = get_positions(trade_client)
    if pos_df.empty:
        logger.info("📋 当前无持仓")
    else:
        logger.info("📋 当前持仓:")
        for _, row in pos_df.iterrows():
            symbol = row['symbol']
            qty = row['qty']
            cost = row['cost_price']
            price = get_real_time_price(data_client, symbol)
            if price:
                pnl = (price - cost) * qty
                logger.info(f"   {symbol}: {qty:.4f}股, 成本${cost:.2f}, 现价${price:.2f}, 盈亏${pnl:.2f}")
            else:
                logger.info(f"   {symbol}: {qty:.4f}股, 成本${cost:.2f}, 价格获取中")

# ================== 风险监控线程 ==================
def get_risk_level_and_excess(net_liquidation: float, maintenance_margin: float, total_market_value: float) -> Tuple[str, float]:
    if total_market_value <= 0:
        return "SAFE", net_liquidation
    required = total_market_value * maintenance_margin_pct
    if maintenance_margin > 0:
        required = maintenance_margin
    margin_excess = net_liquidation - required
    net_abs = abs(net_liquidation) if net_liquidation != 0 else 1
    if net_liquidation <= 0:
        if margin_excess < RISK_LEVEL_CRITICAL * net_abs:
            return "CRITICAL", margin_excess
        elif margin_excess < 0:
            return "DANGER", margin_excess
        else:
            return "WARNING", margin_excess
    else:
        excess_pct = margin_excess / net_abs
        if excess_pct >= RISK_LEVEL_SAFE_PCT:
            return "SAFE", margin_excess
        elif excess_pct >= RISK_LEVEL_CAUTION_PCT:
            return "CAUTION", margin_excess
        elif excess_pct >= RISK_LEVEL_WARNING_PCT:
            return "WARNING", margin_excess
        else:
            return "DANGER", margin_excess

def risk_monitor_thread(trade_client, stop_event: threading.Event):
    logger.info("风险监控线程已启动")
    while not stop_event.is_set():
        try:
            if not is_us_market_open() or get_force_closed_today():
                time.sleep(RISK_CHECK_INTERVAL_SEC)
                continue

            net, _, _, maint, mkt_val, _, _, calls = get_account_balance_with_retry(trade_client)
            level, excess = get_risk_level_and_excess(net, maint, mkt_val)

            if level != "SAFE":
                logger.warning(f"风控等级: {level}, 保证金过剩=${excess:.2f}")
            if calls:
                logger.warning(f"券商风控通知: {calls}")
                set_risk_force_sell(True)
                force_close_all(trade_client, f"券商风控 {calls}")
                set_risk_force_sell(False)
                set_force_closed_today(True)
                time.sleep(60)
                continue

            if level == "CRITICAL" or level == "DANGER":
                set_risk_force_sell(True)
                force_close_all(trade_client, f"{level} 强制平仓")
                set_risk_force_sell(False)
                set_force_closed_today(True)
            else:
                set_risk_force_sell(False)
        except Exception as e:
            logger.error(f"风控线程异常: {e}")
        time.sleep(RISK_CHECK_INTERVAL_SEC)
    logger.info("风险监控线程已停止")

# ================== 主程序 ==================
def main():
    logger.info("=" * 50)
    logger.info("Webull 短线策略 - 动态资金分配版")
    logger.info(f"止盈 {TAKE_PROFIT_PCT*100:.2f}% | 止损 {STOP_LOSS_PCT*100:.2f}%")
    logger.info(f"买入阈值 {PROB_THRESHOLD_BUY*100:.0f}% | 单日最大亏损 ${MAX_DAILY_LOSS}")
    logger.info(f"监控: {', '.join(SYMBOLS)}")
    logger.info(f"资金规则: 总购买力扣除7%后，确保每只股票至少${MIN_PER_STOCK_USD}，动态决定持股数量(3→2→1)")
    logger.info("=" * 50)

    try:
        api_client, trade_client, data_client = init_webull_clients()
    except Exception as e:
        logger.error("初始化失败，程序退出")
        return

    models = load_models()
    if not models:
        logger.error("没有加载任何模型，请确保 models/ 目录下有对应的模型文件")
        return

    print_startup_summary(trade_client, data_client)

    stop_event = threading.Event()
    risk_thread = threading.Thread(target=risk_monitor_thread, args=(trade_client, stop_event), daemon=True)
    risk_thread.start()

    last_short = time.time()
    last_long = time.time()
    last_total_buying_power = None   # 用于检测购买力是否变化

    try:
        while True:
            if get_risk_force_sell():
                time.sleep(5)
                continue

            now_ts = time.time()
            if not is_us_market_open():
                set_force_closed_today(False)
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            now_et = get_current_et()
            now_time = now_et.time()
            phase = get_time_phase_name(now_time)
            logger.info(f"⏰ {now_et.strftime('%H:%M:%S')} | 当前模式: {phase}")

            if get_force_closed_today():
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            if is_force_sell_time(now_time):
                logger.info("15:15 强制平仓")
                force_close_all(trade_client, "15:15 强制平仓")
                set_force_closed_today(True)
                time.sleep(60)
                continue

            total, cash, buying_power, _, _, _, _, _ = get_account_balance_with_retry(trade_client)
            if total == 0 and cash == 0 and buying_power == 0:
                logger.warning("获取账户余额失败，跳过本轮")
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            if now_ts - last_long >= LONG_SUMMARY_INTERVAL_SEC:
                logger.info(f"📊 净资产: ${total:.2f} | 现金: ${cash:.2f} | 购买力: ${buying_power:.2f}")
                last_long = now_ts
                last_short = now_ts
            elif now_ts - last_short >= SHORT_SUMMARY_INTERVAL_SEC:
                logger.info(f"📊 资产: ${total:.2f} | 持仓: {len(get_positions(trade_client))}只")
                last_short = now_ts

            # 获取持仓和市值
            pos_df = get_positions(trade_client)
            current_holdings = {}
            current_total_market_value = 0.0
            if not pos_df.empty and 'symbol' in pos_df.columns:
                for _, row in pos_df.iterrows():
                    symbol = row['symbol']
                    qty = row['qty']
                    price = get_real_time_price(data_client, symbol)
                    if price:
                        market_val = qty * price
                        current_holdings[symbol] = qty
                        current_total_market_value += market_val
                    else:
                        market_val = qty * row['cost_price']
                        current_holdings[symbol] = qty
                        current_total_market_value += market_val

            total_buying_power = buying_power + current_total_market_value
            usable_power = total_buying_power * (1 - RESERVE_RATIO)   # 扣除预留后的可用资金

            # 动态计算目标股票数量
            target_stocks = 3
            per_stock = usable_power / 3
            if per_stock < MIN_PER_STOCK_USD:
                target_stocks = 2
                per_stock = usable_power / 2
                if per_stock < MIN_PER_STOCK_USD:
                    target_stocks = 1
                    per_stock = usable_power

            # 如果购买力发生变化（或首轮），输出当前分配策略
            if last_total_buying_power != total_buying_power or last_total_buying_power is None:
                logger.info(f"💰 总购买力: ${total_buying_power:.2f} (剩余: ${buying_power:.2f} + 持仓: ${current_total_market_value:.2f})")
                logger.info(f"📐 动态分配: 目标持股 {target_stocks} 只，每只目标 ${per_stock:.2f} (预留7%后可用 ${usable_power:.2f})")
                last_total_buying_power = total_buying_power

            # 收集符合条件的买入信号
            if not is_force_sell_time(now_time) and not get_force_closed_today() and not update_daily_loss(0):
                signals = []
                for symbol in SYMBOLS:
                    if symbol not in models:
                        continue
                    df = get_market_data_with_retry(data_client, symbol, 300)
                    if df.empty:
                        continue
                    current_price = df['close'].iloc[-1]
                    xgb, lstm, scaler = models[symbol]
                    buy_prob = predict_probability(data_client, symbol, xgb, lstm, scaler)
                    logger.info(f"🔍 {symbol} | 预测概率: {buy_prob:.3f}")

                    buy_allowed, buy_reason = is_buy_allowed_with_log(now_time, buy_prob)
                    if not buy_allowed:
                        continue

                    current_qty = current_holdings.get(symbol, 0)
                    current_market_value = current_qty * current_price
                    current_ratio = current_market_value / total_buying_power if total_buying_power > 0 else 0.0
                    if current_qty > 0 and not ALLOW_ADDING_TO_EXISTING:
                        continue
                    if current_ratio >= MAX_SINGLE_POSITION_PCT:
                        continue
                    signals.append((symbol, buy_prob, current_price, df))

                # 按概率排序，只取前 target_stocks 只
                signals.sort(key=lambda x: x[1], reverse=True)
                signals = signals[:target_stocks]

                for symbol, buy_prob, current_price, df in signals:
                    current_qty = current_holdings.get(symbol, 0)
                    current_market_value = current_qty * current_price
                    need_to_buy = per_stock - current_market_value
                    if need_to_buy <= 0:
                        continue
                    need_to_buy = min(need_to_buy, buying_power)
                    if need_to_buy < MIN_BUY_AMOUNT:
                        continue
                    logger.info(f"🎯 买入 {symbol} | 价格 {current_price:.2f} | 概率 {buy_prob:.3f} | 金额 ${need_to_buy:.2f}")
                    success, msg, oid = place_buy_order(trade_client, symbol, need_to_buy)
                    if success:
                        atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0
                        position_tracker.add(symbol, current_price, atr_val)
                        current_holdings[symbol] = current_holdings.get(symbol, 0) + (need_to_buy / current_price)
                        buying_power -= need_to_buy
                        time.sleep(0.5)
                    else:
                        logger.error(f"买入失败 {symbol}: {msg}")
                    if buying_power < MIN_BUY_AMOUNT:
                        break

            # 卖出逻辑
            pos_df = get_positions(trade_client)
            for symbol in SYMBOLS:
                if symbol not in models:
                    continue
                current_qty = pos_df[pos_df['symbol'] == symbol]['qty'].values[0] if not pos_df.empty and 'symbol' in pos_df.columns and not pos_df[pos_df['symbol'] == symbol].empty else 0
                if current_qty == 0:
                    continue
                df = get_market_data_with_retry(data_client, symbol, 300)
                if df.empty:
                    continue
                current_price = df['close'].iloc[-1]
                xgb, lstm, scaler = models[symbol]
                buy_prob = predict_probability(data_client, symbol, xgb, lstm, scaler)
                should_sell, reason, qty_to_sell, profit_usd = check_exit_conditions(
                    trade_client, data_client, symbol, current_price, df, buy_prob, now_time
                )
                if should_sell:
                    if profit_usd is not None and profit_usd < 0:
                        update_daily_loss(profit_usd)
                    success, msg, oid = place_sell_order(trade_client, symbol, qty_to_sell, reason, profit_usd)
                    if success:
                        if abs(qty_to_sell - current_qty) < 0.0001:
                            position_tracker.remove(symbol)
                        time.sleep(0.5)
                        total, cash, buying_power, _, _, _, _, _ = get_account_balance_with_retry(trade_client)
                    else:
                        logger.error(f"卖出失败 {symbol}: {msg}")
                time.sleep(0.2)

            logger.info(f"⏳ 等待 {CHECK_INTERVAL_SEC} 秒...")
            time.sleep(CHECK_INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("用户手动停止机器人")
    except Exception as e:
        logger.exception(f"程序异常: {e}")
    finally:
        logger.info("正在停止机器人...")
        stop_event.set()
        risk_thread.join(timeout=5)
        logger.info("机器人已退出")

if __name__ == "__main__":
    main()