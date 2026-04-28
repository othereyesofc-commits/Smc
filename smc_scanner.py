"""
╔══════════════════════════════════════════════════════════════════════╗
║           APEX SMC SCANNER — Binance Futures                        ║
║   Detecta automaticamente oportunidades SMC de alta probabilidade   ║
║   e envia alertas no Telegram em tempo real.                        ║
╚══════════════════════════════════════════════════════════════════════╝

SETUPS DETECTADOS:
  1. Liquidity Sweep + CHoCH  (maior probabilidade)
  2. BOS + Retorno ao Order Block
  3. FVG dentro de Order Block (confluência)
  4. Equal Highs/Lows (liquidez acumulada prestes a ser varrida)

COMO FUNCIONA:
  - A cada 15 minutos, busca os top pares por volume na Binance Futures
  - Analisa cada par nos timeframes 15m e 1h
  - Pontua cada setup de 0-10 com base em confluências SMC
  - Envia alerta apenas para setups com score >= 7
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import os
import schedule
import logging

# ── Configuração de logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Configurações (lidas de variáveis de ambiente para segurança) ──────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "SEU_TOKEN_AQUI")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "SEU_CHAT_ID_AQUI")

TIMEFRAMES        = ["15m", "1h"]          # timeframes analisados
TOP_PAIRS_COUNT   = 40                      # quantos pares escanear
MIN_VOLUME_USDT   = 30_000_000             # volume mínimo diário (30M USDT)
MIN_SCORE         = 7                       # score mínimo para alertar (0-10)
CANDLES_LIMIT     = 200                     # candles por análise
SWING_WINDOW      = 5                       # janela para detectar swing points
EQL_TOLERANCE     = 0.0025                  # tolerância 0.25% para Equal H/L
SWEEP_TOLERANCE   = 0.003                   # tolerância 0.30% para sweep
IMPULSE_MIN_BODY  = 0.6                     # corpo mínimo de vela impulsiva (60%)

BASE_URL = "https://fapi.binance.com"      # Binance USDT-M Futures

# ── Sessões de trading (horário UTC) ──────────────────────────────────────────
SESSIONS = {
    "Asia":     (22, 2),   # 22:00 - 02:00 UTC (Acumulação AMD)
    "London":   (2,  5),   # 02:00 - 05:00 UTC (Manipulação AMD)
    "New York": (12, 17),  # 12:00 - 17:00 UTC (Distribuição AMD)
}


# ══════════════════════════════════════════════════════════════════════════════
#  CAMADA 1 — BUSCA DE DADOS (Binance Futures API)
# ══════════════════════════════════════════════════════════════════════════════

def get_top_pairs() -> list[str]:
    """
    Retorna os top pares USDT perpetual futures ordenados por volume 24h.
    Exclui pares de stablecoins e tokens problemáticos.
    """
    try:
        resp = requests.get(f"{BASE_URL}/fapi/v1/ticker/24hr", timeout=10)
        resp.raise_for_status()
        tickers = resp.json()

        # Filtra apenas pares USDT e com volume mínimo
        filtered = [
            t for t in tickers
            if t["symbol"].endswith("USDT")
            and float(t["quoteVolume"]) >= MIN_VOLUME_USDT
            and not any(s in t["symbol"] for s in ["BUSD", "USDC", "TUSD", "DAI"])
        ]

        # Ordena por volume decrescente e pega os top N
        sorted_pairs = sorted(filtered, key=lambda x: float(x["quoteVolume"]), reverse=True)
        symbols = [t["symbol"] for t in sorted_pairs[:TOP_PAIRS_COUNT]]

        log.info(f"📊 {len(symbols)} pares selecionados para análise")
        return symbols

    except Exception as e:
        log.error(f"Erro ao buscar pares: {e}")
        # Fallback para os principais pares caso a API falhe
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]


def get_candles(symbol: str, interval: str, limit: int = CANDLES_LIMIT) -> pd.DataFrame | None:
    """
    Busca candles OHLCV da Binance Futures e retorna um DataFrame limpo.
    Colunas: open, high, low, close, volume (todos float64)
    """
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(f"{BASE_URL}/fapi/v1/klines", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # Converte para tipos numéricos e timestamp legível
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    except Exception as e:
        log.warning(f"Erro ao buscar candles {symbol}/{interval}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CAMADA 2 — ALGORITMOS SMC (o cérebro do bot)
# ══════════════════════════════════════════════════════════════════════════════

def find_swing_points(df: pd.DataFrame, window: int = SWING_WINDOW) -> dict:
    """
    Detecta Swing Highs e Swing Lows estruturais.

    Um Swing High é uma máxima que é MAIOR que todas as máximas
    nas 'window' velas anteriores E posteriores — é um topo real.
    Um Swing Low é o oposto — um fundo real.

    Retorna dicionário com índices e preços de cada swing point.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs = []  # lista de (índice, preço)
    swing_lows  = []

    for i in range(window, n - window):
        # Verifica se é o ponto mais alto na janela
        is_swing_high = all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                        all(highs[i] >= highs[i+j] for j in range(1, window+1))

        # Verifica se é o ponto mais baixo na janela
        is_swing_low  = all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                        all(lows[i] <= lows[i+j] for j in range(1, window+1))

        if is_swing_high:
            swing_highs.append((i, highs[i]))
        if is_swing_low:
            swing_lows.append((i, lows[i]))

    return {"highs": swing_highs, "lows": swing_lows}


def detect_equal_levels(swings: dict, tolerance: float = EQL_TOLERANCE) -> dict:
    """
    Detecta Equal Highs e Equal Lows — as pools de liquidez.

    Dois swing highs são "iguais" se a diferença percentual entre eles
    for menor que a tolerância (padrão 0.25%). Esses níveis têm
    ordens stop acumuladas — o smart money vai até eles para coletar.
    """
    eq_highs = []
    eq_lows  = []

    # Compara cada par de swing highs
    highs = swings["highs"]
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            price_i, price_j = highs[i][1], highs[j][1]
            diff = abs(price_i - price_j) / price_i
            if diff <= tolerance:
                eq_highs.append({
                    "price": (price_i + price_j) / 2,  # nível médio
                    "idx1": highs[i][0],
                    "idx2": highs[j][0],
                    "strength": 1 / diff if diff > 0 else 999  # quanto mais igual, mais forte
                })

    # Compara cada par de swing lows
    lows = swings["lows"]
    for i in range(len(lows)):
        for j in range(i+1, len(lows)):
            price_i, price_j = lows[i][1], lows[j][1]
            diff = abs(price_i - price_j) / price_i
            if diff <= tolerance:
                eq_lows.append({
                    "price": (price_i + price_j) / 2,
                    "idx1": lows[i][0],
                    "idx2": lows[j][0],
                    "strength": 1 / diff if diff > 0 else 999
                })

    # Ordena pelo mais recente (maior índice)
    eq_highs.sort(key=lambda x: x["idx2"], reverse=True)
    eq_lows.sort(key=lambda x:  x["idx2"], reverse=True)

    return {"eq_highs": eq_highs, "eq_lows": eq_lows}


def detect_market_structure(df: pd.DataFrame, swings: dict) -> dict:
    """
    Detecta BOS (Break of Structure) e CHoCH (Change of Character).

    BOS = rompimento do último swing point na direção da tendência atual.
        → Confirma continuação. Procure OBs para entrar na direção.

    CHoCH = rompimento do último swing point CONTRA a tendência atual.
        → Sinal de possível reversão. Fique atento ao próximo sweep.

    Retorna a estrutura mais recente detectada.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)

    swing_highs = swings["highs"]
    swing_lows  = swings["lows"]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"type": None, "direction": None}

    # Pega os dois últimos swing points de cada lado
    last_high   = swing_highs[-1]   # (idx, price)
    prev_high   = swing_highs[-2]
    last_low    = swing_lows[-1]
    prev_low    = swing_lows[-2]

    # Determina tendência atual: se o último high e low são maiores que os anteriores → bullish
    is_bullish_structure = (last_high[1] > prev_high[1]) and (last_low[1] > prev_low[1])
    is_bearish_structure = (last_high[1] < prev_high[1]) and (last_low[1] < prev_low[1])

    # Verifica se a última vela rompeu algum nível
    latest_close = closes[-1]
    latest_high  = highs[-1]
    latest_low   = lows[-1]

    result = {"type": None, "direction": None, "level": None, "candle_idx": n-1}

    if is_bullish_structure:
        # Em estrutura bullish: CHoCH = fecha abaixo do último swing low
        if latest_close < last_low[1]:
            result = {"type": "CHoCH", "direction": "bearish",
                      "level": last_low[1], "candle_idx": n-1}
        # BOS bullish = fecha acima do último swing high
        elif latest_close > last_high[1]:
            result = {"type": "BOS", "direction": "bullish",
                      "level": last_high[1], "candle_idx": n-1}

    elif is_bearish_structure:
        # Em estrutura bearish: CHoCH = fecha acima do último swing high
        if latest_close > last_high[1]:
            result = {"type": "CHoCH", "direction": "bullish",
                      "level": last_high[1], "candle_idx": n-1}
        # BOS bearish = fecha abaixo do último swing low
        elif latest_close < last_low[1]:
            result = {"type": "BOS", "direction": "bearish",
                      "level": last_low[1], "candle_idx": n-1}

    return result


def detect_order_blocks(df: pd.DataFrame, swings: dict) -> dict:
    """
    Detecta Order Blocks bullish e bearish.

    Bullish OB = última vela BEARISH antes de um movimento impulsivo de ALTA.
    Bearish OB = última vela BULLISH antes de um movimento impulsivo de BAIXA.

    Uma vela é "impulsiva" quando seu corpo representa > 60% do range total
    E é maior que a vela anterior — indica participação institucional.
    """
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)

    bullish_obs = []  # OBs de demanda (zonas de compra)
    bearish_obs = []  # OBs de oferta (zonas de venda)

    for i in range(2, n - 1):
        body      = abs(closes[i] - opens[i])
        range_    = highs[i] - lows[i]
        prev_body = abs(closes[i-1] - opens[i-1])
        prev_range = highs[i-1] - lows[i-1]

        if range_ == 0:
            continue

        body_ratio = body / range_

        # Vela impulsiva bullish: vela verde com corpo grande, maior que a anterior
        is_bullish_impulse = (closes[i] > opens[i] and
                              body_ratio >= IMPULSE_MIN_BODY and
                              body > prev_body)

        # Vela impulsiva bearish: vela vermelha com corpo grande, maior que a anterior
        is_bearish_impulse = (closes[i] < opens[i] and
                              body_ratio >= IMPULSE_MIN_BODY and
                              body > prev_body)

        if is_bullish_impulse and i > 0:
            # O OB Bullish é a última vela BEARISH antes desta impulsiva
            for j in range(i-1, max(i-5, 0), -1):
                if closes[j] < opens[j]:  # vela bearish
                    bullish_obs.append({
                        "idx": j,
                        "top": opens[j],      # topo do OB (abertura da vela bearish)
                        "bottom": closes[j],  # fundo do OB (fechamento)
                        "impulse_size": body,
                        "mitigated": False,
                    })
                    break

        if is_bearish_impulse and i > 0:
            # O OB Bearish é a última vela BULLISH antes desta impulsiva
            for j in range(i-1, max(i-5, 0), -1):
                if closes[j] > opens[j]:  # vela bullish
                    bearish_obs.append({
                        "idx": j,
                        "top": closes[j],     # topo do OB (fechamento da vela bullish)
                        "bottom": opens[j],   # fundo do OB (abertura)
                        "impulse_size": body,
                        "mitigated": False,
                    })
                    break

    # Verifica quais OBs já foram mitigados (preço passou através deles)
    current_price = closes[-1]
    for ob in bullish_obs:
        # OB bullish mitigado se o preço fechou abaixo do fundo
        if current_price < ob["bottom"]:
            ob["mitigated"] = True

    for ob in bearish_obs:
        # OB bearish mitigado se o preço fechou acima do topo
        if current_price > ob["top"]:
            ob["mitigated"] = True

    # Retorna apenas OBs não mitigados, ordenados do mais recente
    active_bull_obs = [ob for ob in bullish_obs if not ob["mitigated"]][-3:]
    active_bear_obs = [ob for ob in bearish_obs if not ob["mitigated"]][-3:]

    return {"bullish": active_bull_obs, "bearish": active_bear_obs}


def detect_fvg(df: pd.DataFrame) -> dict:
    """
    Detecta Fair Value Gaps (FVGs) — desequilíbrios de preço.

    FVG Bullish: candle[i-1].high < candle[i+1].low
        → Há um gap entre a máxima de i-1 e a mínima de i+1
        → O preço costuma retornar para preencher (mitigar) esse gap

    FVG Bearish: candle[i-1].low > candle[i+1].high
        → Gap de baixa — zona de oferta desequilibrada
    """
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    bullish_fvgs = []
    bearish_fvgs = []

    for i in range(1, n - 1):
        # FVG Bullish: gap entre máx[i-1] e mín[i+1]
        if lows[i+1] > highs[i-1]:
            gap_size = lows[i+1] - highs[i-1]
            gap_pct  = gap_size / highs[i-1]
            if gap_pct > 0.001:  # gap mínimo de 0.1% para ser relevante
                # Verifica se ainda não foi mitigado
                current_low = closes[-1]
                mitigated   = current_low < lows[i+1]
                bullish_fvgs.append({
                    "idx": i,
                    "top": lows[i+1],
                    "bottom": highs[i-1],
                    "size_pct": gap_pct,
                    "mitigated": mitigated,
                })

        # FVG Bearish: gap entre mín[i-1] e máx[i+1]
        if highs[i+1] < lows[i-1]:
            gap_size = lows[i-1] - highs[i+1]
            gap_pct  = gap_size / lows[i-1]
            if gap_pct > 0.001:
                current_high = closes[-1]
                mitigated    = current_high > highs[i+1]
                bearish_fvgs.append({
                    "idx": i,
                    "top": lows[i-1],
                    "bottom": highs[i+1],
                    "size_pct": gap_pct,
                    "mitigated": mitigated,
                })

    # Retorna apenas FVGs ativos, mais recentes
    active_bull = [f for f in bullish_fvgs if not f["mitigated"]][-3:]
    active_bear = [f for f in bearish_fvgs if not f["mitigated"]][-3:]

    return {"bullish": active_bull, "bearish": active_bear}


def detect_liquidity_sweep(df: pd.DataFrame, swings: dict,
                           eq_levels: dict) -> dict | None:
    """
    Detecta o setup mais poderoso do SMC: Liquidity Sweep + CHoCH.

    O processo é:
    1. Preço varre um nível de liquidez (Equal Low, swing low)
       → Wick penetra o nível MAS a vela FECHA acima (sweep, não run)
    2. Logo após, ocorre um CHoCH confirmando reversão
    → Este é o momento em que o smart money coletou as ordens
       e está iniciando o movimento real.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n      = len(df)

    # Analisa apenas as últimas 20 velas (setup recente é mais relevante)
    lookback = min(20, n - 2)

    for i in range(n - lookback, n - 1):
        candle_low   = lows[i]
        candle_high  = highs[i]
        candle_close = closes[i]
        candle_open  = df["open"].values[i]

        # ── Teste de SWEEP BULLISH ────────────────────────────────────────────
        # Procura por swing lows ou Equal Lows que foram varridos
        for sl in swings["lows"]:
            sl_idx, sl_price = sl
            if sl_idx >= i:
                continue

            # O wick entrou abaixo do swing low (varreu a liquidez)
            swept_below = candle_low < sl_price * (1 - SWEEP_TOLERANCE)

            # MAS o fechamento ficou ACIMA do swing low (não foi run, foi sweep)
            closed_above = candle_close > sl_price

            if swept_below and closed_above:
                # Confirma CHoCH: a próxima vela fecha acima da abertura da vela de sweep
                next_close = closes[min(i+1, n-1)]
                choch_confirmed = next_close > candle_open

                return {
                    "type": "Liquidity Sweep + CHoCH",
                    "direction": "LONG 🟢",
                    "sweep_level": sl_price,
                    "sweep_idx": i,
                    "choch_confirmed": choch_confirmed,
                    "entry_zone_top": candle_close,
                    "entry_zone_bottom": candle_low,
                    "stop_aggressive": candle_low * 0.998,
                    "stop_conservative": candle_low * 0.995,
                }

        # ── Teste de SWEEP BEARISH ────────────────────────────────────────────
        for sh in swings["highs"]:
            sh_idx, sh_price = sh
            if sh_idx >= i:
                continue

            swept_above  = candle_high > sh_price * (1 + SWEEP_TOLERANCE)
            closed_below = candle_close < sh_price

            if swept_above and closed_below:
                next_close = closes[min(i+1, n-1)]
                choch_confirmed = next_close < candle_open

                return {
                    "type": "Liquidity Sweep + CHoCH",
                    "direction": "SHORT 🔴",
                    "sweep_level": sh_price,
                    "sweep_idx": i,
                    "choch_confirmed": choch_confirmed,
                    "entry_zone_top": candle_high,
                    "entry_zone_bottom": candle_close,
                    "stop_aggressive": candle_high * 1.002,
                    "stop_conservative": candle_high * 1.005,
                }

    return None


def get_amd_session() -> str:
    """
    Retorna a sessão AMD atual baseado no horário UTC.
    Útil para contextualizar o alerta — setup na London tem mais peso.
    """
    hour = datetime.now(timezone.utc).hour
    if 2 <= hour < 5:
        return "⚡ LONDON (Manipulação — ALTA ATENÇÃO)"
    elif 12 <= hour < 17:
        return "🗽 NEW YORK (Distribuição — Melhor para entrar)"
    elif 22 <= hour or hour < 2:
        return "🌙 ASIA (Acumulação — Aguardar)"
    else:
        return "🔄 Entre sessões"


# ══════════════════════════════════════════════════════════════════════════════
#  CAMADA 3 — SISTEMA DE SCORE (qualidade do setup)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_setup_score(
    sweep: dict | None,
    structure: dict,
    order_blocks: dict,
    fvgs: dict,
    eq_levels: dict,
    timeframe: str,
    current_price: float,
) -> tuple[int, list[str]]:
    """
    Pontua a qualidade do setup de 0 a 10 com base em confluências SMC.

    Cada confluência adiciona pontos:
    - Sweep detectado:          +3 pontos (fundamento do setup)
    - CHoCH confirmado:         +2 pontos (confirmação da reversão)
    - OB na zona de entrada:    +2 pontos (confluência institucional)
    - FVG na zona de entrada:   +1 ponto  (desequilíbrio adicional)
    - Equal Highs/Lows próximos:+1 ponto  (alvo claro de liquidez)
    - Timeframe 1h:             +1 ponto  (maior confiabilidade)
    """
    score = 0
    reasons = []

    if sweep:
        score += 3
        reasons.append("✅ Liquidity Sweep detectado")

        if sweep.get("choch_confirmed"):
            score += 2
            reasons.append("✅ CHoCH confirmado após sweep")
        else:
            score += 1
            reasons.append("⚠️ CHoCH pendente de confirmação")

        # Verifica OB na zona de entrada do sweep
        direction_is_long = "LONG" in sweep["direction"]
        relevant_obs = order_blocks["bullish"] if direction_is_long else order_blocks["bearish"]

        for ob in relevant_obs[-2:]:
            ob_in_zone = (ob["bottom"] <= sweep["entry_zone_top"] and
                          ob["top"] >= sweep["entry_zone_bottom"])
            if ob_in_zone:
                score += 2
                reasons.append(f"✅ Order Block na zona de entrada")
                break

        # Verifica FVG na zona de entrada
        relevant_fvgs = fvgs["bullish"] if direction_is_long else fvgs["bearish"]
        for fvg in relevant_fvgs[-2:]:
            fvg_in_zone = (fvg["bottom"] <= sweep["entry_zone_top"] and
                           fvg["top"] >= sweep["entry_zone_bottom"])
            if fvg_in_zone:
                score += 1
                reasons.append("✅ FVG dentro da zona de entrada")
                break

        # Verifica liquidez próxima como alvo (EQL/EQH)
        if direction_is_long and eq_levels["eq_highs"]:
            score += 1
            reasons.append(f"✅ Equal Highs acima como alvo de liquidez")
        elif not direction_is_long and eq_levels["eq_lows"]:
            score += 1
            reasons.append(f"✅ Equal Lows abaixo como alvo de liquidez")

    # Timeframe 1h é mais confiável que 15m
    if timeframe == "1h":
        score += 1
        reasons.append("✅ Setup em timeframe 1H (maior peso)")

    return min(score, 10), reasons


# ══════════════════════════════════════════════════════════════════════════════
#  CAMADA 4 — FORMATAÇÃO E ENVIO DE ALERTAS (Telegram)
# ══════════════════════════════════════════════════════════════════════════════

def format_alert(
    symbol: str,
    timeframe: str,
    sweep: dict,
    score: int,
    reasons: list[str],
    current_price: float,
    eq_levels: dict,
) -> str:
    """
    Formata a mensagem de alerta para o Telegram.
    Inclui todos os dados necessários para a decisão: entrada, stop, alvo.
    """
    session = get_amd_session()
    direction = sweep["direction"]
    is_long = "LONG" in direction

    # Calcula alvo baseado na próxima pool de liquidez
    if is_long and eq_levels["eq_highs"]:
        target = eq_levels["eq_highs"][0]["price"]
        rr_distance = target - sweep["entry_zone_top"]
        stop_distance = sweep["entry_zone_top"] - sweep["stop_aggressive"]
        rr_ratio = rr_distance / stop_distance if stop_distance > 0 else 0
    elif not is_long and eq_levels["eq_lows"]:
        target = eq_levels["eq_lows"][0]["price"]
        rr_distance = sweep["entry_zone_bottom"] - target
        stop_distance = sweep["stop_aggressive"] - sweep["entry_zone_bottom"]
        rr_ratio = rr_distance / stop_distance if stop_distance > 0 else 0
    else:
        # Estimativa de alvo baseada no tamanho do sweep
        pct_move = abs(current_price - sweep["sweep_level"]) / sweep["sweep_level"]
        target = current_price * (1 + pct_move * 2) if is_long else current_price * (1 - pct_move * 2)
        rr_ratio = 2.0

    score_bar = "█" * score + "░" * (10 - score)
    choch_status = "✅ CONFIRMADO" if sweep.get("choch_confirmed") else "⏳ AGUARDANDO"

    msg = f"""
⚡ *APEX SMC — SINAL DETECTADO*
━━━━━━━━━━━━━━━━━━━━━━
🪙 *Par:* `{symbol}` | ⏱ `{timeframe}`
📍 *Preço Atual:* `{current_price:.6f}`
🎯 *Setup:* {sweep['type']}
📊 *Direção:* {direction}
🔄 *CHoCH:* {choch_status}

📈 *ZONAS DE OPERAÇÃO*
━━━━━━━━━━━━━━━━━━━━━━
🟡 Entrada:  `{sweep['entry_zone_bottom']:.6f}` — `{sweep['entry_zone_top']:.6f}`
🟢 Alvo:     `{target:.6f}`
🔴 Stop (agressivo):    `{sweep['stop_aggressive']:.6f}`
🔴 Stop (conservador):  `{sweep['stop_conservative']:.6f}`
⚖️ R:R Estimado: `{rr_ratio:.1f}x`

⭐ *SCORE:* `{score}/10`
`{score_bar}`

✅ *CONFLUÊNCIAS:*
{chr(10).join(reasons)}

🕐 *Sessão:* {session}
━━━━━━━━━━━━━━━━━━━━━━
⚠️ _Sempre confirme no gráfico antes de entrar._
    """.strip()

    return msg


def send_telegram(message: str) -> bool:
    """Envia mensagem formatada para o Telegram via Bot API."""
    if TELEGRAM_TOKEN == "SEU_TOKEN_AQUI":
        # Modo de teste: exibe no console
        print("\n" + "="*60)
        print(message)
        print("="*60 + "\n")
        return True

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        log.error(f"Erro ao enviar Telegram: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  CAMADA 5 — ORQUESTRADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

# Cache para evitar alertas duplicados no mesmo setup
_alerted_setups: set = set()


def analyze_pair(symbol: str, timeframe: str) -> dict | None:
    """
    Análise SMC completa de um par/timeframe.
    Retorna o melhor setup encontrado ou None.
    """
    df = get_candles(symbol, timeframe)
    if df is None or len(df) < 50:
        return None

    current_price = df["close"].iloc[-1]

    # Executa todos os algoritmos SMC
    swings    = find_swing_points(df)
    eq_levels = detect_equal_levels(swings)
    structure = detect_market_structure(df, swings)
    obs       = detect_order_blocks(df, swings)
    fvgs      = detect_fvg(df)
    sweep     = detect_liquidity_sweep(df, swings, eq_levels)

    if not sweep:
        return None

    # Calcula score de qualidade
    score, reasons = calculate_setup_score(
        sweep, structure, obs, fvgs, eq_levels, timeframe, current_price
    )

    if score < MIN_SCORE:
        return None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "sweep": sweep,
        "score": score,
        "reasons": reasons,
        "current_price": current_price,
        "eq_levels": eq_levels,
    }


def run_scan():
    """
    Ciclo principal do scanner.
    Executa a cada 15 minutos via scheduler.
    """
    log.info("🔍 Iniciando ciclo de scan...")
    session = get_amd_session()
    log.info(f"Sessão atual: {session}")

    pairs = get_top_pairs()
    best_setups = []

    for symbol in pairs:
        for tf in TIMEFRAMES:
            try:
                result = analyze_pair(symbol, tf)
                if result:
                    best_setups.append(result)
                    log.info(f"🎯 Setup encontrado: {symbol}/{tf} — Score {result['score']}/10")
                time.sleep(0.1)  # Rate limiting gentil com a API
            except Exception as e:
                log.warning(f"Erro analisando {symbol}/{tf}: {e}")

    # Ordena pelos melhores setups
    best_setups.sort(key=lambda x: x["score"], reverse=True)

    # Envia alertas (máximo 5 por ciclo para não spammar)
    alerts_sent = 0
    for setup in best_setups[:5]:
        # Gera chave única para evitar re-alertar o mesmo setup
        cache_key = f"{setup['symbol']}_{setup['timeframe']}_{setup['sweep']['sweep_idx']}"
        if cache_key in _alerted_setups:
            continue

        msg = format_alert(
            symbol=setup["symbol"],
            timeframe=setup["timeframe"],
            sweep=setup["sweep"],
            score=setup["score"],
            reasons=setup["reasons"],
            current_price=setup["current_price"],
            eq_levels=setup["eq_levels"],
        )

        if send_telegram(msg):
            _alerted_setups.add(cache_key)
            alerts_sent += 1
            log.info(f"📱 Alerta enviado: {setup['symbol']}/{setup['timeframe']}")

    if alerts_sent == 0:
        log.info("😴 Nenhum setup de alta qualidade encontrado neste ciclo.")
    else:
        log.info(f"✅ {alerts_sent} alerta(s) enviado(s) neste ciclo.")

    # Limpa cache antigo (mantém apenas últimas 500 entradas)
    if len(_alerted_setups) > 500:
        old = list(_alerted_setups)[:200]
        for k in old:
            _alerted_setups.discard(k)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRADA DO PROGRAMA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("🚀 APEX SMC SCANNER iniciado")
    log.info(f"📊 Configuração: Top {TOP_PAIRS_COUNT} pares | Score mínimo: {MIN_SCORE}/10")
    log.info(f"⏱  Timeframes: {', '.join(TIMEFRAMES)} | Scan a cada 15 minutos")

    # Executa imediatamente ao iniciar
    run_scan()

    # Agenda para rodar a cada 15 minutos
    schedule.every(15).minutes.do(run_scan)

    log.info("⏳ Aguardando próximo ciclo (a cada 15min)...")
    while True:
        schedule.run_pending()
        time.sleep(30)
