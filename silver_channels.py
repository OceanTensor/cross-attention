"""
silver_channels.py — 32채널 마스터 정의 (단일 진실원천)
═══════════════════════════════════════════════════════════════════
OceanTensor ST-MMT 큐브의 32채널 확정 정의.
모든 silver_etl_*.py는 이 파일에서 채널 번호·소스를 import한다.
→ 채널 번호 충돌 원천 차단.

확정: 2026-07-07 (32채널정보.docx 기준)
단위 원칙:
  - 영양염(din/dip/sio2/no3/nh4): μmol/L (점관측 μg/L → ÷원자량, Silver 단계)
  - current: cm/s → m/s (÷100)
  - 나머지: 원 단위 유지
═══════════════════════════════════════════════════════════════════
"""

# ─── 32채널 마스터 ───
# 각 채널: (번호, 이름, 소스종류, 소스컬럼, 변환, 단위)
#   소스종류: point_env(v_env_observation/KOEM/NIFS), kma, khoa, derived, satellite, static
#   변환: 단위변환 제수/함수 (None=변환없음)
CHANNEL_MASTER = {
    # ── 점관측 환경 (KOEM/NIFS femo/soo) ──
    "ch00": ("SST",            "point_env", "sst",       None,    "°C"),
    "ch01": ("DIN",            "point_env", "din",       14.007,  "μmol/L"),  # μg/L ÷14.007
    "ch02": ("DIP",            "point_env", "dip",       30.974,  "μmol/L"),
    "ch03": ("SiO2",           "point_env", "sio2",      28.086,  "μmol/L"),
    "ch04": ("DIN_DIP_ratio",  "derived",   None,        None,    "ratio"),
    "ch05": ("Salinity",       "point_env", "salinity",  None,    "psu"),
    "ch06": ("Precipitation",  "kma",       "rn",        None,    "mm"),
    "ch07": ("Solar_Radiation","kma",       "icsr",      None,    "W/m²"),
    "ch08": ("Chlorophyll",    "point_env", "chla",      None,    "mg/m³"),   # =μg/L (동일 수치)
    "ch09": ("DO",             "point_env", "do",        None,    "mg/L"),
    # ── 해류 (KHOA/HYCOM) ──
    "ch10": ("Current_U",      "khoa",      "u",         100.0,   "m/s"),     # cm/s ÷100
    "ch11": ("Current_V",      "khoa",      "v",         100.0,   "m/s"),
    # ── SST 파생 (위성 후) ──
    "ch12": ("SST_anomaly",    "derived",   None,        None,    "°C"),
    "ch13": ("SST_7d_avg",     "derived",   None,        None,    "°C"),
    "ch14": ("Days_Since_Rain","derived",   None,        None,    "days"),
    "ch15": ("Turbidity",      "satellite", None,        None,    "-"),
    # ── 기상 (KMA) ──
    "ch16": ("Wind_Speed",     "kma",       "ws",        None,    "m/s"),
    "ch17": ("Wind_Dir_sin",   "kma",       "wd",        "sin",   "-"),
    "ch18": ("Wind_Dir_cos",   "kma",       "wd",        "cos",   "-"),
    "ch19": ("Air_Temp",       "kma",       "ta",        None,    "°C"),
    "ch20": ("pH",             "point_env", "ph",        None,    "-"),
    "ch21": ("NO3",            "point_env", "no3",       14.007,  "μmol/L"),
    "ch22": ("NH4",            "point_env", "nh4",       14.007,  "μmol/L"),
    "ch23": ("SST_gradient",   "derived",   None,        None,    "°C/km"),
    "ch24": ("NIR_idx",        "satellite", None,        None,    "-"),
    # ── 시간 인코딩 ──
    "ch25": ("Month_sin",      "static",    None,        "sin",   "-"),
    "ch26": ("Month_cos",      "static",    None,        "cos",   "-"),
    "ch27": ("Current_Speed",  "derived",   None,        None,    "m/s"),     # √(u²+v²)
    # ── 시계열 평균 (위성 후) ──
    "ch28": ("Chl_7d_avg",     "derived",   None,        None,    "mg/m³"),
    "ch29": ("NIR_7d_avg",     "derived",   None,        None,    "-"),
    "ch30": ("SST_30d_avg",    "derived",   None,        None,    "°C"),
    "ch31": ("MLD",            "derived",   None,        None,    "m"),
}

# ─── 소스별 채널 그룹 (각 silver_etl 파일이 담당) ───
def channels_by_source(source):
    """특정 소스가 담당하는 채널 리스트 반환."""
    return {ch: v for ch, v in CHANNEL_MASTER.items() if v[1] == source}

POINT_ENV_CHANNELS = channels_by_source("point_env")  # multi/nifs: sst,din,dip,sio2,sal,chla,do,ph,no3,nh4
KMA_CHANNELS       = channels_by_source("kma")         # 강수,일사,풍속,풍향,기온
KHOA_CHANNELS      = channels_by_source("khoa")        # current u,v
DERIVED_CHANNELS   = channels_by_source("derived")     # 비율,anomaly,gradient,평균
SATELLITE_CHANNELS = channels_by_source("satellite")   # turbidity,nir
STATIC_CHANNELS    = channels_by_source("static")      # month sin/cos

# ─── 채널 순서 (큐브 축 순서) ───
CHANNEL_ORDER = [f"ch{i:02d}" for i in range(32)]

# ─── 검증 ───
def validate():
    assert len(CHANNEL_MASTER) == 32, f"채널 수 {len(CHANNEL_MASTER)} ≠ 32"
    for i in range(32):
        ch = f"ch{i:02d}"
        assert ch in CHANNEL_MASTER, f"{ch} 누락"
    print(f"✓ 32채널 마스터 검증 통과")
    print(f"  point_env: {len(POINT_ENV_CHANNELS)}개 {list(POINT_ENV_CHANNELS.keys())}")
    print(f"  kma:       {len(KMA_CHANNELS)}개 {list(KMA_CHANNELS.keys())}")
    print(f"  khoa:      {len(KHOA_CHANNELS)}개 {list(KHOA_CHANNELS.keys())}")
    print(f"  derived:   {len(DERIVED_CHANNELS)}개 {list(DERIVED_CHANNELS.keys())}")
    print(f"  satellite: {len(SATELLITE_CHANNELS)}개 {list(SATELLITE_CHANNELS.keys())}")
    print(f"  static:    {len(STATIC_CHANNELS)}개 {list(STATIC_CHANNELS.keys())}")

if __name__ == "__main__":
    validate()
