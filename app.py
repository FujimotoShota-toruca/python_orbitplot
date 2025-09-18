"""
軌道可視化ツール
使い方:
1) `pip install streamlit plotly numpy`
2) `streamlit run app.py`
"""

import math
from math import sin, cos, sqrt
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 物理定数（km, s, rad）
MU_EARTH = 398600.4418  # 地球重力定数 [km^3/s^2]
R_EARTH = 6378.137      # 地球半径（球近似）[km]
OMEGA_EARTH = 7.2921159e-5  # 地球自転角速度 [rad/s]

DEG2RAD = math.pi/180.0
RAD2DEG = 180.0/math.pi

# ------------------------------
# 軌道計算ユーティリティ
# ------------------------------

def kepler_E_from_M(M: np.ndarray, e: float, tol: float = 1e-10, itmax: int = 30) -> np.ndarray:
    """離心近点角Eをケプラー方程式 M = E - e*sinE から求める（ニュートン法）。"""
    M = np.mod(M, 2*np.pi)
    if e < 1e-12:
        return M.copy()
    E = np.where(e < 0.8, M, np.pi*np.ones_like(M))
    for _ in range(itmax):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f/fp
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def coe_to_r_eci(a: float, e: float, i_deg: float, raan_deg: float, argp_deg: float,
                 M0_deg: float, t: np.ndarray) -> np.ndarray:
    """クラシカル6要素 + M0（平均近点離角）からECI位置ベクトルr(t) [km]を返す。"""
    i = i_deg*DEG2RAD
    Omega = raan_deg*DEG2RAD
    omega = argp_deg*DEG2RAD
    M0 = M0_deg*DEG2RAD

    n = sqrt(MU_EARTH/a**3)  # 平均運動
    M = M0 + n*t
    E = kepler_E_from_M(M, e)

    cosE = np.cos(E); sinE = np.sin(E)
    r_pf_x = a*(cosE - e)
    r_pf_y = a*sqrt(1 - e**2)*sinE
    r_pf = np.vstack([r_pf_x, r_pf_y, np.zeros_like(r_pf_x)])  # 3×N

    cO, sO = cos(Omega), sin(Omega)
    ci, si = cos(i), sin(i)
    co, so = cos(omega), sin(omega)

    R3_O = np.array([[ cO, -sO, 0],[ sO,  cO, 0],[ 0, 0, 1]])
    R1_i = np.array([[ 1, 0, 0],[ 0, ci, -si],[ 0, si,  ci]])
    R3_o = np.array([[ co, -so, 0],[ so,  co, 0],[ 0, 0, 1]])
    Q = R3_O @ R1_i @ R3_o
    r_eci = Q @ r_pf  # 3×N
    return r_eci.T  # N×3


def eci_to_ecef(r_eci: np.ndarray, t: np.ndarray, theta0: float = 0.0) -> np.ndarray:
    """ECI -> ECEF 変換（簡略：地球自転のみ）。theta(t) = theta0 + ω⊕ t"""
    theta = theta0 + OMEGA_EARTH * t
    c = np.cos(theta); s = np.sin(theta)
    r_ecef = np.empty_like(r_eci)
    r_ecef[:, 0] =  c * r_eci[:, 0] + s * r_eci[:, 1]
    r_ecef[:, 1] = -s * r_eci[:, 0] + c * r_eci[:, 1]
    r_ecef[:, 2] =  r_eci[:, 2]
    return r_ecef


def ecef_to_llh_spherical(r_ecef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF座標 -> 球体地球の緯度[deg]/経度[deg]/高度[km]。経度は -180..180 に正規化。"""
    x, y, z = r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hyp))
    rmag = np.sqrt(x**2 + y**2 + z**2)
    alt = rmag - R_EARTH
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon, alt

# ------------------------------
# UI（Streamlit）
# ------------------------------

st.set_page_config(page_title="軌道6要素可視化(ECI/ECEF/地図)", layout="wide")
st.title("軌道6要素可視化：ECI / ECEF / 地図")

# 利用規約等
with st.expander("このツールの仮定及び注意事項・利用規約"):
    st.markdown(
        """
- **利用規約**
    - 本アプリは教育・デモ目的で提供されます（設計・運用・航法等に利用しないでください）。
    - 出力結果の正確性・有用性・特定目的適合性を保証しません。
    - 本アプリの利用により生じたいかなる損害についても、製作者は一切の責任を負いません。
    - **禁止事項**：
        - 不正アクセス負荷試験やスクレイピング等の過度なリクエスト
        - 法令違反
        - HFのContent Policyに反する行為。
    - 本アプリ内の第三者コンテンツ（地図・画像・ライブラリ等）は各ライセンスに従います。必要な帰属表示を削除しないでください。
    - 準拠法：日本法
    - お問い合わせ：TBD
    - ※本アプリは Hugging Face Spaces 上で提供され、非永続ストレージが利用されます。
- **プライバシー・ポリシー**
    - 本アプリは個人情報を能動的に収集しません。
    - 入力は計算・可視化のためにのみ使用され、継続保存しません。
    - アクセスログ等の技術情報は、Hugging Face のインフラにより取得・保管される場合があります
        - 詳細はHugging Faceのプライバシーポリシー参照。
    - 当方で外部解析ツールは使用していません。
    - お問い合わせ:TBD
        """
    )

with st.sidebar:
    st.header("軌道6要素")
    a = st.slider("軌道長半径 a [km]", min_value=6578, max_value=50000, value=6878, step=10,
                  help="地表から~500 kmの円軌道 ≒ 6878 km くらいが目安")
    e = st.slider("離心率 e", min_value=0.0, max_value=0.9, value=0.001, step=0.001)
    i_deg = st.slider("軌道傾斜角 i [deg]", min_value=0.0, max_value=180.0, value=51.6, step=0.1)
    raan_deg = st.slider("昇交点赤経 Ω [deg]", min_value=0.0, max_value=360.0, value=0.0, step=0.5)
    argp_deg = st.slider("近地点引数 ω [deg]", min_value=0.0, max_value=360.0, value=0.0, step=0.5)
    M0_deg = st.slider("初期平均近点離角 M0 [deg]", min_value=0.0, max_value=360.0, value=0.0, step=1.0)

    st.header("伝搬・描画")
    hours = st.slider("描画時間 [hours]", min_value=0.2, max_value=48.0, value=6.0, step=0.2)
    dt = st.slider("タイムステップ dt [s]", min_value=5, max_value=600, value=60, step=5)

    st.header("表示切替")
    show_eci = st.checkbox("ECI表示", value=True)
    show_ecef = st.checkbox("ECEF表示", value=True)
    show_map = st.checkbox("地図表示（メルカトル）", value=True)
    st.caption("※ 3つともONで一括表示")

# 時間配列と軌道
T_end = int(hours*3600)
N = max(2, int(T_end//dt)+1)
t = np.linspace(0.0, T_end, N)

r_eci = coe_to_r_eci(a, e, i_deg, raan_deg, argp_deg, M0_deg, t)
r_ecef = eci_to_ecef(r_eci, t, theta0=0.0)
lat, lon, alt = ecef_to_llh_spherical(r_ecef)

# 表示時刻の選択（現在位置マーカー用）
with st.sidebar:
    st.header("表示時刻")
    k = st.slider("インデックス k (0…N-1)", min_value=0, max_value=N-1, value=min(N-1, N//4))
    st.caption(f"t = {t[k]:.0f} s, 高度 ≈ {alt[k]:.0f} km")

# ------------------------------
# 図の生成
# ------------------------------

def make_earth_sphere(radius=R_EARTH, nu=48, nv=24):
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(-np.pi/2, np.pi/2, nv)
    uu, vv = np.meshgrid(u, v)
    x = radius*np.cos(vv)*np.cos(uu)
    y = radius*np.cos(vv)*np.sin(uu)
    z = radius*np.sin(vv)
    return x, y, z

figs = []

# 共通の色（黄緑＆赤）
ORBIT_COLOR = "mediumspringgreen"   # ★ 軌道線の色（黄緑）
MARKER_COLOR = "orangered"        # ★ 現在位置の色（赤）

# ECI 3D 図
if show_eci:
    x, y, z = make_earth_sphere()
    fig_eci = go.Figure()
    fig_eci.add_surface(x=x, y=y, z=z, opacity=0.6, showscale=False)
    fig_eci.add_trace(go.Scatter3d(
        x=r_eci[:,0], y=r_eci[:,1], z=r_eci[:,2], mode="lines",
        name="軌道(ECI)",
        line=dict(width=4, color=ORBIT_COLOR)  # ★ 色指定
    ))
    fig_eci.add_trace(go.Scatter3d(
        x=[r_eci[k,0]], y=[r_eci[k,1]], z=[r_eci[k,2]], mode="markers",
        marker=dict(size=6, color=MARKER_COLOR),  # ★ 色指定
        name="衛星(ECI)"
    ))
    fig_eci.update_layout(scene=dict(aspectmode="data"), title="ECI座標系（慣性系）")
    figs.append(("ECI", fig_eci))

# ECEF 3D 図
if show_ecef:
    x, y, z = make_earth_sphere()
    fig_ecef = go.Figure()
    fig_ecef.add_surface(x=x, y=y, z=z, opacity=0.6, showscale=False)
    fig_ecef.add_trace(go.Scatter3d(
        x=r_ecef[:,0], y=r_ecef[:,1], z=r_ecef[:,2], mode="lines",
        name="軌道(ECEF)",
        line=dict(width=4, color=ORBIT_COLOR)  # ★ 色指定
    ))
    fig_ecef.add_trace(go.Scatter3d(
        x=[r_ecef[k,0]], y=[r_ecef[k,1]], z=[r_ecef[k,2]], mode="markers",
        marker=dict(size=6, color=MARKER_COLOR),  # ★ 色指定
        name="衛星(ECEF)"
    ))
    fig_ecef.update_layout(scene=dict(aspectmode="data"), title="ECEF座標系（地球固定）")
    figs.append(("ECEF", fig_ecef))

# 地図（メルカトル：海/陸カラー）
if show_map:
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(
        lat=lat, lon=lon, mode="lines",
        name="軌道",
        line=dict(color=ORBIT_COLOR, width=2)  # ★ 色指定
    ))
    fig_map.add_trace(go.Scattergeo(
        lat=[lat[k]], lon=[lon[k]], mode="markers",
        name="現在位置",
        marker=dict(size=8, color=MARKER_COLOR)  # ★ 色指定
    ))

    fig_map.update_geos(
        projection_type="mercator",
        showcountries=True,
        showcoastlines=True,
        showland=True,  landcolor="tan",  # 陸
        showocean=True, oceancolor="steelblue",  # 海
        lataxis_showgrid=True,
        lonaxis_showgrid=True,
    )
    fig_map.update_layout(title="地図表示（Mercator：海/陸カラー）")
    figs.append(("Map", fig_map))

# 凡例の配置
LEGEND_BOTTOM = dict(
    orientation="h",
    x=0.5, xanchor="center",
    y=-0.18, yanchor="top"
)
for _, fig in figs:
    fig.update_layout(legend=LEGEND_BOTTOM, margin=dict(b=110))

# レイアウト配置
cols = st.columns(len(figs)) if figs else []
for col, (name, fig) in zip(cols, figs):
    with col:
        st.plotly_chart(fig, use_container_width=True)

# 追加情報
with st.expander("このツールの仮定"):
    st.markdown(
        """
- **仮定**:
    - 二体問題
    - 球体地球
    - 固定自転角速度
    - GMST
    - 初期角=0（ECI→ECEF）
        """
    )