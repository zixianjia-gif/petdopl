import streamlit as st
import pandas as pd
import numpy as np
import time

# ==========================================
# 0. 页面配置 (必须放在最前面)
# ==========================================
st.set_page_config(
    page_title="中石化大连院PET解聚工厂", 
    layout="wide", 
    page_icon="🏭"
)

# ==========================================
# 1. 依赖检查与模型加载
# ==========================================
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from scipy.optimize import minimize
except ImportError as e:
    st.error(f"❌ 启动失败：缺少必要库。请在终端运行: pip install plotly scikit-learn scipy pandas numpy streamlit")
    st.stop()

@st.cache_resource
def load_and_train_model():
    """基于真实DOE数据训练模型"""
    data = {
        'Temp': [190, 180, 190, 190, 180, 190, 200, 200, 190, 210, 180, 200, 180, 180, 190],
        'Time': [0.5, 1.0, 2.5, 1.5, 1.0, 1.5, 1.0, 2.0, 1.5, 1.5, 1.0, 2.0, 2.0, 1.0, 3.5],
        'Ratio': [4, 3, 4, 4, 3, 6, 3, 5, 4, 4, 5, 5, 3, 5, 4],
        'Cat':   [0.8, 0.6, 0.8, 0.8, 1.0, 0.8, 0.6, 0.6, 0.8, 0.8, 0.6, 1.0, 0.6, 0.6, 0.8],
        'Yield': [45.20, 62.20, 78.82, 78.93, 71.91, 78.62, 75.42, 83.67, 75.28, 73.50, 23.15, 82.94, 76.24, 12.93, 80.29]
    }
    df = pd.DataFrame(data)
    X = df[['Temp', 'Time', 'Ratio', 'Cat']]
    y = df['Yield']
    
    model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    model.fit(X, y)
    return model, df

# ==========================================
# 2. 数字孪生计算引擎
# ==========================================
class PETFactoryDigitalTwin:
    def __init__(self, model):
        self.model = model
        # --- 基础价格参数 (人民币 ¥) ---
        self.prices = {
            "BHET": 11.0,      # ¥/kg (产品售价)
            "EG": 4.5,         # ¥/kg (原料乙二醇)
            "Catalyst": 25.0,  # ¥/kg (金属锌/乙酸锌)
            "Energy": 0.7      # ¥/kWh (电费)
        }
        
        # --- 原料库 ---
        self.feedstocks = {
            "无色瓶片 (Clear Flakes)":    {"purity": 0.98, "price": 6.0, "sep_difficulty": 1.0},
            "蓝白瓶片 (Blue/White Flakes)": {"purity": 0.96, "price": 5.2, "sep_difficulty": 1.2},
            "油瓶/杂瓶 (Oil Bottles)":      {"purity": 0.90, "price": 4.2, "sep_difficulty": 2.5},
            "聚酯标签纸 (PET Label)":       {"purity": 0.85, "price": 2.8, "sep_difficulty": 3.0},
            "PET/PE复合膜 (Composite Film)": {"purity": 0.68, "price": 2.1, "sep_difficulty": 4.5},
            "PET/Al/PE复合膜 (Al-Film)":     {"purity": 0.58, "price": 1.8, "sep_difficulty": 6.0},
            "废旧服装/废纺 (Textile Waste)":  {"purity": 0.25, "price": 1.0, "sep_difficulty": 8.0}
        }
        
        # LCA 因子
        self.lca = {"elec": 0.5, "eg": 1.2, "avoided": 2.8, "process": 0.2, "catalyst": 5.0}

    def simulate(self, inputs):
        fs_name = inputs['fs_type']
        is_colored = inputs['is_colored']
        fs_props = self.feedstocks[fs_name]
        
        mass_in = inputs['mass']
        temp, time, ratio, cat = inputs['temp'], inputs['time'], inputs['ratio'], inputs['cat']
        eg_recycle = inputs['eg_recycle'] / 100.0
        
        # 预测产率
        pred_yield = self.model.predict(pd.DataFrame([[temp, time, ratio, cat]], columns=['Temp', 'Time', 'Ratio', 'Cat']))[0]
        pred_yield = max(0.0, min(99.9, pred_yield)) 
        
        # 物料平衡
        pet_pure = mass_in * fs_props['purity']
        bhet_theory = pet_pure * 1.323
        bhet_actual = bhet_theory * (pred_yield / 100)
        
        # EG消耗
        eg_in = pet_pure * ratio
        eg_chem_used = bhet_actual * 0.244
        eg_loss = (eg_in - eg_chem_used) * (1 - eg_recycle)
        eg_fresh_needed = eg_chem_used + eg_loss
        
        # 能耗
        total_mass = mass_in + eg_in + (pet_pure * cat/100)
        energy = (total_mass * 2.0 * (temp - 25) / 3600) + (total_mass/1000 * 5.0 * time)
        vol = total_mass / 1100 
        sty = bhet_actual / (vol * time) if time > 0 else 0
        
        sep_idx = fs_props['sep_difficulty'] + ((100-pred_yield)/20)
        if is_colored: sep_idx += 2.0
        
        # 成本计算
        c_raw = mass_in * fs_props['price']
        c_eg = eg_fresh_needed * self.prices['EG']
        c_energy = energy * self.prices['Energy']
        c_cat = (bhet_actual * 0.05) * self.prices['Catalyst'] # 按消耗定额
        
        c_post = 0.0
        if is_colored:
            c_post = (c_raw + c_eg + c_cat + c_energy) * 0.12
        
        total_cost = c_raw + c_eg + c_cat + c_energy + c_post
        revenue = bhet_actual * self.prices['BHET']
        profit = revenue - total_cost
        
        # 碳足迹
        co2_emit = (energy * self.lca['elec']) + \
                   (eg_fresh_needed * self.lca['eg']) + \
                   (mass_in * self.lca['process']) + \
                   (bhet_actual * 0.05 * self.lca['catalyst'])
        if is_colored: co2_emit *= 1.08
        co2_avoid = bhet_actual * self.lca['avoided']
        
        return {
            "yield": pred_yield, "bhet": bhet_actual, "profit": profit,
            "sty": sty, "sep_index": sep_idx, "energy_int": energy/bhet_actual if bhet_actual>0 else 0,
            "co2_net": co2_avoid - co2_emit,
            "costs": {"原料成本": c_raw, "EG溶剂": c_eg, "催化剂(Zn)": c_cat, "能耗": c_energy, "脱色/后处理": c_post}
        }

    def optimize(self, current_inputs, target='profit'):
        """
        AI 优化器升级版：同时优化 温度(x0), 时间(x1), EG比例(x2)
        """
        def objective(x):
            inp = current_inputs.copy()
            # 映射优化变量
            inp['temp'], inp['time'], inp['ratio'] = x[0], x[1], x[2]
            res = self.simulate(inp)
            # 目标函数
            return -res['profit'] if target == 'profit' else -res['co2_net']
            
        # 定义边界: 温度(170-210), 时间(0.5-5.0), EG比例(2.0-8.0)
        bounds = [(170, 210), (0.5, 5.0), (2.0, 8.0)]
        
        # 初始猜测
        x0 = [current_inputs['temp'], current_inputs['time'], current_inputs['ratio']]
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds)
        return res.x # 返回 [opt_temp, opt_time, opt_ratio]

# ==========================================
# 3. 界面 UI 渲染
# ==========================================

try:
    model, raw_data = load_and_train_model()
    twin = PETFactoryDigitalTwin(model)
except Exception as e:
    st.error(f"系统错误: {e}")
    st.stop()

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("## 🏭 SINOPEC DALIAN")
    st.markdown("### 参数控制台")
    st.divider()
    
    st.subheader("1. 原料属性")
    fs_type = st.selectbox("废料类型", list(twin.feedstocks.keys()), index=0)
    
    default_colored = True if any(x in fs_type for x in ["有色", "废纺", "膜", "杂"]) else False
    is_colored = st.checkbox("包含色素/杂质 (需脱色)", value=default_colored)
    mass = st.number_input("投入量 (kg)", 100, 10000, 1000, step=100)
    rec_suggest = 60 if "膜" in fs_type or "废纺" in fs_type else 90
    eg_recycle = st.slider("EG 循环利用率 (%)", 50, 99, rec_suggest)
    
    st.subheader("2. 反应工艺")
    temp = st.slider("温度 (°C)", 170, 215, 195)
    time_h = st.slider("时间 (h)", 0.5, 5.0, 2.5, step=0.1)
    ratio = st.slider("EG/PET 质量比", 2.0, 8.0, 4.0, help="AI优化时会自动调整此参数")
    cat = st.slider("催化剂添加量 (wt%)", 0.2, 1.5, 0.8)
    
    inputs = {
        'fs_type': fs_type, 'is_colored': is_colored, 'mass': mass, 
        'temp': temp, 'time': time_h, 'ratio': ratio, 'cat': cat, 'eg_recycle': eg_recycle
    }

# --- 计算 ---
curr_res = twin.simulate(inputs)

st.markdown("# 中石化大连院PET解聚 — 数字孪生工厂")
st.markdown(f"**工况**: `{fs_type}` | 纯度: {twin.feedstocks[fs_type]['purity']*100}% | 催化剂: 金属锌体系")
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 生产大屏", "🤖 智能寻优", "📈 数据洞察"])

with tab1:
    # 第一行 KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("BHET 净产量", f"{curr_res['bhet']:.1f} kg")
    k2.metric("批次净利润", f"¥{curr_res['profit']:.2f}", delta_color="normal" if curr_res['profit']>0 else "inverse")
    k3.metric("时空产率 (STY)", f"{curr_res['sty']:.2f} kg/m³/h")
    k4.metric("CO2 净减排", f"{curr_res['co2_net']:.1f} kg")
    
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("成本构成 (CNY)")
        fig_pie = px.pie(
            values=list(curr_res['costs'].values()), 
            names=list(curr_res['costs'].keys()), 
            hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c2:
        st.subheader("工艺诊断")
        st.markdown(f"""
        - **转化率**: `{curr_res['yield']:.2f}%`
        - **EG成本**: `¥{curr_res['costs']['EG溶剂']:.0f}`
        - **能耗**: `{curr_res['energy_int']:.2f} kWh/kg`
        """)
        
        # --- 智能亏损分析与建议 ---
        if curr_res['profit'] < 0:
            st.error("⚠️ **当前工艺处于亏损状态！**")
            st.markdown("**潜在原因分析：**")
            reasons = []
            if twin.feedstocks[fs_type]['purity'] < 0.7:
                reasons.append("🔴 **原料品质过低**：有效PET含量太少，导致产出不足以覆盖固定成本。")
            if curr_res['costs']['EG溶剂'] > curr_res['costs']['原料成本']:
                reasons.append("🔴 **溶剂消耗过大**：EG/PET比例可能过高，或循环回收率太低。")
            if curr_res['yield'] < 70:
                reasons.append("🔴 **转化率不足**：反应条件（温度/时间）未达到最佳窗口。")
            
            for r in reasons:
                st.markdown(r)
            if not reasons:
                st.markdown("🔴 **综合成本过高**：建议使用 AI 寻找更经济的配方。")
                
            st.info("👉 **建议操作**：点击上方 **「🤖 智能寻优」** 标签页，让 AI 自动平衡 产率 vs 成本。")
        else:
            st.success("✅ **当前工艺盈利良好**。")

with tab2:
    st.markdown("### 🎯 AI 全参数工艺优化")
    st.info("本模块使用 SLSQP 算法，同时调整 **【温度】、【时间】 和 【EG/PET比例】**，寻找利润最大化的平衡点。")
    
    col_opt1, col_opt2 = st.columns([1, 2])
    with col_opt1:
        target = st.radio("优化目标", ["💰 最大化利润", "🌍 最小化碳排放"])
        if st.button("🚀 启动全维优化", type="primary"):
            with st.spinner("正在遍历参数空间 (Temp, Time, Ratio)..."):
                t_key = 'profit' if "利润" in target else 'co2'
                # 调用优化器
                best_params = twin.optimize(inputs, t_key)
                
                # 模拟最优结果
                opt_inputs = inputs.copy()
                opt_inputs['temp'], opt_inputs['time'], opt_inputs['ratio'] = best_params[0], best_params[1], best_params[2]
                opt_res = twin.simulate(opt_inputs)
                time.sleep(0.8)
                
            st.balloons()
            st.success("✅ 优化完成！")
            
            # 结果展示
            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("推荐温度", f"{best_params[0]:.1f} °C", delta=f"{best_params[0]-temp:.1f}")
            c_res2.metric("推荐时间", f"{best_params[1]:.1f} h", delta=f"{best_params[1]-time_h:.1f}")
            c_res3.metric("推荐 EG/PET比", f"{best_params[2]:.1f}", delta=f"{best_params[2]-ratio:.1f}")
            
            st.markdown("#### 💡 优化效果对比")
            col_d1, col_d2 = st.columns(2)
            gain = opt_res['profit'] - curr_res['profit']
            col_d1.metric("利润提升", f"¥{gain:.2f}")
            col_d2.metric("优化后产率", f"{opt_res['yield']:.1f}%")
            
            if opt_res['profit'] < 0:
                st.warning("⚠️ 注意：即使在AI优化后，该低价值原料仍难以盈利。建议：1. 提高EG循环率至95%以上；2. 压低原料采购价。")

with tab3:
    st.markdown("**3D 响应面：温度 vs EG配比 vs 产率**")
    t_rng = np.linspace(170, 210, 25)
    r_rng = np.linspace(2, 6, 25)
    X_g, Y_g = np.meshgrid(t_rng, r_rng)
    Z_g = np.zeros_like(X_g)
    for i in range(25):
        for j in range(25):
            Z_g[i,j] = model.predict(pd.DataFrame([[X_g[i,j], time_h, Y_g[i,j], cat]], columns=['Temp', 'Time', 'Ratio', 'Cat']))[0]
    
    fig3d = go.Figure(data=[go.Surface(z=Z_g, x=t_rng, y=r_rng, colorscale='Tealgrn')])
    fig3d.update_layout(scene=dict(xaxis_title='Temp', yaxis_title='Ratio', zaxis_title='Yield'), height=500, margin=dict(l=0, r=0, b=0, t=0))
    fig3d.add_trace(go.Scatter3d(x=[temp], y=[ratio], z=[curr_res['yield']], mode='markers', marker=dict(size=8, color='red'), name='Current'))
    st.plotly_chart(fig3d, use_container_width=True)