"""
=============================================================================
style.py — Global Streamlit CSS Injector
=============================================================================
Injects custom CSS for professional UI with theme support (Light/Dark).
=============================================================================
"""

import streamlit as st

def get_css(theme='dark'):
    if theme == 'light':
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;700;900&display=swap');
        :root {
          --primary   : #4F46E5;
          --secondary : #DB2777;
          --accent    : #0D9488;
          --success   : #16A34A;
          --warning   : #D97706;
          --danger    : #DC2626;
          --bg-color  : #F8FAFC;
          --card-bg   : #FFFFFF;
          --card2-bg  : #F1F5F9;
          --border    : #E2E8F0;
          --text      : #1E293B;
          --muted     : #64748B;
        }
        html, body, [class*="css"] {
          font-family: 'Inter', sans-serif !important;
          background-color: var(--bg-color) !important;
          color: var(--text) !important;
        }
        .stApp { background: var(--bg-color) !important; }
        [data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid var(--border) !important; }
        [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: var(--text) !important; }
        .hero-banner {
          background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 40%, #C7D2FE 100%);
          border-radius: 18px; padding: 3rem 2.5rem; margin-bottom: 2rem; position: relative; overflow: hidden;
          box-shadow: 0 10px 30px rgba(79,70,229,0.1); animation: bannerFadeIn 0.7s ease-out;
        }
        .hero-title {
          font-family: 'Outfit', sans-serif !important; font-size: 2.6rem; font-weight: 900;
          background: linear-gradient(135deg, #312E81 0%, #4F46E5 100%);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
        }
        .hero-subtitle { font-size: 1.05rem; color: #475569; font-weight: 500; line-height: 1.6; }
        .metric-card {
          background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px;
          padding: 1.5rem; text-align: center; transition: transform 0.2s ease, box-shadow 0.2s ease;
          box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }
        .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 20px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2rem; font-weight: 800; color: var(--primary); }
        .metric-label { font-size: 0.85rem; color: var(--muted); margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .section-header {
          font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--text);
          padding-bottom: 0.5rem; border-bottom: 2px solid var(--primary); margin: 2rem 0 1.2rem;
        }
        .info-card {
          background: var(--card2-bg); border-left: 4px solid var(--primary); border-radius: 10px;
          padding: 1.2rem 1.5rem; margin-bottom: 1rem; color: var(--text);
        }
        .status-green  { background: #DCFCE7; border: 1px solid #22C55E; border-radius: 10px; padding: 1rem; color: #166534; }
        .status-orange { background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 10px; padding: 1rem; color: #92400E; }
        .status-red    { background: #FEE2E2; border: 1px solid #EF4444; border-radius: 10px; padding: 1rem; color: #991B1B; }
        .status-dark   { background: #7F1D1D; border: 1px solid #450A0A; border-radius: 10px; padding: 1rem; color: #FEF2F2; }
        .stButton > button {
          background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
          color: white !important; border: none !important; border-radius: 10px !important;
          padding: 0.65rem 1.8rem !important; font-weight: 600 !important;
        }
        [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] {
          background: #FFFFFF !important; border: 1px solid var(--border) !important; color: var(--text) !important;
        }
        .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 99px; font-size: 0.78rem; font-weight: 600; }
        .badge-primary { background: #E0E7FF; color: #3730A3; }
        .badge-success { background: #DCFCE7; color: #166534; }
        .badge-danger  { background: #FEE2E2; color: #991B1B; }
        .badge-warning { background: #FEF3C7; color: #92400E; }
        .custom-footer { margin-top: 4rem; padding: 1.5rem; border-top: 1px solid var(--border); text-align: center; color: var(--muted); font-size: 0.85rem; }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;700;900&display=swap');
        :root {
          --primary   : #6366F1;
          --secondary : #EC4899;
          --accent    : #14B8A6;
          --dark      : #0F0F1A;
          --card-bg   : #1A1A2E;
          --card2-bg  : #16213E;
          --border    : #2D2D55;
          --text      : #E2E8F0;
          --muted     : #94A3B8;
        }
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: var(--dark) !important; color: var(--text) !important; }
        .stApp { background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #0D0D1B 100%) !important; }
        #MainMenu, footer, header { visibility: hidden; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #0D0D1B 0%, #1A1A2E 100%) !important; border-right: 1px solid var(--border) !important; }
        [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: var(--text) !important; }
        .hero-banner {
          background: linear-gradient(135deg, #1E1B4B 0%, #312E81 40%, #6366F1 100%); border-radius: 18px; padding: 3rem 2.5rem; margin-bottom: 2rem; position: relative; overflow: hidden;
          box-shadow: 0 20px 60px rgba(99,102,241,0.35); animation: bannerFadeIn 0.7s ease-out;
        }
        .hero-title {
          font-family: 'Outfit', sans-serif !important; font-size: 2.6rem; font-weight: 900; background: linear-gradient(135deg, #fff 0%, #C7D2FE 100%);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
        }
        .hero-subtitle { font-size: 1.05rem; color: rgba(255,255,255,0.75); font-weight: 400; line-height: 1.6; }
        .metric-card {
          background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; text-align: center; transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(99,102,241,0.25); }
        .metric-value { font-size: 2rem; font-weight: 800; color: var(--primary); }
        .metric-label { font-size: 0.85rem; color: var(--muted); margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .section-header { font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; color: #fff; padding-bottom: 0.5rem; border-bottom: 2px solid var(--primary); margin: 2rem 0 1.2rem; }
        .info-card { background: var(--card2-bg); border-left: 4px solid var(--primary); border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
        .status-green  { background: rgba(34,197,94,0.12);  border: 1px solid #22C55E; border-radius: 10px; padding: 1rem; }
        .status-orange { background: rgba(245,158,11,0.12); border: 1px solid #F59E0B; border-radius: 10px; padding: 1rem; }
        .status-red    { background: rgba(239,68,68,0.12);  border: 1px solid #EF4444; border-radius: 10px; padding: 1rem; }
        .status-dark   { background: rgba(239,68,68,0.25);  border: 1px solid #B91C1C; border-radius: 10px; padding: 1rem; }
        .stButton > button {
          background: linear-gradient(135deg, var(--primary), var(--secondary)) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 1.8rem !important; font-weight: 600 !important;
        }
        [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] { background: #1A1A2E !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }
        [data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; }
        .js-plotly-plot .plotly, .plot-container { background: transparent !important; }
        .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 99px; font-size: 0.78rem; font-weight: 600; }
        .badge-primary { background: rgba(99,102,241,0.25); color: #A5B4FC; }
        .badge-success { background: rgba(34,197,94,0.25);  color: #86EFAC; }
        .badge-danger  { background: rgba(239,68,68,0.25);  color: #FCA5A5; }
        .badge-warning { background: rgba(245,158,11,0.25); color: #FCD34D; }
        .custom-footer { margin-top: 4rem; padding: 1.5rem; border-top: 1px solid var(--border); text-align: center; color: var(--muted); font-size: 0.85rem; }
        </style>
        """

def inject_css() -> None:
    theme = st.session_state.get('theme', 'dark')
    st.markdown(get_css(theme), unsafe_allow_html=True)

def hero_banner(title: str, subtitle: str = '', emoji: str = '⚙️') -> None:
    st.markdown(f"""
    <div class="hero-banner">
      <div style="font-size:3.5rem;margin-bottom:0.5rem;">{emoji}</div>
      <div class="hero-title">{title}</div>
      <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def metric_card(value, label: str, delta: str = '') -> str:
    delta_html = f'<div style="font-size:0.8rem;color:var(--muted);margin-top:4px">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
      <div class="metric-value">{value}</div>
      <div class="metric-label">{label}</div>
      {delta_html}
    </div>
    """

def footer() -> None:
    st.markdown("""
    <div class="custom-footer">
      🔧 CNC Predictive Maintenance System &nbsp;|&nbsp; Built with ❤️ using Python & Streamlit &nbsp;|&nbsp; Final Year Engineering Project 2025
    </div>
    """, unsafe_allow_html=True)
