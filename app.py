import math
import threading
import av
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

st.set_page_config(page_title="MathHand Web", page_icon="∑", layout="wide")
st.markdown("""
<style>
.stApp { background:#080b10; color:#edf2f7; }
[data-testid="stSidebar"] { background:#11161d; }
[data-testid="stMetric"] { background:#171d25; border:1px solid #27313d; padding:12px; border-radius:8px; }
div[data-testid="stMarkdownContainer"] p { color:#d4deea; }
</style>
""", unsafe_allow_html=True)
x = sp.symbols("x", real=True)
SAFE_NAMES = {"x":x,"X":x,"pi":sp.pi,"PI":sp.pi,"e":sp.E,"E":sp.E,"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"cot":sp.cot,"sec":sp.sec,"csc":sp.csc,"asin":sp.asin,"acos":sp.acos,"atan":sp.atan,"sqrt":sp.sqrt,"log":sp.log,"ln":sp.log,"exp":sp.exp,"abs":sp.Abs,"Abs":sp.Abs}

def parse_function(raw):
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    raw = raw.strip().replace("π", "pi").replace("^", "**")
    transforms = standard_transformations + (implicit_multiplication_application, convert_xor)
    return sp.simplify(parse_expr(raw, local_dict=SAFE_NAMES, transformations=transforms, evaluate=True))

def number(expr, value):
    try:
        result = complex(sp.N(expr.subs(x, value)))
        if abs(result.imag) < 1e-8 and math.isfinite(result.real): return float(result.real)
    except Exception: pass
    return None

def fmt(expr):
    try:
        if expr == sp.oo: return "+∞"
        if expr == -sp.oo: return "−∞"
        if expr is sp.nan or expr == sp.nan: return "no determinado"
        return sp.sstr(sp.simplify(expr))
    except Exception: return str(expr)

def sample(expr, lo, hi, n=3000):
    xs = np.linspace(lo, hi, n)
    fn = sp.lambdify(x, expr, modules=["numpy"])
    try:
        with np.errstate(all="ignore"): ys = np.asarray(fn(xs), dtype=float)
        if ys.ndim == 0: ys = np.full(xs.shape, float(ys))
        ys[~np.isfinite(ys)] = np.nan
        ys[np.abs(ys) > 1e6] = np.nan
        jumps = np.abs(np.diff(ys))
        threshold = max(20, np.nanpercentile(jumps[np.isfinite(jumps)],98) if np.isfinite(jumps).any() else 20)
        ys[1:][jumps > threshold] = np.nan
    except Exception: ys = np.full(xs.shape, np.nan)
    return xs, ys

def limits_at(expr, point):
    try:
        left, right = sp.limit(expr,x,point,dir="-"), sp.limit(expr,x,point,dir="+")
        both = fmt(left) if sp.simplify(left-right)==0 else "no existe (los límites laterales difieren)"
        return fmt(left),fmt(right),both
    except Exception: return "no determinado","no determinado","no determinado"

class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock=threading.Lock(); self.gesture=0
        self.hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=0.6,min_tracking_confidence=0.55)
        self.drawer=mp.solutions.drawing_utils
    def recv(self,frame):
        image=cv2.flip(frame.to_ndarray(format="bgr24"),1)
        result=self.hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)); count=0
        if result.multi_hand_landmarks:
            marks=result.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(image,marks,mp.solutions.hands.HAND_CONNECTIONS)
            pts=marks.landmark; tips,pips=(8,12,16,20),(6,10,14,18)
            count=sum(1 for tip,pip in zip(tips,pips) if pts[tip].y<pts[pip].y)
            if abs(pts[4].x-pts[2].x)>0.08: count+=1
        with self.lock: self.gesture=count
        return av.VideoFrame.from_ndarray(image,format="bgr24")

def find_critical_points(expr,derivative):
    found=[]
    try:
        for root in sp.solve(sp.Eq(derivative,0),x):
            val=complex(sp.N(root))
            if abs(val.imag)<1e-8 and -10<=val.real<=10:
                yv=number(expr,val.real)
                if yv is not None: found.append((float(val.real),yv))
    except Exception: pass
    gx,gy=sample(derivative,-10,10,4001); valid=np.isfinite(gy)
    for i in range(len(gx)-1):
        if valid[i] and valid[i+1] and gy[i]*gy[i+1]<0:
            lo,hi=gx[i],gx[i+1]
            for _ in range(40):
                mid=(lo+hi)/2; mv=number(derivative,mid)
                if mv is None: break
                if gy[i]*mv<=0: hi=mid
                else: lo=mid
            px=(lo+hi)/2; py=number(expr,px)
            if py is not None and all(abs(px-old[0])>1e-4 for old in found): found.append((px,py))
    return sorted(found)

st.title("Cálculo en Movimiento — MathHand")
st.caption("Explora cálculo diferencial con gestos de la mano, cámara o controles manuales.")
st_autorefresh(interval=2000,key="mathhand-refresh")
if "func_text" not in st.session_state: st.session_state.func_text="x**3 - 3*x"
if "a" not in st.session_state: st.session_state.a=1.0
with st.form("function_form"):
    left,right=st.columns([5,1])
    with left: function_text=st.text_input("Función f(x)",value=st.session_state.func_text,help="Ejemplos: x^2, sin(x), tan(x), 1/(x-2), pi*x + e")
    with right:
        st.write(""); st.write("")
        apply=st.form_submit_button("Aplicar función",use_container_width=True)
if apply: st.session_state.func_text=function_text
try:
    expr=parse_function(st.session_state.func_text); derivs=[sp.diff(expr,x,i) for i in range(1,4)]
except Exception as exc:
    st.error(f"No pude interpretar la función: {exc}"); st.stop()
camcol,resultcol=st.columns([1,2])
with camcol:
    st.subheader("Cámara y modo")
    st.warning("Al iniciar la cámara, el video se transmite al servidor de MathHand para reconocer los dedos.")
    ctx=webrtc_streamer(key="mathhand-camera",video_processor_factory=HandProcessor,media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":15,"max":20}},"audio":False},rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},async_processing=True)
    st.caption("Permite el acceso a la cámara. La transmisión se procesa para reconocer dedos.")
    st.markdown("**Selecciona el modo manualmente si la cámara no está disponible:**")
    mode=st.radio("Interacción",[1,2,3,4,5],horizontal=True,format_func=lambda m:{1:"1 dedo · Punto",2:"2 dedos · Tangente",3:"3 dedos · Derivadas",4:"4 dedos · Críticos",5:"5 dedos · Límites"}[m],label_visibility="collapsed")
    if ctx.video_processor:
        with ctx.video_processor.lock: detected=ctx.video_processor.gesture
        if detected in (1,2,3,4,5):
            mode=detected; st.success(f"Gesto reconocido: {detected} dedo(s)")
        else: st.info("Muestra de 1 a 5 dedos frente a la cámara.")
with resultcol:
    amin,amax=st.columns([3,1])
    with amin:
        a=st.number_input("Valor de a (número real)",value=float(st.session_state.a),step=0.1,format="%.6g")
        st.session_state.a=a
    with amax: st.metric("Modo",f"{mode} dedo(s)")
    st.subheader("Gráfica de la función")
    xmin,xmax=(-10.,10.)
    if mode==1 and abs(a)>10: xmin,xmax=a-10,a+10
    elif mode==2 and abs(a)>10: xmin,xmax=a-5,a+5
    xs,ys=sample(expr,xmin,xmax); fig=go.Figure()
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",name="f(x)",line=dict(color="#2f9cff",width=3),connectgaps=False))
    fa,slope=number(expr,a),number(derivs[0],a)
    if mode in (1,2) and fa is not None: fig.add_trace(go.Scatter(x=[a],y=[fa],mode="markers",name="(a, f(a))",marker=dict(color="#ff5f66",size=12)))
    if mode==2 and fa is not None and slope is not None:
        tx=np.linspace(max(xmin,a-5),min(xmax,a+5),200)
        fig.add_trace(go.Scatter(x=tx,y=fa+slope*(tx-a),mode="lines",name="Tangente",line=dict(color="#f3c94b",dash="dot",width=2)))
    if mode==3:
        for derivative,color,name in zip(derivs,["#f3c94b","#9b7cff","#35d6d0"],["f′(x)","f″(x)","f‴(x)"]):
            dx,dy=sample(derivative,xmin,xmax)
            fig.add_trace(go.Scatter(x=dx,y=dy,mode="lines",name=name,line=dict(color=color,width=2)))
    if mode==4:
        critical=find_critical_points(expr,derivs[0])
        if critical: fig.add_trace(go.Scatter(x=[p[0] for p in critical],y=[p[1] for p in critical],mode="markers+text",text=[f"({p[0]:.2f}, {p[1]:.2f})" for p in critical],textposition="top center",name="Puntos críticos",marker=dict(color="#ff5f66",size=11)))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#11161d",plot_bgcolor="#0b1016",height=500,margin=dict(l=20,r=20,t=30,b=30),xaxis_title="x",yaxis_title="y",legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Resultado / explicación")
    st.markdown("**f(x) =** " + sp.sstr(expr))
    if mode in (1,2):
        if fa is None: st.warning(f"f({a:g}) no está definida como número real. Elige otro valor para a.")
        else:
            c1,c2,c3,c4=st.columns(4); c1.metric("a",f"{a:.6g}"); c2.metric("f(a)",f"{fa:.6g}"); c3.metric("f′(a)","No definida" if slope is None else f"{slope:.6g}"); c4.metric("Pendiente","No definida" if slope is None else f"{slope:.6g}")
        if mode==2 and fa is not None and slope is not None: st.latex(rf"y = f(a)+f'(a)(x-a) = {sp.N(fa,6)} + {sp.N(slope,6)}(x-({a:g}))")
    elif mode==3:
        for i,d in enumerate(derivs,1):
            val=number(d,a); st.markdown(f"**f^({i})(x) =** {sp.sstr(d)}  **f^({i})(a) =** {val if val is not None else 'no definida'}")
    elif mode==4:
        st.markdown("**Puntos críticos en [-10, 10]:** "+(", ".join(f"({px:.5g}, {py:.5g})" for px,py in critical) if critical else "no se encontraron"))
    elif mode==5:
        try:
            left,right=sp.limit(expr,x,0,dir="-"),sp.limit(expr,x,0,dir="+")
            st.markdown(f"**Límite por la izquierda en 0:** {fmt(left)}  \n**Por la derecha:** {fmt(right)}  \n**Límite bilateral:** {fmt(left) if sp.simplify(left-right)==0 else 'no existe'}")
            st.markdown(f"**Cuando x → +∞:** {fmt(sp.limit(expr,x,sp.oo))}  \n**Cuando x → −∞:** {fmt(sp.limit(expr,x,-sp.oo))}")
        except Exception as exc: st.info(f"No se pudo determinar alguno de los límites ({type(exc).__name__}).")
        singularities=[]
        try:
            _,den=sp.together(expr).as_numer_denom(); solutions=sp.solveset(den,x,domain=sp.Interval(-10,10)) if den!=1 else sp.EmptySet
            if isinstance(solutions,sp.FiniteSet): singularities=list(solutions)
        except Exception: pass
        if singularities:
            st.markdown("**Discontinuidades detectadas entre −10 y 10:**")
            for point in singularities:
                ltxt,rtxt,_=limits_at(expr,point); st.markdown(f"En x = {fmt(point)}: límite izquierdo {ltxt}; derecho {rtxt}.")
        else: st.markdown("No se detectaron polos racionales en [-10, 10].")
st.divider()
st.caption("MathHand Web: usa HTTPS para cámara. El modo manual y el campo a siguen disponibles sin cámara.")
