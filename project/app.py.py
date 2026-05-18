import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, vit_b_16
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os

# 页面配置
st.set_page_config(page_title="Medical Image Diagnosis Demo", layout="wide")
st.title("🩻 Single-Domain Generalization in Medical Imaging")
st.markdown("**CHEN HONGDA (DC229862) & WANG YUFENG (DC228400)** | Supervisor: Prof. Long Chen | University of Macau")
st.markdown("---")

# ------------------------------
# 全局配置
# ------------------------------
CLASSES_BRAIN = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASSES_EYE = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
IMG_SIZE = 224

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 模型加载函数（需要替换为实际权重路径）
# ------------------------------
@st.cache_resource
def load_model(model_name, student_method, dataset):
    """
    加载训练好的模型权重
    参数:
        model_name: "ResNet-50" 或 "ViT-B"
        student_method: "Chen" 或 "Wang" + 具体方法
        dataset: "Brain MRI" 或 "Eye Fundus"
    返回: model, class_names
    """
    num_classes = 4
    
    if model_name == "ResNet-50":
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:  # ViT-B
        model = vit_b_16(pretrained=False)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    
    # 构建权重路径（请根据实际保存路径修改）
    # 示例格式: weights/brain_Chen_Noise_resnet50.pth
    method_map = {
        ("Chen", "Noise"): "noise",
        ("Chen", "SalfMix"): "salfmix", 
        ("Chen", "Elastic"): "elastic",
        ("Wang", "Fourier/RASS"): "fourier_rass",
        ("Wang", "BiasField"): "biasfield",
        ("Wang", "Motion"): "motion"
    }
    
    method_key = method_map.get((student_method, st.session_state.get("method_detail", "Noise")))
    if method_key is None:
        method_key = "baseline"
    
    # 数据集标识
    dataset_key = "brain" if dataset == "Brain MRI" else "eye"
    model_key = "resnet50" if model_name == "ResNet-50" else "vitb16"
    
    # 权重文件名
    # TODO: 请根据您实际保存的模型路径修改这里
    weight_path = f"weights/{dataset_key}_{student_method}_{method_key}_{model_key}.pth"
    
    if os.path.exists(weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location=device)
            model.load_state_dict(checkpoint)
            st.success(f"✅ 已加载模型: {weight_path}")
        except Exception as e:
            st.warning(f"⚠️ 加载失败: {e}，使用随机权重")
    else:
        st.warning(f"⚠️ 未找到权重文件 {weight_path}，使用随机权重演示")
    
    model = model.to(device)
    model.eval()
    
    class_names = CLASSES_BRAIN if dataset == "Brain MRI" else CLASSES_EYE
    return model, class_names

# ------------------------------
# 图像预处理
# ------------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# ------------------------------
# 预测函数
# ------------------------------
def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    return class_names[pred_idx], probs

# ------------------------------
# 绘制置信度条形图
# ------------------------------
def plot_confidence(probs, class_names):
    fig = go.Figure(data=[
        go.Bar(x=class_names, y=probs * 100, 
               marker_color=['#2ecc71' if p == max(probs) else '#e74c3c' for p in probs],
               text=[f"{p*100:.1f}%" for p in probs], textposition='outside')
    ])
    fig.update_layout(
        title="预测置信度",
        xaxis_title="疾病类别",
        yaxis_title="置信度 (%)",
        yaxis_range=[0, 100],
        height=400
    )
    return fig

# ------------------------------
# 侧边栏：选择配置
# ------------------------------
with st.sidebar:
    st.header("⚙️ 模型配置")
    
    dataset = st.radio("📁 数据集", ["Brain MRI", "Eye Fundus"])
    model_type = st.radio("🧠 模型架构", ["ResNet-50", "ViT-B"])
    
    st.divider()
    st.header("👨‍🎓 训练方法")
    
    student = st.radio("选择学生的方法", ["Chen Hongda", "Wang Yufeng"])
    
    if student == "Chen Hongda":
        method = st.selectbox("增强方法", ["Noise", "SalfMix", "Elastic"])
        st.caption("Chen的方法: 噪声注入 / 显著性混合 / 弹性变形")
    else:
        method = st.selectbox("增强方法", ["Fourier/RASS", "BiasField", "Motion"])
        st.caption("Wang的方法: 傅里叶风格 / 偏置场照明 / 运动伪影")
    
    st.session_state["method_detail"] = method
    
    st.divider()
    st.markdown("### 📌 使用说明")
    st.markdown("""
    1. 上传医学图像
    2. 选择模型和训练方法
    3. 点击"开始诊断"
    4. 查看预测结果和置信度
    """)
    
    st.markdown("---")
    st.caption("⚠️ 演示模式: 如未放置权重文件，将使用随机预测")

# ------------------------------
# 主区域
# ------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传医学图像")
    uploaded_file = st.file_uploader("选择图像文件", type=["png", "jpg", "jpeg", "dcm"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="上传的图像", use_column_width=True)

with col2:
    st.subheader("🔬 诊断结果")
    
    if st.button("🔍 开始诊断", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("请先上传图像")
        else:
            with st.spinner("加载模型中..."):
                model, class_names = load_model(model_type, student, dataset)
            
            with st.spinner("推理中..."):
                image_tensor = preprocess_image(image)
                pred_class, probs = predict(model, image_tensor, class_names)
            
            # 显示结果
            st.success(f"### 预测结果: **{pred_class}**")
            
            # 置信度图表
            fig = plot_confidence(probs, class_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示详细信息
            with st.expander("📊 详细预测概率"):
                for cls, prob in zip(class_names, probs):
                    st.progress(prob, text=f"{cls}: {prob*100:.2f}%")

# ------------------------------
# 底部：模型说明
# ------------------------------
st.divider()
st.subheader("📖 模型说明")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    **Chen Hongda 的方法** (Mix-based & Simulation)
    - **Noise**: 高斯噪声注入，增强鲁棒性
    - **SalfMix**: 显著性区域混合，保留关键病灶特征
    - **Elastic**: 弹性变形，模拟解剖结构变化
    
    **最佳表现**:
    - 脑肿瘤 + ViT-B + Noise: 92.87% Acc
    - 眼病 + ResNet-50 + Noise: 78.43% Acc
    """)

with col_b:
    st.markdown("""
    **Wang Yufeng 的方法** (Acquisition-aware)
    - **Fourier/RASS**: 频域幅度扰动，改变图像风格
    - **BiasField**: 平滑偏置场，模拟光照/磁场变化
    - **Motion**: 运动伪影/模糊，模拟采集过程失真
    
    **最佳表现**:
    - 脑肿瘤 + ResNet-50 + baseline: 94.1% Acc
    - 眼病 + ViT-B + Motion: 45.8% Acc
    """)

st.caption("💡 提示: 将训练好的模型权重文件放入 `weights/` 文件夹即可启用真实推理。格式: `{dataset}_{student}_{method}_{model}.pth`")