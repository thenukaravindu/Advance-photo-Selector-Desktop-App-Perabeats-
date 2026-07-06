import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import shutil
import io
import hashlib
import concurrent.futures
import streamlit as st
import tkinter as tk
from tkinter import filedialog

# Streamlit Page Configuration
st.set_page_config(
    page_title="PERABEATS Photo Selecter",
    page_icon="perabeats_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0e1117;
    }
    
    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1e1e38 0%, #0d0d1e 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #3b3b6d;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        color: #ffffff;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #9e9ec5;
        font-size: 1.15rem;
        font-weight: 400;
    }
    
    /* Image cards */
    .photo-card {
        background: #161a24;
        border-radius: 12px;
        border: 1px solid #2d3345;
        padding: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transition: transform 0.2s, border-color 0.2s;
    }
    .photo-card:hover {
        transform: translateY(-4px);
        border-color: #4f46e5;
    }
    
    /* Stat cards */
    .stat-container {
        background: #1e2230;
        border-radius: 12px;
        border: 1px solid #2f364d;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #6366f1;
        margin-bottom: 5px;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #8c95b2;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Responsive custom grids */
    .gallery-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        grid-gap: 20px;
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache Directory Setup
CACHE_DIR = os.path.join(os.getcwd(), ".photo_cully_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# --- NIMA PyTorch Architecture ---
class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        base_model = models.mobilenet_v2(weights=None)
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )
            
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.classifier(x)
        return x

# --- Cache Model Initialization ---
@st.cache_resource
def get_nima_model(use_local_weights=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "epoch-82.pth"
    model = None
    model_type = "None"
    
    if use_local_weights and os.path.exists(weights_path):
        try:
            import pyiqa
            model = pyiqa.create_metric('nima', device=device)
            checkpoint = torch.load(weights_path, map_location=device)
            if 'params' in checkpoint:
                model.net.load_state_dict(checkpoint['params'])
            else:
                model.net.load_state_dict(checkpoint)
            model.eval()
            model_type = "Local weights (epoch-82.pth)"
        except Exception as e:
            model = None
            print(f"Error loading local weights in Streamlit: {e}")
            
    if model is None:
        try:
            import pyiqa
            model = pyiqa.create_metric('nima', device=device)
            model_type = "Pre-trained PyIQA NIMA"
        except Exception as e:
            st.error(f"Failed to load PyIQA: {e}")
            
    return model, model_type, device

# --- Image Loading Helper ---
def load_image(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    raw_extensions = {'.cr2', '.nef', '.arw', '.dng', '.cr3', '.orf', '.rw2', '.pef'}
    
    if ext in raw_extensions:
        import rawpy
        try:
            with rawpy.imread(img_path) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        return Image.open(io.BytesIO(thumb.data))
                except Exception:
                    pass
                
                try:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                except Exception:
                    rgb = raw.postprocess(half_size=True)
                return Image.fromarray(rgb)
        except Exception as e:
            raise RuntimeError(f"Failed to decode RAW image: {e}")
    else:
        return Image.open(img_path).convert('RGB')

# --- NIMA Transform & Score Calculation ---
nima_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_mean_score(output):
    probs = output.cpu().numpy()[0]
    weights = np.arange(1, 11)
    return np.sum(probs * weights)

# --- UI Session State Initializations ---
if 'input_folder' not in st.session_state:
    st.session_state['input_folder'] = r"D:\Club_Photos\Event_Bulk"
    if not os.path.exists(st.session_state['input_folder']):
        st.session_state['input_folder'] = ""

if 'results' not in st.session_state:
    st.session_state['results'] = []

if 'selections' not in st.session_state:
    st.session_state['selections'] = {}

# Callback to update checkboxes when slider changes
def update_selections_by_threshold():
    t = st.session_state['threshold']
    for item in st.session_state['results']:
        st.session_state['selections'][item['name']] = item['score'] >= t

# Callback to handle check/uncheck actions in UI
def on_checkbox_change(name):
    st.session_state['selections'][name] = st.session_state[f"keep_{name}"]

# --- GUI Layout ---

import base64
logo_html = '<div class="header-title">📸 PERABEATS Photo Selecter</div>'
if os.path.exists("perabeats_logo.png"):
    try:
        with open("perabeats_logo.png", "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="width: 140px; height: 140px; border-radius: 50%; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4); border: 2px solid #6366f1; object-fit: cover;"><div class="header-title">PERABEATS Photo Selecter</div>'
    except Exception:
        pass

# Title Header
st.markdown(f'<div class="header-card">{logo_html}<div class="header-subtitle">Premium AI-Powered Photo Selection & Aesthetic Assessment</div></div>', unsafe_allow_html=True)

# Sidebar Configuration
if os.path.exists("perabeats_logo.png"):
    st.sidebar.image("perabeats_logo.png", use_container_width=True)
st.sidebar.header("⚙️ Configuration")

# Model Selection
model_choice = st.sidebar.selectbox(
    "Select Model weights",
    ["Local Weights (epoch-82.pth)", "PyIQA Model Hub"],
    index=0
)
use_local = (model_choice == "Local Weights (epoch-82.pth)")

# Load Model
model, loaded_type, device = get_nima_model(use_local_weights=use_local)

st.sidebar.info(f"**Loaded Model:** {loaded_type}\n\n**Device:** {device}")

# Clear cache button
if st.sidebar.button("🧹 Clear Thumbnail Cache"):
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    st.sidebar.success("Cache cleared!")

# Folder Selector Section
st.subheader("📂 Select Photos Directory")
col_path, col_browse = st.columns([5, 1])

with col_path:
    input_folder_input = st.text_input(
        "Photos Directory Path",
        value=st.session_state['input_folder'],
        placeholder="Enter path or click Browse",
        label_visibility="collapsed"
    )
    if input_folder_input != st.session_state['input_folder']:
        st.session_state['input_folder'] = input_folder_input

with col_browse:
    if st.button("Browse Folder 📁", use_container_width=True):
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        selected = filedialog.askdirectory(master=root)
        root.destroy()
        if selected:
            st.session_state['input_folder'] = os.path.normpath(selected)
            st.rerun()

# Run AI Assessment button
valid_extensions = {
    '.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp',
    '.cr2', '.nef', '.arw', '.dng', '.cr3', '.orf', '.rw2', '.pef'
}

run_disabled = not (st.session_state['input_folder'] and os.path.exists(st.session_state['input_folder']))

if st.button("🚀 Run AI Aesthetic Assessment", type="primary", disabled=run_disabled, use_container_width=True):
    folder = st.session_state['input_folder']
    try:
        all_files = os.listdir(folder)
    except Exception as e:
        st.error(f"Error reading folder: {e}")
        all_files = []

    valid_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions]

    if not valid_files:
        st.warning(f"No supported images found in '{folder}'")
    else:
        st.session_state['results'] = []
        st.session_state['selections'] = {}
        
        # Parallel Image Loading & Processing Setup
        max_workers = min(os.cpu_count() or 4, 6)
        preload_limit = max_workers * 2
        
        file_list = list(valid_files)
        futures = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_items = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit initial batch
            for i in range(min(len(file_list), preload_limit)):
                img_name = file_list[i]
                img_path = os.path.join(folder, img_name)
                futures[executor.submit(load_image, img_path)] = (i, img_name)
                
            next_idx = preload_limit
            
            completed = 0
            while futures:
                done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                
                for f in done:
                    idx, img_name = futures.pop(f)
                    try:
                        img = f.result()
                        
                        # Save thumbnail to cache
                        h = hashlib.md5(os.path.join(folder, img_name).encode('utf-8')).hexdigest()
                        thumb_path = os.path.join(CACHE_DIR, f"{h}.jpg")
                        if not os.path.exists(thumb_path):
                            img_thumb = img.copy()
                            img_thumb.thumbnail((300, 300))
                            img_thumb.save(thumb_path, "JPEG", quality=85)
                            img_thumb.close()
                        
                        # Inference
                        with torch.no_grad():
                            score = model(img).item()
                        
                        processed_items.append({
                            'name': img_name,
                            'score': score,
                            'thumb_path': thumb_path
                        })
                        
                        img.close()
                    except Exception as e:
                        st.sidebar.error(f"Error processing {img_name}: {e}")
                    
                    completed += 1
                    progress_bar.progress(completed / len(file_list))
                    status_text.text(f"Assessed {completed}/{len(file_list)} images...")
                    
                    # Submit next file
                    if next_idx < len(file_list):
                        n_name = file_list[next_idx]
                        n_path = os.path.join(folder, n_name)
                        futures[executor.submit(load_image, n_path)] = (next_idx, n_name)
                        next_idx += 1
                        
        progress_bar.empty()
        status_text.empty()
        
        # Save results to session state
        st.session_state['results'] = sorted(processed_items, key=lambda x: x['score'], reverse=True)
        # Default selections based on threshold 5.0
        for item in st.session_state['results']:
            st.session_state['selections'][item['name']] = item['score'] >= 5.0
            
        st.success(f"Successfully processed {len(st.session_state['results'])} photos!")
        st.rerun()

# --- Main Results Display ---
if st.session_state['results']:
    results = st.session_state['results']
    
    st.write("---")
    st.subheader("📊 Assessment Summary")
    
    # Calculate quick stats
    scores = [item['score'] for item in results]
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score_val = np.min(scores)
    
    c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
    with c_stat1:
        st.markdown(f'<div class="stat-container"><div class="stat-val">{len(results)}</div><div class="stat-label">Total Scanned</div></div>', unsafe_allow_html=True)
    with c_stat2:
        st.markdown(f'<div class="stat-container"><div class="stat-val">{avg_score:.2f}</div><div class="stat-label">Average Score</div></div>', unsafe_allow_html=True)
    with c_stat3:
        st.markdown(f'<div class="stat-container"><div class="stat-val">{max_score:.2f}</div><div class="stat-label">Highest Score</div></div>', unsafe_allow_html=True)
    with c_stat4:
        st.markdown(f'<div class="stat-container"><div class="stat-val">{min_score_val:.2f}</div><div class="stat-label">Lowest Score</div></div>', unsafe_allow_html=True)
        
    st.write(" ")
    
    # Controls Section
    st.subheader("🎛️ Filter & Culling Rules")
    col_slider, col_sort = st.columns([3, 1])
    
    with col_slider:
        threshold = st.slider(
            "Minimum Quality Threshold",
            1.0, 10.0,
            value=5.0,
            step=0.1,
            key="threshold",
            on_change=update_selections_by_threshold
        )
        
    with col_sort:
        sort_option = st.selectbox(
            "Sort images by",
            ["Score (High to Low)", "Score (Low to High)", "Filename"]
        )

    # Sort results
    if sort_option == "Score (High to Low)":
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    elif sort_option == "Score (Low to High)":
        sorted_results = sorted(results, key=lambda x: x['score'])
    else:
        sorted_results = sorted(results, key=lambda x: x['name'])

    # Display Photo Grid
    st.write(" ")
    
    selected_count = sum(1 for name, val in st.session_state['selections'].items() if val)
    discarded_count = len(results) - selected_count
    
    st.markdown(f"#### 🖼️ Image Selection Grid (Selected: `{selected_count}` | Discarded: `{discarded_count}`)")
    st.caption("Tip: You can manually check/uncheck any photo card to override the AI's selection decision.")
    
    cols = st.columns(4)
    for idx, item in enumerate(sorted_results):
        name = item['name']
        score = item['score']
        thumb = item['thumb_path']
        
        # Color score badge based on quality tier
        badge_color = "#10B981" if score >= 6.5 else ("#F59E0B" if score >= 5.0 else "#EF4444")
        
        # Checked status
        is_checked = st.session_state['selections'].get(name, False)
        
        # Render Card
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="photo-card">
                <div style="text-align: center; margin-bottom: 8px;">
                    <span style="background-color: {badge_color}; color: white; padding: 3px 8px; border-radius: 20px; font-size: 0.85rem; font-weight: bold;">
                        Score: {score:.2f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display cached thumbnail
            st.image(thumb, use_container_width=True)
            
            # Text layout for checkbox & filename
            st.checkbox(
                f"{name[:18]}..." if len(name) > 20 else name,
                value=is_checked,
                key=f"keep_{name}",
                on_change=on_checkbox_change,
                args=(name,)
            )
            
            st.write(" ")

    st.write("---")
    st.subheader("💾 Export Selection")
    
    # Save Action
    if st.button("🚀 Copy Selected Photos to 'Best_Selected' Folder", type="primary", use_container_width=True):
        selected_to_copy = [item for item in results if st.session_state['selections'].get(item['name'], False)]
        if not selected_to_copy:
            st.warning("No photos selected for copying. Adjust threshold or check photos above.")
        else:
            output_folder = os.path.join(st.session_state['input_folder'], "Best_Selected")
            os.makedirs(output_folder, exist_ok=True)
            
            copy_bar = st.progress(0)
            copy_status = st.empty()
            
            copied_count = 0
            for i, item in enumerate(selected_to_copy):
                copy_status.text(f"Copying {item['name']} ({i+1}/{len(selected_to_copy)})...")
                src = os.path.join(st.session_state['input_folder'], item['name'])
                dst = os.path.join(output_folder, item['name'])
                try:
                    shutil.copy(src, dst)
                    copied_count += 1
                except Exception as e:
                    st.error(f"Failed to copy {item['name']}: {e}")
                copy_bar.progress((i + 1) / len(selected_to_copy))
                
            copy_status.empty()
            copy_bar.empty()
            st.success(f"🎉 Successfully copied {copied_count} photos to: {output_folder}")
else:
    st.write(" ")
    st.info("💡 Please select your photos folder above and click 'Run AI Aesthetic Assessment' to start.")
