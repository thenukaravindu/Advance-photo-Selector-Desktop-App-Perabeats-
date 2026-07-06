import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import numpy as np
import shutil
import io
import hashlib
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import concurrent.futures

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

# NIMA Transforms
nima_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def calculate_mean_score(output):
    probs = output.cpu().numpy()[0]
    weights = np.arange(1, 11)
    return np.sum(probs * weights)

# Cache Directory
CACHE_DIR = os.path.join(os.getcwd(), ".photo_cully_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variables for model state and culling
loaded_model = None
loaded_model_type = "None"
loaded_device = None

processed_results = [] # list of dicts: {'name', 'score', 'thumb_path'}
selections = {}        # dict: {name: True/False}
card_widgets = {}      # dict: {name: Frame}
image_refs = {}        # dict: {name: PhotoImage}
card_checkboxes = {}   # dict: {name: Checkbutton}
checkbox_vars = {}     # dict: {name: BooleanVar}

gui_queue = queue.Queue()

def get_resource_path(relative_path):
    import sys
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
        return os.path.join(base_dir, relative_path)
    return relative_path

# --- Model Loader Helper ---
def load_nima_model(use_local_weights=True):
    global loaded_model, loaded_model_type, loaded_device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = get_resource_path("epoch-82.pth")
        
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
            print(f"Error loading local weights: {e}")
            
    if model is None:
        try:
            import pyiqa
            model = pyiqa.create_metric('nima', device=device)
            model_type = "Pre-trained PyIQA NIMA"
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyIQA: {e}")
            
    loaded_model = model
    loaded_model_type = model_type
    loaded_device = device
    return model, model_type, device

# --- GUI Class ---
class PhotoCullyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PERABEATS Photo Selecter - Desktop App")
        self.root.geometry("1150x780")
        self.root.minsize(900, 600)
        
        # Set Window Icon
        logo_path = get_resource_path("perabeats_logo.png")
        if os.path.exists(logo_path):
            try:
                logo_icon_pil = Image.open(logo_path)
                self.window_icon_ref = ImageTk.PhotoImage(logo_icon_pil)
                self.root.iconphoto(False, self.window_icon_ref)
            except Exception:
                pass
        
        # Setup Animation States
        self.scroll_target = None
        self.is_scroll_animating = False
        self.progress_animation_target = 0.0
        self.is_progress_animating = False
        
        self.prev_total = 0
        self.prev_selected = 0
        self.prev_discarded = 0
        
        self.setup_styles()
        self.build_ui()
        
        # Start queue poller
        self.root.after(100, self.poll_queue)
        
        # Detect model load on start
        self.status_lbl.config(text="Initializing NIMA model...")
        threading.Thread(target=self.init_model_worker, daemon=True).start()

    def setup_styles(self):
        # Dark Theme Palette
        self.bg_color = "#121214"
        self.card_bg = "#1e1e24"
        self.card_hover_bg = "#2d2d3a"
        self.accent_color = "#6366f1"
        self.text_color = "#ffffff"
        self.text_secondary = "#a5a5c5"
        
        self.root.configure(bg=self.bg_color)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure standard widgets
        self.style.configure('.', background=self.bg_color, foreground=self.text_color, font=('Segoe UI', 10))
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color)
        self.style.configure('TEntry', fieldbackground='#1d1d24', background='#1d1d24', foreground='#ffffff', bordercolor='#2d3345')
        
        # Sidebar Frame styling
        self.style.configure('Sidebar.TFrame', background='#16161a')
        
        # Custom TTK Buttons
        self.style.configure('Accent.TButton', background=self.accent_color, foreground='#ffffff', font=('Segoe UI', 10, 'bold'), borderwidth=0)
        self.style.map('Accent.TButton', background=[('active', '#4f46e5'), ('disabled', '#2d2d3a')], foreground=[('disabled', '#8a8a9a')])
        
        self.style.configure('Normal.TButton', background='#2e303e', foreground='#ffffff', borderwidth=0)
        self.style.map('Normal.TButton', background=[('active', '#3c3f54')])
        
        # Progress & Combobox
        self.style.configure('Horizontal.TProgressbar', background=self.accent_color, troughcolor='#1d1d24')
        self.style.configure('TCombobox', fieldbackground='#1d1d24', background='#2e303e', foreground='#ffffff')

    def build_ui(self):
        # Layout: Main Grid
        self.root.columnconfigure(0, weight=0, minsize=320) # Sidebar
        self.root.columnconfigure(1, weight=1)              # Gallery
        self.root.rowconfigure(0, weight=1)
        
        # --- SIDEBAR PANEL ---
        sidebar = ttk.Frame(self.root, style='Sidebar.TFrame')
        sidebar.grid(row=0, column=0, sticky="nswe", padx=0, pady=0)
        sidebar.columnconfigure(0, weight=1)
        
        # App Title Card
        title_frame = tk.Frame(sidebar, bg='#0d0d14', bd=0)
        title_frame.pack(fill="x", padx=0, pady=0)
        
        # Load logo if exists
        logo_img = None
        logo_path = get_resource_path("perabeats_logo.png")
        if os.path.exists(logo_path):
            try:
                pil_logo = Image.open(logo_path)
                pil_logo.thumbnail((110, 110))
                logo_img = ImageTk.PhotoImage(pil_logo)
                self.sidebar_logo_ref = logo_img  # keep reference
            except Exception:
                pass

        if logo_img:
            logo_lbl = tk.Label(title_frame, image=logo_img, bg="#0d0d14")
            logo_lbl.pack(pady=(15, 0), padx=15, anchor="w")
            title_lbl = tk.Label(title_frame, text="PERABEATS Photo Selecter", font=("Segoe UI", 14, "bold"), fg="#ffffff", bg="#0d0d14")
            title_lbl.pack(pady=(5, 2), padx=15, anchor="w")
        else:
            title_lbl = tk.Label(title_frame, text="📸 PERABEATS Photo Selecter", font=("Segoe UI", 16, "bold"), fg="#ffffff", bg="#0d0d14")
            title_lbl.pack(pady=(20, 2), padx=15, anchor="w")
            
        subtitle_lbl = tk.Label(title_frame, text="Premium AI Photo Selection", font=("Segoe UI", 9, "italic"), fg=self.text_secondary, bg="#0d0d14")
        subtitle_lbl.pack(pady=(0, 15), padx=15, anchor="w")
        
        # Settings Container
        settings_frame = ttk.Frame(sidebar, style='Sidebar.TFrame')
        settings_frame.pack(fill="both", expand=True, padx=20, pady=15)
        
        # Folder Select
        ttk.Label(settings_frame, text="Photos Folder:", font=("Segoe UI", 10, "bold"), background='#16161a').pack(anchor="w", pady=(10, 5))
        
        folder_pick_frame = ttk.Frame(settings_frame, style='Sidebar.TFrame')
        folder_pick_frame.pack(fill="x", pady=2)
        folder_pick_frame.columnconfigure(0, weight=1)
        
        self.folder_var = tk.StringVar(value=r"D:\Club_Photos\Event_Bulk")
        if not os.path.exists(self.folder_var.get()):
            self.folder_var.set("")
            
        self.folder_entry = ttk.Entry(folder_pick_frame, textvariable=self.folder_var)
        self.folder_entry.grid(row=0, column=0, sticky="we", ipady=3, padx=(0, 5))
        
        browse_btn = ttk.Button(folder_pick_frame, text="Browse 📁", style="Normal.TButton", command=self.browse_folder)
        browse_btn.grid(row=0, column=1, sticky="e")
        
        # Model Selection
        ttk.Label(settings_frame, text="Model Weights:", font=("Segoe UI", 10, "bold"), background='#16161a').pack(anchor="w", pady=(15, 5))
        self.model_combo = ttk.Combobox(settings_frame, values=["Local Weights (epoch-82.pth)", "PyIQA Model Hub"], state="readonly")
        self.model_combo.set("Local Weights (epoch-82.pth)")
        self.model_combo.pack(fill="x", pady=2)
        
        # Model Status box
        self.model_info_lbl = tk.Label(settings_frame, text="Model: Loading...\nDevice: cpu", font=("Segoe UI", 9), fg=self.text_secondary, bg="#1d1d24", justify="left", relief="flat", padx=10, pady=10)
        self.model_info_lbl.pack(fill="x", pady=10)
        
        # Slider for Threshold
        ttk.Label(settings_frame, text="Aesthetic Threshold:", font=("Segoe UI", 10, "bold"), background='#16161a').pack(anchor="w", pady=(15, 2))
        
        slider_val_frame = ttk.Frame(settings_frame, style='Sidebar.TFrame')
        slider_val_frame.pack(fill="x")
        
        self.threshold_val_lbl = tk.Label(slider_val_frame, text="5.0", font=("Segoe UI", 11, "bold"), fg=self.accent_color, bg='#16161a')
        self.threshold_val_lbl.pack(side="right")
        
        self.threshold_scale = ttk.Scale(settings_frame, from_=1.0, to=10.0, value=5.0, command=self.on_threshold_slider)
        self.threshold_scale.pack(fill="x", pady=(2, 10))
        
        # Actions
        self.run_btn = ttk.Button(settings_frame, text="Run AI Assessment", style="Accent.TButton", command=self.start_culling_thread)
        self.run_btn.pack(fill="x", pady=(20, 10), ipady=5)
        
        # Progress Bar & Logs
        self.progress_bar = ttk.Progressbar(settings_frame, mode="determinate", style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x", pady=(10, 2))
        
        self.status_lbl = ttk.Label(settings_frame, text="Ready", font=("Segoe UI", 9, "italic"), background='#16161a')
        self.status_lbl.pack(anchor="w")
        
        # Sidebar Cache cleaner
        clear_cache_btn = ttk.Button(settings_frame, text="🧹 Clear Thumbnail Cache", style="Normal.TButton", command=self.clear_thumbnail_cache)
        clear_cache_btn.pack(fill="x", side="bottom", pady=10)
        
        # --- GALLERY PANEL ---
        gallery_panel = ttk.Frame(self.root)
        gallery_panel.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)
        gallery_panel.columnconfigure(0, weight=1)
        gallery_panel.rowconfigure(2, weight=1)
        
        # Stats summary row
        stats_frame = ttk.Frame(gallery_panel)
        stats_frame.grid(row=0, column=0, sticky="we", pady=(0, 15))
        stats_frame.columnconfigure((0, 1, 2), weight=1)
        
        self.stat_total_val, self.stat_total_card, self.stat_total_lbl = self.create_stat_card(stats_frame, "0", "Total Scanned", 0)
        self.stat_selected_val, self.stat_selected_card, self.stat_selected_lbl = self.create_stat_card(stats_frame, "0", "Selected", 1, color="#10b981")
        self.stat_discarded_val, self.stat_discarded_card, self.stat_discarded_lbl = self.create_stat_card(stats_frame, "0", "Discarded", 2, color="#ef4444")
        
        # Sort selection row
        controls_frame = ttk.Frame(gallery_panel)
        controls_frame.grid(row=1, column=0, sticky="we", pady=(0, 10))
        
        ttk.Label(controls_frame, text="Sort by:").pack(side="left")
        self.sort_combo = ttk.Combobox(controls_frame, values=["Score (High to Low)", "Score (Low to High)", "Filename"], state="readonly", width=22)
        self.sort_combo.set("Score (High to Low)")
        self.sort_combo.pack(side="left", padx=10)
        self.sort_combo.bind("<<ComboboxSelected>>", self.on_sort_changed)
        
        # Scrollable Canvas
        canvas_border = tk.Frame(gallery_panel, bg="#23232b", bd=1)
        canvas_border.grid(row=2, column=0, sticky="nswe")
        canvas_border.rowconfigure(0, weight=1)
        canvas_border.columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_border, bg=self.bg_color, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nswe")
        
        self.scrollbar = ttk.Scrollbar(canvas_border, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.grid_frame = tk.Frame(self.canvas, bg=self.bg_color)
        self.canvas_win = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        
        self.grid_frame.bind("<Configure>", self.on_grid_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Bind Mousewheel scroll
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        
        # Bottom Export bar
        export_frame = ttk.Frame(gallery_panel)
        export_frame.grid(row=3, column=0, sticky="we", pady=(15, 0))
        
        self.export_btn = ttk.Button(export_frame, text="Export Selected Photos (0)", style="Accent.TButton", command=self.export_selected_photos)
        self.export_btn.pack(fill="x", ipady=8)

    def create_stat_card(self, parent, val, label, col, color=None):
        card = tk.Frame(parent, bg=self.card_bg, bd=1, relief="solid", highlightbackground="#2d3345", highlightcolor="#2d3345", highlightthickness=1)
        card.grid(row=0, column=col, padx=10, sticky="we")
        card.columnconfigure(0, weight=1)
        
        fg_col = color if color else self.accent_color
        val_lbl = tk.Label(card, text=val, font=("Segoe UI", 20, "bold"), fg=fg_col, bg=self.card_bg)
        val_lbl.pack(pady=(12, 2))
        
        lbl = tk.Label(card, text=label.upper(), font=("Segoe UI", 8, "bold"), fg=self.text_secondary, bg=self.card_bg)
        lbl.pack(pady=(0, 12))
        return val_lbl, card, lbl

    # --- UI Event Handlers ---
    def browse_folder(self):
        selected = filedialog.askdirectory(initialdir=self.folder_var.get())
        if selected:
            self.folder_var.set(os.path.normpath(selected))

    def clear_thumbnail_cache(self):
        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
                os.makedirs(CACHE_DIR, exist_ok=True)
                messagebox.showinfo("Success", "Thumbnail cache cleared successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear cache: {e}")

    def on_threshold_slider(self, val):
        t_val = float(val)
        self.threshold_val_lbl.config(text=f"{t_val:.1f}")
        self.apply_threshold(t_val)

    def on_sort_changed(self, event=None):
        self.sort_results()

    def on_grid_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        # Resize grid frame width to fit canvas width
        self.canvas.itemconfig(self.canvas_win, width=event.width)
        self.rearrange_grid()

    def on_mousewheel(self, event):
        # Only scroll if scrollbar is active
        val = self.scrollbar.get()
        if val != (0.0, 1.0):
            current_scroll = self.canvas.yview()[0]
            direction = -1 if event.delta > 0 else 1
            scroll_step = 0.08
            
            if self.scroll_target is None:
                self.scroll_target = current_scroll
                
            viewport_size = val[1] - val[0]
            max_scroll = 1.0 - viewport_size
            self.scroll_target = max(0.0, min(max_scroll, self.scroll_target + direction * scroll_step))
            
            if not self.is_scroll_animating:
                self.is_scroll_animating = True
                self.animate_scroll()

    def animate_scroll(self):
        if self.scroll_target is None:
            self.is_scroll_animating = False
            return
            
        current = self.canvas.yview()[0]
        diff = self.scroll_target - current
        
        if abs(diff) < 0.002:
            self.canvas.yview("moveto", self.scroll_target)
            self.scroll_target = None
            self.is_scroll_animating = False
            return
            
        new_pos = current + diff * 0.25
        self.canvas.yview("moveto", new_pos)
        self.root.after(12, self.animate_scroll)

    # --- Queue Polling (Thread Communication) ---
    def poll_queue(self):
        try:
            while True:
                msg_type, *args = gui_queue.get_nowait()
                
                if msg_type == "INIT_DONE":
                    model_name, dev_name = args
                    self.model_info_lbl.config(text=f"Model: {model_name}\nDevice: {dev_name}")
                    self.status_lbl.config(text="Ready")
                    
                elif msg_type == "INIT_ERROR":
                    err_msg = args[0]
                    self.model_info_lbl.config(text=f"Model Load Failed!\n{err_msg}")
                    self.status_lbl.config(text="Init Error")
                    messagebox.showerror("Model Load Error", f"Failed to initialize aesthetic model:\n{err_msg}")
                    
                elif msg_type == "PROGRESS":
                    pct, text = args
                    self.update_progress_bar_smooth(pct)
                    self.status_lbl.config(text=text)
                    
                elif msg_type == "ITEM":
                    item = args[0]
                    processed_results.append(item)
                    # Set default selection state (threshold 5.0)
                    selections[item['name']] = item['score'] >= float(self.threshold_scale.get())
                    
                elif msg_type == "DONE":
                    self.run_btn.config(state="normal")
                    self.folder_entry.config(state="normal")
                    self.status_lbl.config(text="Assessment complete!")
                    self.update_progress_bar_smooth(1.0)
                    
                    self.sort_results()
                    messagebox.showinfo("Complete", f"Successfully processed {len(processed_results)} images!")
                    
                elif msg_type == "ERROR":
                    err = args[0]
                    self.run_btn.config(state="normal")
                    self.folder_entry.config(state="normal")
                    self.status_lbl.config(text="Failed")
                    messagebox.showerror("Execution Error", f"An error occurred during processing:\n{err}")
                    
                gui_queue.task_done()
        except queue.Empty:
            pass
            
        self.root.after(100, self.poll_queue)

    # --- Worker Threads ---
    def init_model_worker(self):
        try:
            use_local = (self.model_combo.get() == "Local Weights (epoch-82.pth)")
            _, model_name, dev = load_nima_model(use_local_weights=use_local)
            gui_queue.put(("INIT_DONE", model_name, str(dev)))
        except Exception as e:
            gui_queue.put(("INIT_ERROR", str(e)))

    def start_culling_thread(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.exists(folder):
            messagebox.showwarning("Invalid Path", "Please select a valid folder path first.")
            return
            
        # Clear UI state
        self.clear_ui_grid()
        processed_results.clear()
        selections.clear()
        self.update_stats_display()
        
        self.run_btn.config(state="disabled")
        self.folder_entry.config(state="disabled")
        
        use_local = (self.model_combo.get() == "Local Weights (epoch-82.pth)")
        
        threading.Thread(target=self.culling_worker, args=(folder, use_local), daemon=True).start()

    def culling_worker(self, folder, use_local):
        try:
            # Check if model is loaded or needs reload
            global loaded_model, loaded_model_type, loaded_device
            if loaded_model is None or (use_local and "Local weights" not in loaded_model_type) or (not use_local and "Local weights" in loaded_model_type):
                gui_queue.put(("PROGRESS", 0.05, "Reloading NIMA model..."))
                load_nima_model(use_local_weights=use_local)
                gui_queue.put(("INIT_DONE", loaded_model_type, str(loaded_device)))
                
            model = loaded_model
            loaded_type = loaded_model_type
            device = loaded_device
            
            gui_queue.put(("PROGRESS", 0.1, "Scanning input directory..."))
            
            all_files = os.listdir(folder)
            valid_extensions = {
                '.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp',
                '.cr2', '.nef', '.arw', '.dng', '.cr3', '.orf', '.rw2', '.pef'
            }
            valid_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions]
            
            if not valid_files:
                raise RuntimeError("No supported images found in the selected folder.")
                
            total_files = len(valid_files)
            
            # Setup parallel preloading
            max_workers = min(os.cpu_count() or 4, 6)
            preload_limit = max_workers * 2
            
            file_list = list(valid_files)
            futures = {}
            
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit initial batch
                for i in range(min(len(file_list), preload_limit)):
                    img_name = file_list[i]
                    img_path = os.path.join(folder, img_name)
                    futures[executor.submit(load_image, img_path)] = (i, img_name)
                    
                next_idx = preload_limit
                
                while futures:
                    done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    
                    for f in done:
                        idx, img_name = futures.pop(f)
                        try:
                            img = f.result()
                            
                            # Cache thumbnail
                            h = hashlib.md5(os.path.join(folder, img_name).encode('utf-8')).hexdigest()
                            thumb_path = os.path.join(CACHE_DIR, f"{h}.jpg")
                            if not os.path.exists(thumb_path):
                                img_thumb = img.copy()
                                img_thumb.thumbnail((300, 300))
                                img_thumb.save(thumb_path, "JPEG", quality=85)
                                img_thumb.close()
                                
                            # Model inference
                            with torch.no_grad():
                                score = model(img).item()
                                    
                            gui_queue.put(("ITEM", {
                                'name': img_name,
                                'score': float(score),
                                'thumb_path': thumb_path
                            }))
                            
                            img.close()
                        except Exception as e:
                            # Print to console/log
                            print(f"Error processing {img_name}: {e}")
                            
                        completed += 1
                        gui_queue.put(("PROGRESS", completed / total_files, f"Assessed {completed}/{total_files} images..."))
                        
                        # Load next
                        if next_idx < len(file_list):
                            n_name = file_list[next_idx]
                            n_path = os.path.join(folder, n_name)
                            futures[executor.submit(load_image, n_path)] = (next_idx, n_name)
                            next_idx += 1
                            
            gui_queue.put(("DONE",))
        except Exception as e:
            gui_queue.put(("ERROR", str(e)))

    # --- UI Grid Drawing & Styling ---
    def clear_ui_grid(self):
        for name, frame in card_widgets.items():
            frame.destroy()
        card_widgets.clear()
        image_refs.clear()
        card_checkboxes.clear()
        checkbox_vars.clear()

    def sort_results(self):
        if not processed_results:
            return
            
        sort_val = self.sort_combo.get()
        if sort_val == "Score (High to Low)":
            processed_results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_val == "Score (Low to High)":
            processed_results.sort(key=lambda x: x['score'])
        else:
            processed_results.sort(key=lambda x: x['name'])
            
        self.build_grid_cards()
        self.rearrange_grid()
        self.update_stats_display()

    def build_grid_cards(self):
        # We build widgets for each photo item if they don't already exist.
        for item in processed_results:
            name = item['name']
            if name in card_widgets:
                continue
                
            score = item['score']
            thumb_path = item['thumb_path']
            
            # Card Frame
            card = tk.Frame(self.grid_frame, bg=self.card_bg, bd=0)
            
            # Score badge color
            badge_color = "#10b981" if score >= 6.5 else ("#f59e0b" if score >= 5.0 else "#ef4444")
            
            # Load thumbnail
            try:
                pil_img = Image.open(thumb_path)
                # Max constraint inside layout
                pil_img.thumbnail((200, 150))
                photo_img = ImageTk.PhotoImage(pil_img)
                image_refs[name] = photo_img # keep ref
            except Exception:
                photo_img = None
                
            # Thumbnail label
            img_lbl = tk.Label(card, image=photo_img, bg=self.card_bg)
            img_lbl.pack(padx=5, pady=(5, 2))
            
            # Score label
            score_lbl = tk.Label(card, text=f"Score: {score:.2f}", font=("Segoe UI", 10, "bold"), fg=badge_color, bg=self.card_bg)
            score_lbl.pack(pady=2)
            
            # Checkbox setup
            is_checked = selections.get(name, False)
            var = tk.BooleanVar(value=is_checked)
            checkbox_vars[name] = var
            
            disp_name = name[:18] + "..." if len(name) > 20 else name
            chk = tk.Checkbutton(
                card, 
                text=disp_name, 
                variable=var, 
                bg=self.card_bg, 
                fg=self.text_color, 
                selectcolor="#1d1d24", 
                activebackground=self.card_bg, 
                activeforeground=self.text_color,
                highlightthickness=0,
                command=lambda n=name: self.on_checkbox_toggled(n)
            )
            chk.pack(pady=(2, 6))
            card_checkboxes[name] = chk
            
            # Hover bindings
            self.bind_hover_effects(card, [img_lbl, score_lbl, chk])
            
            card_widgets[name] = card

    def bind_hover_effects(self, card, children):
        def on_enter(e):
            self.animate_card_hover(card, children, self.card_bg, self.card_hover_bg)
            
        def on_leave(e):
            self.animate_card_hover(card, children, self.card_hover_bg, self.card_bg)
            
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        for w in children:
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)

    def animate_card_hover(self, card, children, start_bg, end_bg, steps=5, step=0):
        if not card.winfo_exists():
            return
        factor = (step + 1) / steps
        bg = self.interpolate_hex_color(start_bg, end_bg, factor)
        
        card.config(bg=bg)
        for w in children:
            if w.winfo_exists():
                if isinstance(w, tk.Checkbutton):
                    w.config(bg=bg, activebackground=bg, selectcolor="#23232b" if end_bg == self.card_hover_bg else "#1d1d24")
                else:
                    w.config(bg=bg)
                    
        if step < steps - 1:
            self.root.after(15, lambda: self.animate_card_hover(card, children, start_bg, end_bg, steps, step + 1))

    def update_progress_bar_smooth(self, target_pct):
        self.progress_animation_target = target_pct * 100
        if not self.is_progress_animating:
            self.is_progress_animating = True
            self.animate_progress_bar()

    def animate_progress_bar(self):
        if not self.progress_bar.winfo_exists():
            self.is_progress_animating = False
            return
        current = self.progress_bar['value']
        target = self.progress_animation_target
        diff = target - current
        
        if abs(diff) < 0.5:
            self.progress_bar['value'] = target
            self.is_progress_animating = False
            return
            
        self.progress_bar['value'] = current + diff * 0.2
        self.root.after(15, self.animate_progress_bar)

    def flash_stat_card(self, card, label, count_lbl, default_bg, default_fg, flash_bg, flash_fg):
        if not card.winfo_exists():
            return
        card.config(bg=flash_bg)
        label.config(bg=flash_bg, fg=flash_fg)
        count_lbl.config(bg=flash_bg, fg=flash_fg)
        
        self.animate_stat_card_fade(card, label, count_lbl, flash_bg, default_bg, flash_fg, default_fg, steps=12, step=0)

    def animate_stat_card_fade(self, card, label, count_lbl, start_bg, end_bg, start_fg, end_fg, steps=12, step=0):
        if not card.winfo_exists():
            return
        factor = (step + 1) / steps
        bg = self.interpolate_hex_color(start_bg, end_bg, factor)
        fg = self.interpolate_hex_color(start_fg, end_fg, factor)
        
        card.config(bg=bg)
        label.config(bg=bg, fg=fg)
        count_lbl.config(bg=bg, fg=fg)
        
        if step < steps - 1:
            self.root.after(20, lambda: self.animate_stat_card_fade(card, label, count_lbl, start_bg, end_bg, start_fg, end_fg, steps, step + 1))

    def interpolate_hex_color(self, color1, color2, factor):
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        r = int(r1 + (r2 - r1) * factor)
        g = int(g1 + (g2 - g1) * factor)
        b = int(b1 + (b2 - b1) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def rearrange_grid(self, event=None):
        if not processed_results:
            return
            
        canvas_width = self.canvas.winfo_width()
        card_width = 220 # including padding
        cols = max(1, canvas_width // card_width)
        
        # Hide all first
        for name, frame in card_widgets.items():
            frame.grid_forget()
            
        # Grid them in current sorted order
        for idx, item in enumerate(processed_results):
            name = item['name']
            frame = card_widgets.get(name)
            if frame:
                r = idx // cols
                c = idx % cols
                frame.grid(row=r, column=c, padx=8, pady=8, sticky="nswe")

    def on_checkbox_toggled(self, name):
        selections[name] = checkbox_vars[name].get()
        self.update_stats_display()

    def apply_threshold(self, threshold):
        if not processed_results:
            return
            
        for item in processed_results:
            name = item['name']
            is_selected = item['score'] >= threshold
            selections[name] = is_selected
            
            # Update checkbutton variables programmatically without triggering callbacks
            if name in checkbox_vars:
                checkbox_vars[name].set(is_selected)
                
        self.update_stats_display()

    def update_stats_display(self):
        total = len(processed_results)
        selected = sum(1 for name, val in selections.items() if val)
        discarded = total - selected
        
        # Color flash on changes
        if total != self.prev_total:
            self.flash_stat_card(self.stat_total_card, self.stat_total_lbl, self.stat_total_val, self.card_bg, self.text_secondary, "#22253b", self.accent_color)
            self.prev_total = total
            
        if selected != self.prev_selected:
            self.flash_stat_card(self.stat_selected_card, self.stat_selected_lbl, self.stat_selected_val, self.card_bg, self.text_secondary, "#19352e", "#10b981")
            self.prev_selected = selected
            
        if discarded != self.prev_discarded:
            self.flash_stat_card(self.stat_discarded_card, self.stat_discarded_lbl, self.stat_discarded_val, self.card_bg, self.text_secondary, "#351e22", "#ef4444")
            self.prev_discarded = discarded
            
        self.stat_total_val.config(text=str(total))
        self.stat_selected_val.config(text=str(selected))
        self.stat_discarded_val.config(text=str(discarded))
        
        self.export_btn.config(text=f"Export Selected Photos ({selected})")

    # --- Copying Actions ---
    def export_selected_photos(self):
        selected_names = [name for name, is_sel in selections.items() if is_sel]
        if not selected_names:
            messagebox.showwarning("No Selection", "There are no photos currently selected for export.")
            return
            
        folder = self.folder_var.get().strip()
        output_folder = os.path.join(folder, "Best_Selected")
        
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create output directory:\n{e}")
            return
            
        # Perform Copying on a separate thread to prevent freezing
        self.export_btn.config(state="disabled")
        self.run_btn.config(state="disabled")
        
        threading.Thread(target=self.export_worker, args=(folder, output_folder, selected_names), daemon=True).start()

    def export_worker(self, folder, output_folder, selected_names):
        total = len(selected_names)
        copied = 0
        
        for name in selected_names:
            src = os.path.join(folder, name)
            dst = os.path.join(output_folder, name)
            try:
                shutil.copy(src, dst)
                copied += 1
            except Exception as e:
                print(f"Error copying {name}: {e}")
                
            pct = (copied / total)
            gui_queue.put(("PROGRESS", pct, f"Copying {copied}/{total}..."))
            
        gui_queue.put(("PROGRESS", 0.0, "Export complete"))
        
        def on_done():
            self.export_btn.config(state="normal")
            self.run_btn.config(state="normal")
            self.update_stats_display()
            messagebox.showinfo("Export Done", f"Successfully exported {copied} photos to:\n{output_folder}")
            
        self.root.after(0, on_done)


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoCullyApp(root)
    root.mainloop()
