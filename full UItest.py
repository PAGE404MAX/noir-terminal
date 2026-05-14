import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sys
import os
import random
import time
import threading

# --- CONFIGURATION & AESTHETICS ---
BG_COLOR = "#000000"
FG_COLOR = "#FFFFFF"
HL_COLOR = "#1a1a1a"
GLITCH_COLOR = "#33ff33" # Matrix green for code
FONT_MAIN = ("Courier", 10)
FONT_BOLD = ("Courier", 12, "bold")
FONT_GLITCH = ("Courier", 8)

PARANOID_MESSAGES = [
    "INTERCEPTING PACKETS...",
    "ENCRYPTING VOID...",
    "THEY ARE LOOKING THROUGH THE WEBCAM.",
    "CLEANING METADATA RESIDUE...",
    "KEYSTROKES LOGGED BY SYSTEM_01.",
    "STAY QUIET.",
    "THEY KNOW YOU ARE HERE.",
    "DATA LEAK PLUGGED.",
    "SIGNAL INTERRUPTED BY MASK_0X.",
    "ROTATING IP ADDRESSES...",
    "WIPING CACHE BEYOND RECOVERY.",
    "HIDDEN PARTITION DETECTED.",
    "SCANNING FOR PROXIMITY SENSORS...",
    "ACCESS GRANTED TO NO_ONE.",
    "01001000 01000101 01001100 01010000",
]

class ParanoidUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NODE_0x7F // PARANOIA ENGINE")
        self.root.geometry("1100x750")
        self.root.configure(bg=BG_COLOR)
        
        # Grid config
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar = tk.Frame(root, bg=BG_COLOR, width=220, bd=1, relief="solid")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.sidebar.grid_propagate(False)

        # Title in Sidebar
        tk.Label(self.sidebar, text="SYSTEM_CONTROL", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).pack(pady=10)
        tk.Frame(self.sidebar, bg=FG_COLOR, height=1).pack(fill="x", padx=10, pady=5)

        self.canvas = tk.Canvas(self.sidebar, bg=BG_COLOR, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.sidebar, orient="vertical", command=self.canvas.yview, bg=BG_COLOR)
        self.scrollable_frame = tk.Frame(self.canvas, bg=BG_COLOR)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        # Scrollbar is hidden unless needed for aesthetic
        # self.scrollbar.pack(side="right", fill="y") 

        self.add_sidebar_buttons()

        # --- MAIN CONTENT AREA ---
        self.main_area = tk.Frame(root, bg=BG_COLOR, bd=1, relief="solid")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        
        self.current_frame = None
        self.show_welcome()

        # --- BOTTOM LOG ---
        self.log_frame = tk.Frame(root, bg=BG_COLOR, height=120, bd=1, relief="solid")
        self.log_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        self.log_text = tk.Text(self.log_frame, bg=BG_COLOR, fg=FG_COLOR, font=FONT_GLITCH, height=6, state="disabled", borderwidth=0)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # --- STATUS BAR ---
        self.status_bar = tk.Label(root, text="STATUS: COMPROMISED // ENCRYPTION: 12% // WATCHING: TRUE", 
                                   bg=FG_COLOR, fg=BG_COLOR, font=FONT_GLITCH, anchor="w")
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Start log thread
        self.update_logs()
        self.glitch_status()

    def add_sidebar_buttons(self):
        buttons = [
            ("FACE DETECTOR", self.face_detector_ui),
            ("FACE REDACTOR", self.face_redactor_ui),
            ("COMPOSER", self.composer_ui),
            ("STYPE", self.stype_ui),
            ("METAREMOVER", self.metaremover_ui),
            ("---", None),
            ("SHOW SOURCE CODE", self.show_source_code),
            ("REBOOT SYSTEM", lambda: messagebox.showwarning("CRITICAL", "SYSTEM REBOOT INACCESSIBLE. THEY HAVE LOCKED THE KERNEL.")),
            ("EXIT", self.root.quit)
        ]

        for text, command in buttons:
            if text == "---":
                tk.Label(self.scrollable_frame, text="-----------------", bg=BG_COLOR, fg="#444444").pack(pady=10)
                continue
            
            btn = tk.Button(
                self.scrollable_frame, 
                text=f"> {text}", 
                command=command,
                bg=BG_COLOR, 
                fg=FG_COLOR, 
                activebackground=FG_COLOR, 
                activeforeground=BG_COLOR,
                font=FONT_MAIN,
                bd=0,
                pady=12,
                anchor="w",
                width=25,
                padx=10
            )
            btn.pack(fill="x")
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=HL_COLOR))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BG_COLOR))

    def clear_main(self):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = tk.Frame(self.main_area, bg=BG_COLOR)
        self.current_frame.pack(fill="both", expand=True)

    def show_welcome(self):
        self.clear_main()
        
        glitch_text = """
██████╗  █████╗ ██████╗  █████╗ ███╗   ██╗ ██████╗ ██╗██████╗ 
██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗  ██║██╔═══██╗██║██╔══██╗
██████╔╝███████║██████╔╝███████║██╔██╗ ██║██║   ██║██║██║  ██║
██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╗██║██║   ██║██║██║  ██║
██║     ██║  ██║██║  ██║██║  ██║██║ ╚████║╚██████╔╝██║██████╔╝
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚══════╝
        """
        
        tk.Label(self.current_frame, text=glitch_text, bg=BG_COLOR, fg=FG_COLOR, font=("Courier", 8)).pack(pady=40)
        
        lbl = tk.Label(self.current_frame, text="[ IDENTITY VERIFIED: UNKNOWN ]\n[ LOCATION: [REDACTED] ]\n\nTHE SYSTEM IS NOT SAFE.", 
                       bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD, justify="center")
        lbl.pack(expand=True)
        
        tk.Label(self.current_frame, text="DO NOT TRUST THE OUTPUT.\nDO NOT LOOK BEHIND YOU.", bg=BG_COLOR, fg="#555555", font=FONT_GLITCH).pack(side="bottom", pady=20)

    # --- MODULES ---

    def face_detector_ui(self):
        messagebox.showinfo("NOTICE", "FACE DETECTOR WORKS ADD LOGIC")
        self.clear_main()
        tk.Label(self.current_frame, text=">> MODULE: FACE_DETECTOR.EXE", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).pack(pady=20)
        
        # Placeholder for camera box
        cam_box = tk.Frame(self.current_frame, bg=BG_COLOR, bd=1, relief="solid", width=400, height=300)
        cam_box.pack(pady=10)
        cam_box.pack_propagate(False)
        tk.Label(cam_box, text="[ CAMERA SIGNAL LOST ]", bg=BG_COLOR, fg="#333333", font=FONT_BOLD).pack(expand=True)
        
        tk.Button(self.current_frame, text="INITIALIZE SENSORS", command=lambda: messagebox.showinfo("NOTICE", "FACE DETECTOR WORKS ADD LOGIC"),
                  bg=BG_COLOR, fg=FG_COLOR, font=FONT_MAIN, bd=1, relief="solid", padx=20).pack(pady=20)

    def face_redactor_ui(self):
        messagebox.showinfo("NOTICE", "FACE REDACTOR WORKS ADD LOGIC")
        self.clear_main()
        tk.Label(self.current_frame, text=">> MODULE: FACE_REDACTOR.EXE", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).pack(pady=20)
        tk.Label(self.current_frame, text="TARGET ACQUISITION: NONE", bg=BG_COLOR, fg="#FF0000", font=FONT_MAIN).pack()

    def composer_ui(self):
        # messagebox.showinfo("NOTICE", "COMPOSER WORKS ADD LOGIC")
        self.clear_main()
        tk.Label(self.current_frame, text=">> MODULE: COMPOSER (EMPTY)", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).pack(pady=20)
        tk.Label(self.current_frame, text="[ USER-DEFINED LOGIC SPACE ]", bg=BG_COLOR, fg="#222222", font=FONT_MAIN).pack(expand=True)

    def stype_ui(self):
        self.clear_main()
        tk.Label(self.current_frame, text=">> MODULE: S-TYPE (Keystroke Emulation)", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).pack(pady=10)
        
        tk.Label(self.current_frame, text="PASTE DATA STREAM:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_MAIN).pack(anchor="w", padx=40)
        
        text_area = scrolledtext.ScrolledText(self.current_frame, bg=BG_COLOR, fg=FG_COLOR, insertbackground=FG_COLOR, font=FONT_MAIN, height=12, bd=1, relief="solid")
        text_area.pack(fill="both", expand=True, padx=40, pady=10)

        def run_stype():
            messagebox.showinfo("NOTICE", "STYPE WORKS ADD LOGIC")
            self.log("STYPE: Ghost in the machine initiated.")
        
        btn = tk.Button(self.current_frame, text="[ TRANSMIT TO EXTERNAL WINDOW ]", bg=BG_COLOR, fg=FG_COLOR, command=run_stype, font=FONT_BOLD, bd=1, relief="solid", pady=5)
        btn.pack(pady=20)

    def metaremover_ui(self):
        self.clear_main()
        tk.Label(self.current_frame, text=">> MODULE: METAREMOVER (Exif Scrub)", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).pack(pady=10)

        drop_zone = tk.Frame(self.current_frame, bg=BG_COLOR, bd=1, relief="solid", height=150)
        drop_zone.pack(fill="x", padx=40, pady=20)
        drop_zone.pack_propagate(False)
        
        label = tk.Label(drop_zone, text="[ DROP FILE HERE TO SCRUB ]", bg=BG_COLOR, fg="#888888", font=FONT_MAIN)
        label.pack(expand=True)

        def select_file(event=None):
            f = filedialog.askopenfilename()
            if f:
                messagebox.showinfo("NOTICE", "MetaRemover WORKS ADD LOGIC")
                self.log(f"METAREMOVER: Accessing {os.path.basename(f)}...")
                meta_display.config(state="normal")
                meta_display.delete(1.0, tk.END)
                meta_display.insert(tk.END, f"FILE: {f}\nSIZE: {os.path.getsize(f)} bytes\nENCRYPTION: NONE\n\nMETADATA DETECTED:\n- GPS_DATA: 40.7128° N, 74.0060° W\n- DEVICE: IPHONE_SURVEILLANCE_MODEL_X\n- TIMESTAMP: 19:84:00\n\n[ SCRUBBING HIGHLY RECOMMENDED ]")
                meta_display.config(state="disabled")

        drop_zone.bind("<Button-1>", select_file)
        label.bind("<Button-1>", select_file)

        meta_display = scrolledtext.ScrolledText(self.current_frame, bg=BG_COLOR, fg=GLITCH_COLOR, font=FONT_GLITCH, height=10, bd=1, relief="solid")
        meta_display.pack(fill="both", expand=True, padx=40, pady=10)
        meta_display.insert(tk.END, "WAITING FOR INPUT...")
        meta_display.config(state="disabled")

        btn_purge = tk.Button(self.current_frame, text="[ DELETE ALL EVIDENCE ]", bg=BG_COLOR, fg="#FF0000", font=FONT_BOLD, 
                              command=lambda: messagebox.showinfo("NOTICE", "MetaRemover WORKS ADD LOGIC"), bd=1, relief="solid")
        btn_purge.pack(pady=10)

    def show_source_code(self):
        self.clear_main()
        nb = ttk.Notebook(self.current_frame)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        # ttk Notebook Dark Theme
        style = ttk.Style()
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_COLOR, foreground=FG_COLOR, padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", HL_COLOR)])

        try:
            with open(__file__, "r") as f:
                code = f.read()
        except:
            code = "FATAL ERROR: COULD NOT READ SELF."

        lines = code.splitlines()
        chunk_size = 200
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

        for i, chunk in enumerate(chunks):
            frame = tk.Frame(nb, bg=BG_COLOR)
            txt = scrolledtext.ScrolledText(frame, bg=BG_COLOR, fg=GLITCH_COLOR, font=("Consolas", 9), insertbackground=GLITCH_COLOR, bd=0)
            txt.pack(fill="both", expand=True)
            txt.insert(tk.END, "\n".join(chunk))
            txt.config(state="disabled")
            nb.add(frame, text=f"SRC_0{i+1}")

    # --- UTILITIES ---

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def update_logs(self):
        if random.random() < 0.15:
            self.log(random.choice(PARANOID_MESSAGES))
        self.root.after(2500, self.update_logs)

    def glitch_status(self):
        # Make the status bar flicker slightly
        if random.random() < 0.05:
            self.status_bar.config(bg="#FF0000", text="STATUS: BREACHED // SYSTEM FAILURE IMMINENT")
        else:
            self.status_bar.config(bg=FG_COLOR, text=f"STATUS: SECURE // ENCRYPTION: {random.randint(80,99)}% // WATCHING: {'TRUE' if random.random() > 0.5 else 'FALSE'}")
        self.root.after(500, self.glitch_status)

if __name__ == "__main__":
    root = tk.Tk()
    app = ParanoidUI(root)
    root.mainloop()
