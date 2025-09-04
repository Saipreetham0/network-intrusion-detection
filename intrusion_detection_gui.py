import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

class IntrusionDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Intrusion Detection System")
        
        self.label_file_path = tk.Label(master, text="Select Dataset:")
        self.label_file_path.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.file_path = tk.StringVar()
        self.entry_file_path = tk.Entry(master, textvariable=self.file_path, width=40)
        self.entry_file_path.grid(row=0, column=1, columnspan=2, padx=10, pady=5)
        
        self.button_browse = tk.Button(master, text="Browse", command=self.browse_file)
        self.button_browse.grid(row=0, column=3, padx=5, pady=5)
        
        self.label_status = tk.Label(master, text="")
        self.label_status.grid(row=1, column=0, columnspan=4, pady=5)
        
        self.button_run_detection = tk.Button(master, text="Run Intrusion Detection", command=self.run_detection)
        self.button_run_detection.grid(row=2, column=0, columnspan=4, pady=10)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if file_path:
            self.file_path.set(file_path)
            self.label_status.config(text="")
        else:
            self.label_status.config(text="No file selected", fg="red")
            
    def run_detection(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a dataset file.")
            return
        
        # Placeholder for intrusion detection functionality
        messagebox.showinfo("Info", "Intrusion detection will be performed using Random Forest and PCA.")
        

def main():
    root = tk.Tk()
    app = IntrusionDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
