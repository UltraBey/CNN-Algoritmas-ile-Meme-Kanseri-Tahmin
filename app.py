import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from model import example_data
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def fill_form_with_sample(sample):
    for entry, value in zip(entries, sample):
        entry.delete(0, tk.END)
        entry.insert(0, value)

def predict():
    try:
        input_data = pd.DataFrame([{
            "Clump_Thickness": float(entry_clump_thickness.get()),
            "Uniformity_of_Cell_Size": float(entry_uniformity_of_cell_size.get()),
            "Uniformity_of_Cell_Shape": float(entry_uniformity_of_cell_shape.get()),
            "Marginal_Adhesion": float(entry_marginal_adhesion.get()),
            "Single_Epithelial_Cell_Size": float(entry_single_epithelial_cell_size.get()),
            "Bare_Nuclei": float(entry_bare_nuclei.get()),
            "Bland_Chromatin": float(entry_bland_chromatin.get()),
            "Normal_Nucleoli": float(entry_normal_nucleoli.get()),
            "Mitoses": float(entry_mitoses.get())
        }])
        
        prediction = example_data['predict'](input_data)
        result = "Benign (İyi Huylu)" if prediction == 2 else "Malignant (Kötü Huylu)"
        messagebox.showinfo("Sonuç", f"Tahmin: {result}")
    except ValueError as e:
        messagebox.showerror("Hata", f"Lütfen tüm alanlara doğru değerler girin.\nHata: {str(e)}")

def create_entry_row(label_text, row):
    tk.Label(root, text=label_text).grid(row=row, column=0)
    entry = tk.Entry(root)
    entry.grid(row=row, column=1)
    return entry

root = tk.Tk()
root.title("Meme Kanseri Tahmini")

labels = ["Clump Thickness", "Uniformity of Cell Size", 
          "Uniformity of Cell Shape", "Marginal Adhesion", 
          "Single Epithelial Cell Size", "Bare Nuclei", 
          "Bland Chromatin", "Normal Nucleoli", "Mitoses"]

entries = [create_entry_row(label, i) for i, label in enumerate(labels)]

entry_clump_thickness = entries[0]
entry_uniformity_of_cell_size = entries[1]
entry_uniformity_of_cell_shape = entries[2]
entry_marginal_adhesion = entries[3]
entry_single_epithelial_cell_size = entries[4]
entry_bare_nuclei = entries[5]
entry_bland_chromatin = entries[6]
entry_normal_nucleoli = entries[7]
entry_mitoses = entries[8]

tk.Button(root, text="Tahmin Et", command=predict).grid(row=9, column=0, columnspan=2)

def create_sample_button(text, sample, row, col):
    tk.Button(root, text=text, command=lambda: fill_form_with_sample(sample)).grid(row=row, column=col)

for i, sample_type in enumerate(['benign', 'malignant']):
    for j, sample in enumerate(example_data[sample_type]):
        create_sample_button(f"Örnek {j+1} ({'Benign - İyi Huylu' if sample_type == 'benign' else 'Malignant - Kötü Huylu'})", sample, 10 + i, j)

def show_confusion_matrix():
    conf_matrix = example_data['plot_confusion_matrix']()
    window = tk.Toplevel()
    window.title("Karışıklık Matrisi")
    
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (Karışıklık Matrisi)')
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Benign (İyi Huylu)', 'Malignant (Kötü Huylu)'])
    ax.set_yticklabels(['Benign (İyi Huylu)', 'Malignant (Kötü Huylu)'])
    ax.set_ylabel('Actual (Gerçek)')
    ax.set_xlabel('Predicted (Tahmin)')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

tk.Button(root, text="Karışıklık Matrisini Göster", command=show_confusion_matrix).grid(row=12, column=0, columnspan=3)

root.mainloop()
