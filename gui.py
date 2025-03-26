import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from main import AG

class GUI:

    def __init__(self, ventana):
        # Configuracion de la ventana
        self.ventana = ventana
        self.ventana.title("Algoritmo Genetico de cultivos")
        self.ventana.geometry("1200x800")
         # Variables
        self.num_poblacion = tk.IntVar(value=50)
        self.num_generaciones = tk.IntVar(value=100)
        self.prob_mutacion = tk.DoubleVar(value=0.1)
        self.prob_crossover = tk.DoubleVar(value=0.9)
        self.num_elite = tk.IntVar(value=1)
        self.cantidad_agua = tk.IntVar(value=100)
        self.cantidad_fertilizante = tk.IntVar(value=100)
        self.cantidad_pesticida = tk.IntVar(value=100)

        self.ag = AG(
            self.num_poblacion.get(),
            self.num_generaciones.get(),
            self.prob_mutacion.get(),
            self.prob_crossover.get(),
            self.num_elite.get(),
            self.cantidad_agua.get(),
            self.cantidad_fertilizante.get(),
            self.cantidad_pesticida.get()
        )
        self.ag.datos_csv = None

        # Estilos
        style = ttk.Style()
        style.configure("White.TFrame", background="white")
        style.configure("Gray.TFrame", background="#DCDCDC")

       

        self.create_tabs()

    
    def create_tabs(self):
        notebook = ttk.Notebook(self.ventana)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_datos = ttk.Frame(notebook)
        notebook.add(tab_datos, text="Datos")

        tab_graficas = ttk.Frame(notebook)
        notebook.add(tab_graficas, text="Graficas")

        self.create_widget(tab_datos)

        self.create_graph(tab_graficas)


    def create_widget(self, parent):

        # Frame para los entrys
        frame_entrys = ttk.Frame(parent, relief="raised", borderwidth=1, style="Gray.TFrame")
        frame_entrys.pack(side=tk.LEFT, fill=tk.Y)
        # Frame para los resultados
        self.frame_results = ttk.Frame(parent,style="White.TFrame")
        self.frame_results.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self.frame_results,columns=(), show="headings")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.frame_data = ttk.Frame(self.frame_results, relief="sunken", borderwidth=1, style="White.TFrame")
        self.frame_data.pack(fill=tk.BOTH, expand=True)

        # Entry para los parametros del AG
        self.label_parametros = ttk.Label(frame_entrys, text="Parametros del AG", font=("Arial", 10), anchor="center")
        self.label_parametros.pack(fill=tk.X, pady=10)
        self.frame_poblacion = self.create_labeled_entry(frame_entrys, "numero de poblacion", self.num_poblacion)
        self.frame_poblacion = self.create_labeled_entry(frame_entrys, "numero de generaciones", self.num_generaciones)
        self.frame_poblacion = self.create_labeled_entry(frame_entrys, "probabilidad de mutacion", self.prob_mutacion)
        self.frame_poblacion = self.create_labeled_entry(frame_entrys, "probabilidad de cruza", self.prob_crossover)
        self.frame_poblacion = self.create_labeled_entry(frame_entrys, "numero de elite", self.num_elite)

        self.label_recursos = ttk.Label(frame_entrys, text="Recursos Disponibles", font=("Arial", 10), anchor="center")
        self.label_recursos.pack(fill=tk.X, pady=10)

        self.frame_agua = self.create_labeled_entry(frame_entrys, "cantidad de agua disponible L", self.cantidad_agua)
        self.frame_fertilizante = self.create_labeled_entry(frame_entrys, "cantidad de fertilizante disponible kg", self.cantidad_fertilizante)
        self.frame_pesticida = self.create_labeled_entry(frame_entrys, "cantidad de pesticida disponible kg", self.cantidad_pesticida)

        self.button_load = ttk.Button(frame_entrys, text="Cargar Datos", command=self.load_csv)
        self.button_load.pack(fill=tk.X, padx=5, pady=10)

        self.button = ttk.Button(frame_entrys, text="Ejecutar AG", command= self.ejecutar_ag)
        self.button.pack(fill=tk.X, padx=5, pady=5)
    
    def load_csv(self):
        data = self.ag.load_csv()
        if data is not None:
            self.ag = AG(
                self.num_poblacion.get(),
                self.num_generaciones.get(),
                self.prob_mutacion.get(),
                self.prob_crossover.get(),
                self.num_elite.get(),
                self.cantidad_agua.get(),
                self.cantidad_fertilizante.get(),
                self.cantidad_pesticida.get()
            )
            self.ag.datos_csv = data  # Asigna los datos cargados a la instancia

            self.tree.delete(*self.tree.get_children())

            self.tree["columns"] = list(data.columns)
            for col in data.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100)

            for _, row in data.iterrows():
                self.tree.insert("", tk.END, values=list(row))
        else:
            messagebox.showerror("Error", "Error al cargar el archivo CSV.")  # Informa al usuario si falla la carga


    def create_graph(self, parent):
        label = ttk.Label(parent, text="Grafica")
        label.pack(fill=tk.X, pady=10)        

    def create_labeled_entry(self, parent, text, variable):
        frame = ttk.Frame(parent, relief="groove", borderwidth=1)
        frame.pack(pady=2, fill=tk.X)
        label = ttk.Label(frame, text=text)
        label.pack(side=tk.LEFT, padx=2, pady=2)
        entry = ttk.Entry(frame, textvariable=variable)
        entry.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.X, expand=True)
        return frame
    
    def show_results(self, mejor_individuo):
       
        for widget in self.frame_data.winfo_children():
            widget.destroy()

        ttk.Label(
            self.frame_data,
            text="Resultados del Mejor Individuo",
            font=("Arial", 14, "bold"),
            anchor="center"
        ).pack(fill=tk.X, pady=10)

        tree = ttk.Treeview(self.frame_data, columns=("Atributo", "Valor"), show="headings", height=10)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree.heading("Atributo", text="Atributo", anchor="center")
        tree.heading("Valor", text="Valor", anchor="center")

        tree.column("Atributo", anchor="w", width=150)
        tree.column("Valor", anchor="center", width=200)

        for key, value in mejor_individuo.items():
            tree.insert("", tk.END, values=(key, value))

        ttk.Button(
            self.frame_data,
            text="Limpiar Resultados",
            command=lambda: self.clear_results()
        ).pack(pady=10)

    def clear_results(self):
        """
        Limpia los resultados mostrados en el frame de datos.
        """
        for widget in self.frame_data.winfo_children():
            widget.destroy()

    def ejecutar_ag(self):
        if self.ag.datos_csv is None or self.ag.datos_csv.empty:
            messagebox.showerror("Error", "Por favor, carga un archivo CSV primero.")
            return
        ag = AG(
            num_poblacion = self.num_poblacion.get(),
            num_generaciones = self.num_generaciones.get(),
            prob_mutacion = self.prob_mutacion.get(),
            prob_crossover = self.prob_crossover.get(),
            num_elite = self.num_elite.get(),
            cantidad_agua = self.cantidad_agua.get(),
            cantidad_fertilizante = self.cantidad_fertilizante.get(),
            cantidad_pesticida = self.cantidad_pesticida.get()
        )
        self.ag.datos_csv = self.ag.datos_csv

        try:
            mejor_individuo = self.ag.optimizar()
            self.show_results(mejor_individuo)
        except Exception as e:
             messagebox.showerror("Error", f"Ocurrió un error durante la optimización: {e}")  #

        

mi_ventana = tk.Tk()
app = GUI(mi_ventana)
mi_ventana.mainloop()