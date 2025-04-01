import tkinter as tk
from tkinter import ttk, messagebox
try:
    from main import AG
except ImportError:
    messagebox.showerror("Error", "No se pudo importar la clase AG desde main.py.")
    exit()


try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:
    messagebox.showerror("Error", "Necesitas instalar matplotlib: pip install matplotlib")
    exit()


try:
    import pandas as pd
    import numpy as np
except ImportError as e:
     messagebox.showerror("Error", f"Falta la dependencia: {e.name}. Instálala (ej: pip install {e.name}).")
     exit()


class GUI:

    def __init__(self, ventana):
        # --- Configuración de la ventana ---
        self.ventana = ventana
        self.ventana.title("Algoritmo Genético de Cultivos (con Gráfica)")
        self.ventana.geometry("1200x800") 

        # --- Variables Tkinter ---
        self.num_poblacion = tk.IntVar(value=50)
        self.num_generaciones = tk.IntVar(value=100)
        self.prob_mutacion = tk.DoubleVar(value=0.3)
        self.prob_crossover = tk.DoubleVar(value=0.8) # Valor común
        self.num_elite = tk.IntVar(value= 0)        # Valor común
        # Nombres más claros para los máximos disponibles en la GUI
        self.cantidad_agua_max_gui = tk.DoubleVar(value=100.0)
        self.cantidad_fertilizante_max_gui = tk.DoubleVar(value=100.0)
        self.cantidad_pesticida_max_gui = tk.DoubleVar(value=100.0)

        # --- Instancia de AG ---
        # Se inicializa, pero se actualizará antes de ejecutar
        self.ag = AG(
            self.num_poblacion.get(),
            self.num_generaciones.get(),
            self.prob_mutacion.get(),
            self.prob_crossover.get(),
            self.num_elite.get(),
            self.cantidad_agua_max_gui.get(),
            self.cantidad_fertilizante_max_gui.get(),
            self.cantidad_pesticida_max_gui.get()
        )
        # AG debe tener self.mejor_aptitud_historial inicializado

        # --- Estilos (simplificado) ---
        style = ttk.Style()
        style.theme_use('clam') # O el tema que prefieras
        # Puedes definir más estilos si quieres

        # --- Atributos para la gráfica Matplotlib ---
        self.fig_aptitud = None
        self.ax_aptitud = None
        self.canvas_aptitud = None
        self.toolbar_aptitud = None

        # --- Crear la interfaz ---
        self.create_tabs()


    def create_tabs(self):
        notebook = ttk.Notebook(self.ventana)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Pestaña 1: Configuración y Datos/Resultados
        tab_datos = ttk.Frame(notebook)
        notebook.add(tab_datos, text=" Configuración y Resultados ")

        # Pestaña 2: Gráfica de Evolución
        tab_graficas = ttk.Frame(notebook) # Usar estilo por defecto o crear uno
        notebook.add(tab_graficas, text=" Evolución Aptitud ")

        # Crear widgets en las pestañas
        self.create_data_config_widgets(tab_datos) # Renombrado
        self.create_graph_widgets(tab_graficas)     # Renombrado


    # Renombrado desde create_widget
    def create_data_config_widgets(self, parent):
        # Frame izquierdo para parámetros y botones
        frame_params = ttk.Frame(parent, width=300, relief="groove", borderwidth=2)
        frame_params.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5)
        frame_params.pack_propagate(False) # Evitar que se encoja

        # Frame derecho para mostrar datos CSV y resultados del mejor
        frame_display = ttk.Frame(parent)
        frame_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 5), pady=5)

        # --- Contenido del Frame Izquierdo (Parámetros) ---
        ttk.Label(frame_params, text="Parámetros del AG", font=("Arial", 11, "bold")).pack(pady=(10,5))
        # Usar nombres únicos para las referencias si las necesitas, sino, no asignar
        self.create_labeled_entry(frame_params, "Población:", self.num_poblacion)
        self.create_labeled_entry(frame_params, "Generaciones:", self.num_generaciones)
        self.create_labeled_entry(frame_params, "Prob. Mutación:", self.prob_mutacion)
        self.create_labeled_entry(frame_params, "Prob. Cruce:", self.prob_crossover)
        self.create_labeled_entry(frame_params, "Nº Élite:", self.num_elite)

        ttk.Separator(frame_params, orient='horizontal').pack(fill='x', pady=10, padx=10)

        ttk.Label(frame_params, text="Recursos Máx. Disponibles", font=("Arial", 11, "bold")).pack(pady=(5,5))
        # Usar las variables DoubleVar aquí
        self.create_labeled_entry(frame_params, "Agua (L):", self.cantidad_agua_max_gui)
        self.create_labeled_entry(frame_params, "Fertilizante (kg):", self.cantidad_fertilizante_max_gui)
        self.create_labeled_entry(frame_params, "Pesticida (kg):", self.cantidad_pesticida_max_gui)

        ttk.Separator(frame_params, orient='horizontal').pack(fill='x', pady=10, padx=10)

        # Botones
        self.button_load = ttk.Button(frame_params, text="Cargar Datos CSV", command=self.load_csv)
        self.button_load.pack(fill=tk.X, padx=10, pady=5)

        self.button_run = ttk.Button(frame_params, text="Ejecutar AG", command= self.ejecutar_ag_con_feedback) # Cambiado nombre
        self.button_run.pack(fill=tk.X, padx=10, pady=5)

        # Botón para limpiar
        self.button_clear = ttk.Button(frame_params, text="Limpiar Todo", command=self.clear_all)
        self.button_clear.pack(fill=tk.X, padx=10, pady=5)


        # --- Contenido del Frame Derecho (Display) ---
        # Frame superior para tabla CSV
        frame_csv_view = ttk.Frame(frame_display, relief="sunken", borderwidth=1)
        frame_csv_view.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        ttk.Label(frame_csv_view, text="Datos CSV Cargados (Cultivables)").pack(anchor='nw', padx=2, pady=2)

        csv_scroll_y = ttk.Scrollbar(frame_csv_view, orient="vertical")
        csv_scroll_x = ttk.Scrollbar(frame_csv_view, orient="horizontal")
        # Renombrado el Treeview para evitar confusión con el de resultados
        self.tree_csv_display = ttk.Treeview(frame_csv_view, columns=(), show="headings", height=15,
                                             yscrollcommand=csv_scroll_y.set, xscrollcommand=csv_scroll_x.set)
        csv_scroll_y.config(command=self.tree_csv_display.yview)
        csv_scroll_x.config(command=self.tree_csv_display.xview)
        csv_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        csv_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree_csv_display.pack(fill=tk.BOTH, expand=True)


        # Frame inferior para tabla de resultados del mejor
        # Renombrado el frame para evitar confusión
        self.frame_best_result_display = ttk.Frame(frame_display, relief="sunken", borderwidth=1, height=200)
        self.frame_best_result_display.pack(fill=tk.X, expand=False, pady=(5, 0))
        self.frame_best_result_display.pack_propagate(False) # Para mantener altura
        ttk.Label(self.frame_best_result_display, text="Mejor Individuo Encontrado", font=("Arial", 11, "bold")).pack(pady=5)
        # El treeview de resultados se creará/limpiará en show_results


    # Renombrado desde create_graph
    def create_graph_widgets(self, parent):
        """Crea y configura el lienzo y la barra de herramientas de Matplotlib."""
        # --- Configuración de Matplotlib ---
        self.fig_aptitud = Figure(figsize=(7, 5), dpi=100) # Ajusta tamaño
        self.ax_aptitud = self.fig_aptitud.add_subplot(111)
        self.reset_graph_appearance() # Establecer apariencia inicial

        # --- Integración con Tkinter ---
        self.canvas_aptitud = FigureCanvasTkAgg(self.fig_aptitud, master=parent)
        canvas_widget = self.canvas_aptitud.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Barra de herramientas
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))
        self.toolbar_aptitud = NavigationToolbar2Tk(self.canvas_aptitud, toolbar_frame)
        self.toolbar_aptitud.update()


    def reset_graph_appearance(self):
        """Limpia y establece la apariencia por defecto de la gráfica."""
        if self.ax_aptitud:
            self.ax_aptitud.cla()
            self.ax_aptitud.set_title("Evolución de la Mejor Aptitud")
            self.ax_aptitud.set_xlabel("Generación")
            self.ax_aptitud.set_ylabel("Mejor Aptitud")
            self.ax_aptitud.grid(True, linestyle=':', alpha=0.7)
            self.ax_aptitud.text(0.5, 0.5, 'Esperando ejecución...',
                                 ha='center', va='center', transform=self.ax_aptitud.transAxes,
                                 fontsize=10, color='gray')
        if self.canvas_aptitud:
            self.canvas_aptitud.draw_idle() # Redibujar


    def load_csv(self):
        """Carga datos CSV usando el método de AG y los muestra."""
        self.clear_all() # Limpiar todo antes de cargar

        data = self.ag.load_csv() # Este método maneja filedialog y errores CSV
        if data is not None and not data.empty:
            self.tree_csv_display.delete(*self.tree_csv_display.get_children()) # Limpiar vista previa

            cols = list(data.columns)
            self.tree_csv_display["columns"] = cols
            self.tree_csv_display.column("#0", width=0, stretch=tk.NO) # Ocultar columna fantasma
            for col in cols:
                self.tree_csv_display.heading(col, text=col, anchor=tk.W)
                # Ancho simple basado en texto
                col_width = max(60, min(150, len(col) * 9))
                self.tree_csv_display.column(col, width=col_width, anchor=tk.W)

            # Mostrar filas (limitar para rendimiento si es necesario)
            for _, row in data.iterrows():
                # Convertir todo a string para Treeview, manejar NaN
                values = [str(v) if pd.notna(v) else "" for v in row.values]
                self.tree_csv_display.insert("", tk.END, values=values)

            messagebox.showinfo("Éxito", f"Datos CSV cargados ({len(data)} filas cultivables).")

        elif self.ag.datos_csv is None: # Error durante la carga (mensaje ya mostrado por AG)
            pass
        else: # Cargado pero vacío después de filtrar
            messagebox.showwarning("Datos Vacíos", "Archivo cargado, pero sin filas cultivables ('EsCultivable' != 1).")


    def create_labeled_entry(self, parent, text, variable):
        """Helper para crear etiqueta + entry."""
        frame = ttk.Frame(parent)
        frame.pack(pady=2, padx=5, fill=tk.X)
        label = ttk.Label(frame, text=text, width=18, anchor='w') # Ancho fijo para alinear
        label.pack(side=tk.LEFT, padx=(0, 5))
        entry = ttk.Entry(frame, textvariable=variable, width=12) # Ancho fijo
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return frame


    def show_results(self, mejor_individuo):
        """Muestra los detalles del mejor individuo en su frame dedicado."""
        # Limpiar el frame de resultados anterior (excepto la etiqueta del título)
        for widget in self.frame_best_result_display.winfo_children():
            if isinstance(widget, ttk.Label) and "Mejor Individuo" in widget.cget("text"):
                continue
            widget.destroy()

        # Crear el Treeview para los resultados
        results_tree = ttk.Treeview(self.frame_best_result_display, columns=("Atributo", "Valor"), show="headings", height=8) # Altura ajustada
        results_scroll = ttk.Scrollbar(self.frame_best_result_display, orient="vertical", command=results_tree.yview)
        results_tree.configure(yscrollcommand=results_scroll.set)

        results_tree.heading("Atributo", text="Atributo", anchor=tk.W)
        results_tree.heading("Valor", text="Valor", anchor=tk.W)
        results_tree.column("Atributo", anchor="w", width=170, stretch=tk.NO)
        results_tree.column("Valor", anchor="w", width=300)

        if mejor_individuo:
            for key, value in mejor_individuo.items():
                # Formateo simple para visualización
                if isinstance(value, float):
                    val_str = f"{value:.4f}" # 4 decimales para floats
                else:
                    val_str = str(value)
                results_tree.insert("", tk.END, values=(key, val_str))
        else:
            results_tree.insert("", tk.END, values=("Resultado", "No disponible"))

        results_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(0,5), padx=(0,5))
        results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0,5))


    # Renombrado de clear_results a clear_all
    def clear_all(self):
        """Limpia la tabla de resultados, la tabla CSV y la gráfica."""
        print("Limpiando interfaz...")
        # Limpiar tabla de resultados
        self.show_results(None)

        # Limpiar tabla CSV
        if hasattr(self, 'tree_csv_display'):
            self.tree_csv_display.delete(*self.tree_csv_display.get_children())
            # Podrías querer resetear las columnas también si varían mucho entre CSVs
            # self.tree_csv_display["columns"] = ()

        # Resetear gráfica
        self.reset_graph_appearance()


    def update_graph(self):
        """Dibuja el historial de aptitud en la gráfica."""
        if not hasattr(self.ag, 'mejor_aptitud_historial'):
             print("AG no tiene 'mejor_aptitud_historial'")
             self.ax_aptitud.text(0.5, 0.5, 'Error: Historial no encontrado en AG', color='red',
                                  ha='center', va='center', transform=self.ax_aptitud.transAxes)
             self.canvas_aptitud.draw_idle()
             return

        historial = self.ag.mejor_aptitud_historial
        self.ax_aptitud.cla() # Limpiar ejes

        if historial:
            generaciones = range(1, len(historial) + 1)
            self.ax_aptitud.plot(generaciones, historial, marker='o', linestyle='-', markersize=4, color='b', label='Mejor Aptitud')

            # Marcar máximo
            if historial:
                max_apt = max(historial)
                max_gen_idx = historial.index(max_apt)
                self.ax_aptitud.plot(max_gen_idx + 1, max_apt, 'r*', markersize=10, label=f'Max ({max_apt:.4f})')

            self.ax_aptitud.legend(fontsize=8)
            self.ax_aptitud.set_xlim(left=0) # Empezar eje X en 0
        else:
            # Mensaje si no hay historial
             self.ax_aptitud.text(0.5, 0.5, 'No hay datos de aptitud para graficar.',
                                  ha='center', va='center', transform=self.ax_aptitud.transAxes, color='gray')

        # Reestablecer títulos y grid que cla() borra
        self.ax_aptitud.set_title("Evolución de la Mejor Aptitud")
        self.ax_aptitud.set_xlabel("Generación")
        self.ax_aptitud.set_ylabel("Mejor Aptitud")
        self.ax_aptitud.grid(True, linestyle=':', alpha=0.7)
        self.fig_aptitud.tight_layout() # Ajustar márgenes
        self.canvas_aptitud.draw_idle() # Redibujar (idle es más seguro desde callbacks)


    # Renombrado ejecutar_ag para claridad
    def ejecutar_ag_con_feedback(self):
        """Ejecuta el AG, actualiza la GUI y proporciona feedback."""
        if self.ag.datos_csv is None or self.ag.datos_csv.empty:
            messagebox.showerror("Error", "Por favor, carga un archivo CSV cultivable primero.")
            return

        # --- Actualizar parámetros del AG desde la GUI ---
        try:
            pob = self.num_poblacion.get()
            gen = self.num_generaciones.get()
            mut = self.prob_mutacion.get()
            cross = self.prob_crossover.get()
            elite = self.num_elite.get()
            # Leer las variables correctas para los máximos
            agua_max = self.cantidad_agua_max_gui.get()
            fert_max = self.cantidad_fertilizante_max_gui.get()
            pest_max = self.cantidad_pesticida_max_gui.get()

            # Validaciones básicas (puedes añadir más)
            if pob <= 0 or gen <= 0: raise ValueError("Población y Generaciones deben ser > 0")
            if not (0.0 <= mut <= 1.0 and 0.0 <= cross <= 1.0): raise ValueError("Probabilidades entre 0 y 1")
            if elite < 0: raise ValueError("Élite debe ser >= 0")
            if elite >= pob:
                messagebox.showwarning("Ajuste", f"Élite ({elite}) >= Población ({pob}). Ajustando a {max(0, pob - 1)}.")
                elite = max(0, pob - 1)
                self.num_elite.set(elite)

            # Actualizar la instancia AG
            self.ag.num_poblacion = pob
            self.ag.num_generaciones = gen
            self.ag.prob_mutacion = mut
            self.ag.prob_crossover = cross
            self.ag.num_elite = elite
            self.ag.cantidad_agua_max = agua_max
            self.ag.cantidad_fertilizante_max = fert_max
            self.ag.cantidad_pesticida_max = pest_max
            print("Parámetros AG actualizados desde GUI.")

        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Error de Parámetro", f"Valor inválido: {e}")
            return

       
        self.ventana.update_idletasks() # Refrescar GUI

        # Deshabilitar botones y cambiar cursor
        self.button_run.config(state=tk.DISABLED, text="Ejecutando...")
        self.button_load.config(state=tk.DISABLED)
        self.button_clear.config(state=tk.DISABLED)
        self.ventana.config(cursor="watch")

        mejor_individuo = None
        try:
            print(f"Iniciando AG ({self.ag.num_generaciones} generaciones)...")
            # AG.optimizar debe llenar self.ag.mejor_aptitud_historial
            mejor_individuo = self.ag.optimizar()
            print("Optimización AG finalizada.")

            # Actualizar GUI con resultados y gráfica
            self.show_results(mejor_individuo)
            self.update_graph() # <--- LLAMADA PARA ACTUALIZAR GRÁFICA

            if not mejor_individuo:
                 messagebox.showinfo("Información", "La optimización no produjo un resultado válido.")

        except Exception as e:
            messagebox.showerror("Error en Ejecución", f"Ocurrió un error durante la optimización:\n{e}")
            import traceback
            print("\n--- TRACEBACK ---")
            traceback.print_exc()
            print("-----------------\n")
            self.clear_all() # Limpiar si hubo error grave

        finally:
            # Rehabilitar botones y restaurar cursor
            self.button_run.config(state=tk.NORMAL, text="Ejecutar AG")
            self.button_load.config(state=tk.NORMAL)
            self.button_clear.config(state=tk.NORMAL)
            self.ventana.config(cursor="")
            print("Interfaz rehabilitada.")


# --- Bloque principal para ejecutar ---
if __name__ == "__main__":
    mi_ventana = tk.Tk()
    app = GUI(mi_ventana)
    mi_ventana.mainloop()