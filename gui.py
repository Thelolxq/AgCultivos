import tkinter as tk
from tkinter import ttk, messagebox
import traceback 
import matplotlib.patches as patches 
import matplotlib.cm as cm 

try:
    # AJUSTA ESTE NOMBRE si guardaste la clase AG en otro archivo
    from main import AG
except ImportError:
    messagebox.showerror("Error de Importación", "No se pudo importar la clase AG.\nAsegúrate de que el archivo (ej: ag_planificador.py) esté en el mismo directorio y no tenga errores.")
    traceback.print_exc() # Imprime el error detallado en la consola
    exit()
except Exception as e:
    messagebox.showerror("Error al Importar AG", f"Ocurrió un error inesperado al importar AG:\n{e}")
    traceback.print_exc()
    exit()


# --- Intenta importar dependencias ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import pandas as pd
    import numpy as np
except ImportError as e:
     messagebox.showerror("Error de Dependencia", f"Falta la dependencia: {e.name}.\nInstálala (ej: pip install {e.name}).")
     exit()


class GUI:

    def __init__(self, ventana):
        # --- Configuración de la ventana ---
        self.ventana = ventana
        self.ventana.title("Optimizador de Plan de Cultivos AG (con Gráfica)") # Título actualizado
        self.ventana.geometry("1200x850") # Un poco más alto para nuevos params/resultados

        # --- Variables Tkinter ---
        self.num_poblacion = tk.IntVar(value=100)       # Valores por defecto ajustados para planes
        self.num_generaciones = tk.IntVar(value=50)
        self.prob_mutacion_cantidad_gui = tk.DoubleVar(value=0.05) # Renombrado para claridad
        self.prob_mutar_cultivo_gui = tk.DoubleVar(value=0.10)    # NUEVO: Probabilidad de mutar cultivo en parcela
        self.prob_crossover = tk.DoubleVar(value=0.7)
        self.num_elite = tk.IntVar(value=5)
        self.num_parcelas_gui = tk.IntVar(value=4)               # NUEVO: Número de parcelas

        # RENOMBRADO: Límites TOTALES disponibles
        self.total_agua_disponible_gui = tk.DoubleVar(value=1800.0)
        self.total_fertilizante_disponible_gui = tk.DoubleVar(value=500.0)
        self.total_pesticida_disponible_gui = tk.DoubleVar(value=100.0)

        # --- Instancia de AG ---
        # Se inicializa con valores por defecto, se actualizará antes de ejecutar
        try:
            self.ag = AG(
                num_poblacion=self.num_poblacion.get(),
                num_generaciones=self.num_generaciones.get(),
                prob_mutacion=self.prob_mutacion_cantidad_gui.get(), # Pasa prob cantidad
                prob_crossover=self.prob_crossover.get(),
                num_elite=self.num_elite.get(),
                num_parcelas=self.num_parcelas_gui.get(),             # Pasa num parcelas
                total_agua_disponible=self.total_agua_disponible_gui.get(), # Pasa totales
                total_fertilizante_disponible=self.total_fertilizante_disponible_gui.get(),
                total_pesticida_disponible=self.total_pesticida_disponible_gui.get(),
                prob_mutar_cultivo_en_parcela=self.prob_mutar_cultivo_gui.get() # Pasa prob mutar cultivo
            )
        except TypeError as e:
             messagebox.showerror("Error al Crear AG", f"Discrepancia en los parámetros al inicializar AG.\nRevisa la definición de __init__ en tu clase AG.\n\nError: {e}")
             traceback.print_exc()
             exit()
        except Exception as e:
             messagebox.showerror("Error al Crear AG", f"Error inesperado al inicializar AG:\n{e}")
             traceback.print_exc()
             exit()

        # --- Estilos ---
        style = ttk.Style()
        style.theme_use('clam')

        # --- Atributos para la gráfica Matplotlib ---
        self.fig_aptitud = None
        self.ax_aptitud = None
        self.ax_plan_map = None
        self.canvas_aptitud = None
        self.toolbar_aptitud = None
        self.crop_colors = {}
        # --- Crear la interfaz ---
        self.create_tabs()


    def create_tabs(self):
        notebook = ttk.Notebook(self.ventana)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tab_datos = ttk.Frame(notebook)
        notebook.add(tab_datos, text=" Configuración y Resultados ")

        tab_graficas = ttk.Frame(notebook)
        notebook.add(tab_graficas, text=" Evolución Aptitud ")

        self.create_data_config_widgets(tab_datos)
        self.create_graph_widgets(tab_graficas)


    def create_data_config_widgets(self, parent):
        frame_params = ttk.Frame(parent, width=350, relief="groove", borderwidth=2) # Un poco más ancho
        frame_params.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5)
        frame_params.pack_propagate(False)

        frame_display = ttk.Frame(parent)
        frame_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 5), pady=5)

        # --- Contenido del Frame Izquierdo (Parámetros) ---
        ttk.Label(frame_params, text="Parámetros del AG", font=("Arial", 11, "bold")).pack(pady=(10,5))
        self.create_labeled_entry(frame_params, "Población:", self.num_poblacion)
        self.create_labeled_entry(frame_params, "Generaciones:", self.num_generaciones)
        self.create_labeled_entry(frame_params, "Nº Parcelas/Slots:", self.num_parcelas_gui) # NUEVO CAMPO
        self.create_labeled_entry(frame_params, "Prob. Mutar Cantidad:", self.prob_mutacion_cantidad_gui) # RENOMBRADO
        self.create_labeled_entry(frame_params, "Prob. Mutar Cultivo:", self.prob_mutar_cultivo_gui)    # NUEVO CAMPO
        self.create_labeled_entry(frame_params, "Prob. Cruce:", self.prob_crossover)
        self.create_labeled_entry(frame_params, "Nº Élite:", self.num_elite)

        ttk.Separator(frame_params, orient='horizontal').pack(fill='x', pady=10, padx=10)

        ttk.Label(frame_params, text="Recursos TOTALES Disponibles", font=("Arial", 11, "bold")).pack(pady=(5,5))
        # Usar las variables renombradas y etiquetas actualizadas
        self.create_labeled_entry(frame_params, "Agua Total (L):", self.total_agua_disponible_gui) # ETIQUETA ACTUALIZADA
        self.create_labeled_entry(frame_params, "Fertilizante Total (kg):", self.total_fertilizante_disponible_gui) # ETIQUETA ACTUALIZADA
        self.create_labeled_entry(frame_params, "Pesticida Total (kg):", self.total_pesticida_disponible_gui) # ETIQUETA ACTUALIZADA

        ttk.Separator(frame_params, orient='horizontal').pack(fill='x', pady=10, padx=10)

        # Botones
        self.button_load = ttk.Button(frame_params, text="Cargar Datos CSV", command=self.load_csv)
        self.button_load.pack(fill=tk.X, padx=10, pady=5)

        self.button_run = ttk.Button(frame_params, text="Optimizar Plan", command= self.ejecutar_ag_con_feedback) # Texto actualizado
        self.button_run.pack(fill=tk.X, padx=10, pady=5)

        self.button_clear = ttk.Button(frame_params, text="Limpiar Todo", command=self.clear_all)
        self.button_clear.pack(fill=tk.X, padx=10, pady=5)


        # --- Contenido del Frame Derecho (Display) ---
        # Frame CSV (sin cambios significativos)
        frame_csv_view = ttk.Frame(frame_display, relief="sunken", borderwidth=1)
        frame_csv_view.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        ttk.Label(frame_csv_view, text="Datos CSV Cargados (Cultivables)").pack(anchor='nw', padx=2, pady=2)
        csv_scroll_y = ttk.Scrollbar(frame_csv_view, orient="vertical")
        csv_scroll_x = ttk.Scrollbar(frame_csv_view, orient="horizontal")
        self.tree_csv_display = ttk.Treeview(frame_csv_view, columns=(), show="headings", height=10, # Altura ajustada
                                             yscrollcommand=csv_scroll_y.set, xscrollcommand=csv_scroll_x.set)
        csv_scroll_y.config(command=self.tree_csv_display.yview)
        csv_scroll_x.config(command=self.tree_csv_display.xview)
        csv_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        csv_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree_csv_display.pack(fill=tk.BOTH, expand=True)


        # Frame Resultados del Plan (más alto)
        self.frame_best_result_display = ttk.Frame(frame_display, relief="sunken", borderwidth=1, height=350) # Más altura
        self.frame_best_result_display.pack(fill=tk.BOTH, expand=True, pady=(5, 0)) # Permitir expandir
        self.frame_best_result_display.pack_propagate(False) # Para mantener tamaño inicial
        ttk.Label(self.frame_best_result_display, text="Mejor Plan Encontrado", font=("Arial", 11, "bold")).pack(pady=5, anchor='nw', padx=5)
        # El Treeview se crea/limpia en show_results_plan


    def create_graph_widgets(self, parent):
        # Crear una figura más alta para acomodar ambos subplots
        self.fig_aptitud = Figure(figsize=(7, 7), dpi=100) # Más altura

        # Subplot superior (2 filas, 1 columna, primer plot) para la aptitud
        self.ax_aptitud = self.fig_aptitud.add_subplot(2, 1, 1)
        self.reset_fitness_graph_appearance() # Renombrado para claridad

        # Subplot inferior (2 filas, 1 columna, segundo plot) para el mapa del plan
        self.ax_plan_map = self.fig_aptitud.add_subplot(2, 1, 2)
        self.reset_plan_map_appearance() # Nueva función para resetear el mapa

        # Ajustar el espacio entre subplots
        self.fig_aptitud.subplots_adjust(hspace=0.4) # Aumentar espacio vertical

        # Crear el canvas y la barra de herramientas (igual que antes)
        self.canvas_aptitud = FigureCanvasTkAgg(self.fig_aptitud, master=parent)
        canvas_widget = self.canvas_aptitud.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))
        self.toolbar_aptitud = NavigationToolbar2Tk(self.canvas_aptitud, toolbar_frame)
        self.toolbar_aptitud.update()

    def reset_fitness_graph_appearance(self): # RENOMBRADO
        """Resetea la apariencia de la gráfica de evolución de aptitud."""
        if self.ax_aptitud:
            self.ax_aptitud.cla()
            self.ax_aptitud.set_title("Evolución de la Mejor Aptitud del Plan")
            self.ax_aptitud.set_xlabel("Generación")
            self.ax_aptitud.set_ylabel("Mejor Aptitud (Plan)")
            self.ax_aptitud.grid(True, linestyle=':', alpha=0.7)
            self.ax_aptitud.text(0.5, 0.5, 'Esperando ejecución...',
                                 ha='center', va='center', transform=self.ax_aptitud.transAxes,
                                 fontsize=10, color='gray')
            
    def reset_plan_map_appearance(self): # NUEVO
        """Resetea la apariencia del mapa del plan de cultivo."""
        if self.ax_plan_map:
            self.ax_plan_map.cla()
            self.ax_plan_map.set_title("Mapa del Mejor Plan de Cultivo")
            self.ax_plan_map.set_xticks([]) # Sin marcas en el eje X por defecto
            self.ax_plan_map.set_yticks([]) # Sin marcas en el eje Y
            # Desactivar los bordes (spines) para un look más limpio
            self.ax_plan_map.spines['top'].set_visible(False)
            self.ax_plan_map.spines['right'].set_visible(False)
            self.ax_plan_map.spines['bottom'].set_visible(False)
            self.ax_plan_map.spines['left'].set_visible(False)
            self.ax_plan_map.text(0.5, 0.5, 'Plan no disponible',
                                 ha='center', va='center', transform=self.ax_plan_map.transAxes,
                                 fontsize=10, color='gray')

    def load_csv(self):
        # Limpiar antes de cargar nuevos datos
        self.clear_all()

        # Usar método de AG que maneja diálogo y errores CSV
        # No necesitamos guardar los datos aquí, AG los tiene
        if self.ag.load_csv() is not None and self.ag.datos_csv is not None and not self.ag.datos_csv.empty:
            data = self.ag.datos_csv # Obtener los datos filtrados de la instancia AG
            self.tree_csv_display.delete(*self.tree_csv_display.get_children())

            cols = list(data.columns)
            self.tree_csv_display["columns"] = cols
            self.tree_csv_display.column("#0", width=0, stretch=tk.NO)
            for col in cols:
                self.tree_csv_display.heading(col, text=col, anchor=tk.W)
                col_width = max(60, min(150, len(col) * 9))
                self.tree_csv_display.column(col, width=col_width, anchor=tk.W, stretch=tk.NO)

            # Limitar filas mostradas para rendimiento si el CSV es muy grande
            max_rows_display = 100
            for i, (_, row) in enumerate(data.iterrows()):
                if i >= max_rows_display:
                    self.tree_csv_display.insert("", tk.END, values=["...", "..."]) # Indicador de más datos
                    break
                values = [str(v) if pd.notna(v) else "" for v in row.values]
                self.tree_csv_display.insert("", tk.END, values=values)

            messagebox.showinfo("Éxito", f"Datos CSV cargados ({len(data)} filas cultivables).")
        elif self.ag.datos_csv is None:
             # El método load_csv de AG ya mostró el error
             pass
        else: # Cargado pero vacío después de filtrar
             messagebox.showwarning("Datos Vacíos", "Archivo cargado, pero sin filas cultivables ('EsCultivable' != 1).")


    def create_labeled_entry(self, parent, text, variable):
        frame = ttk.Frame(parent)
        frame.pack(pady=2, padx=10, fill=tk.X) # Aumentar padx
        label = ttk.Label(frame, text=text, width=20, anchor='w') # Más ancho para etiquetas largas
        label.pack(side=tk.LEFT, padx=(0, 5))
        entry = ttk.Entry(frame, textvariable=variable, width=15) # Un poco más ancho
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return frame


    # MODIFICADO: Para mostrar el PLAN
    def show_results_plan(self, mejor_plan_info):
        """Muestra los detalles del mejor PLAN encontrado."""
        # Limpiar el frame de resultados anterior (excepto la etiqueta del título)
        for widget in self.frame_best_result_display.winfo_children():
            if isinstance(widget, ttk.Label) and "Mejor Plan" in widget.cget("text"):
                continue
            widget.destroy()

        # Crear Treeview para los resultados del plan
        results_tree = ttk.Treeview(self.frame_best_result_display, columns=("Detalle", "Valor"), show="headings", height=15) # Más altura
        results_scroll = ttk.Scrollbar(self.frame_best_result_display, orient="vertical", command=results_tree.yview)
        results_tree.configure(yscrollcommand=results_scroll.set)

        results_tree.heading("Detalle", text="Detalle", anchor=tk.W)
        results_tree.heading("Valor", text="Valor", anchor=tk.W)
        results_tree.column("Detalle", anchor="w", width=200, stretch=tk.NO) # Ancho fijo
        results_tree.column("Valor", anchor="w", width=400) # Más espacio para valores

        if mejor_plan_info and 'plan' in mejor_plan_info:
            plan_list = mejor_plan_info.get('plan', [])
            aptitud = mejor_plan_info.get('aptitud', 0.0)

            # --- Mostrar Información General del Plan ---
            results_tree.insert("", tk.END, values=("Aptitud del Plan", f"{aptitud:.5f}"), tags=('header',))
            plan_str = " -> ".join([p.get('cultivo_nombre', '?') for p in plan_list])
            results_tree.insert("", tk.END, values=("Secuencia Plan", plan_str))

            # --- Calcular y Mostrar Totales Usados (Recalculados aquí) ---
            agua_total = sum(p.get('cantidad_agua', 0) for p in plan_list)
            fert_total = sum(p.get('cantidad_fertilizante', 0) for p in plan_list)
            pest_total = sum(p.get('cantidad_pesticida', 0) for p in plan_list)
            costo_total = sum(p.get('original_costo', 0) for p in plan_list) # Base
            retorno_total = sum(p.get('original_retorno_economico', 0) for p in plan_list) # Base

            results_tree.insert("", tk.END, values=("--- Totales Usados ---", ""), tags=('separator',))
            results_tree.insert("", tk.END, values=("Agua Total", f"{agua_total:.2f} / {self.total_agua_disponible_gui.get():.2f}"))
            results_tree.insert("", tk.END, values=("Fertilizante Total", f"{fert_total:.2f} / {self.total_fertilizante_disponible_gui.get():.2f}"))
            results_tree.insert("", tk.END, values=("Pesticida Total", f"{pest_total:.2f} / {self.total_pesticida_disponible_gui.get():.2f}"))
            results_tree.insert("", tk.END, values=("Costo Total (Base)", f"{costo_total:.2f}"))
            results_tree.insert("", tk.END, values=("Retorno Total (Base)", f"{retorno_total:.2f}"))


            # --- Mostrar Detalles por Parcela ---
            for i, parcela in enumerate(plan_list):
                results_tree.insert("", tk.END, values=(f"--- Parcela {i+1} ---", ""), tags=('separator',))
                results_tree.insert("", tk.END, values=(f"  Cultivo", parcela.get('cultivo_nombre', 'N/A')))

                # Función auxiliar para formatear valores
                def format_value(key, default=0, decimals=2):
                    val = parcela.get(key, default)
                    try: return f"{float(val):.{decimals}f}"
                    except: return str(val)

                results_tree.insert("", tk.END, values=(f"  Agua Asig.", f"{format_value('cantidad_agua')} (Orig: {format_value('original_agua')})"))
                results_tree.insert("", tk.END, values=(f"  Fertiliz. Asig.", f"{format_value('cantidad_fertilizante')} (Orig: {format_value('original_fertilizante')})"))
                results_tree.insert("", tk.END, values=(f"  Pesticida Asig.", f"{format_value('cantidad_pesticida')} (Orig: {format_value('original_pesticida')})"))
                results_tree.insert("", tk.END, values=(f"  Costo Base", format_value('original_costo')))
                results_tree.insert("", tk.END, values=(f"  Retorno Base", format_value('original_retorno_economico')))
                results_tree.insert("", tk.END, values=(f"  Adaptabilidad", format_value('original_adaptabilidad_clima')))

        else:
            results_tree.insert("", tk.END, values=("Resultado", "Plan no disponible"))

        # Estilos para las filas (opcional)
        results_tree.tag_configure('header', background='#D9E5FF', font=('Arial', 10, 'bold'))
        results_tree.tag_configure('separator', background='#E0E0E0')


        results_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(0,5), padx=(0,5))
        results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0,5))


    def clear_all(self):
        print("Limpiando interfaz...")
        # Limpiar tabla de resultados
        self.show_results_plan(None)

        # Limpiar tabla CSV
        if hasattr(self, 'tree_csv_display'):
            self.tree_csv_display.delete(*self.tree_csv_display.get_children())

        # Resetear gráficas
        self.reset_fitness_graph_appearance()
        self.reset_plan_map_appearance() # NUEVO

        # Limpiar diccionario de colores
        self.crop_colors = {}

        # Dibujar el canvas una vez reseteados ambos ejes
        if self.canvas_aptitud:
            self.fig_aptitud.tight_layout() # Ajustar layout
            self.canvas_aptitud.draw_idle()


    def update_plots(self): # RENOMBRADO Y MODIFICADO
        """Actualiza tanto la gráfica de aptitud como el mapa del plan."""
        if not hasattr(self.ag, 'mejor_aptitud_historial'):
             print("AG no tiene 'mejor_aptitud_historial'")
             return
        if not hasattr(self.ag, 'mejor_plan_global'):
            print("AG no tiene 'mejor_plan_global'")
            return

        historial = self.ag.mejor_aptitud_historial
        mejor_plan_info = self.ag.mejor_plan_global

        # --- 1. Actualizar Gráfica de Aptitud (como antes) ---
        self.ax_aptitud.cla() # Limpiar eje de aptitud
        if historial:
            generaciones = range(1, len(historial) + 1)
            self.ax_aptitud.plot(generaciones, historial, marker='.', linestyle='-', markersize=5, color='b', label='Mejor Aptitud Plan')

            max_apt = max(historial)
            if max_apt > 0: # Solo mostrar si hay aptitud positiva
                max_gen_idx = historial.index(max_apt)
                self.ax_aptitud.plot(max_gen_idx + 1, max_apt, 'r*', markersize=10, label=f'Max ({max_apt:.5f})')

            self.ax_aptitud.legend(fontsize=8)
            self.ax_aptitud.set_xlim(left=0)
        else:
             self.ax_aptitud.text(0.5, 0.5, 'No hay datos de aptitud para graficar.',
                                  ha='center', va='center', transform=self.ax_aptitud.transAxes, color='gray')

        self.ax_aptitud.set_title("Evolución de la Mejor Aptitud del Plan")
        self.ax_aptitud.set_xlabel("Generación")
        self.ax_aptitud.set_ylabel("Mejor Aptitud (Plan)")
        self.ax_aptitud.grid(True, linestyle=':', alpha=0.7)

        # --- 2. Actualizar Mapa del Plan ---
        self.ax_plan_map.cla() # Limpiar eje del mapa
        self.reset_plan_map_appearance() # Poner texto por defecto por si no hay plan

        if mejor_plan_info and 'plan' in mejor_plan_info:
            plan_list = mejor_plan_info.get('plan', [])
            if plan_list: # Solo dibujar si hay parcelas en el plan
                num_plots = len(plan_list)

                # Asignar colores únicos a los cultivos si no existen
                unique_crops = sorted(list(set(p.get('cultivo_nombre', 'N/A') for p in plan_list)))
                # Usar un colormap para obtener colores distintos
                color_map = cm.get_cmap('tab20', len(unique_crops)) # 'tab10', 'tab20', 'viridis', etc.
                current_colors = 0
                for crop in unique_crops:
                    if crop not in self.crop_colors:
                        self.crop_colors[crop] = color_map(current_colors)
                        current_colors += 1

                # Dibujar cada parcela como un rectángulo coloreado
                for i, parcela in enumerate(plan_list):
                    cultivo_nombre = parcela.get('cultivo_nombre', 'N/A')
                    color = self.crop_colors.get(cultivo_nombre, 'lightgrey') # Gris si algo falla

                    # Crear rectángulo (posición x,y, ancho, alto)
                    # Ancho 0.9 para dejar un pequeño espacio
                    rect = patches.Rectangle((i, 0), 0.9, 1,
                                             facecolor=color,
                                             edgecolor='black',
                                             linewidth=0.5)
                    self.ax_plan_map.add_patch(rect)

                    # Añadir nombre del cultivo dentro del rectángulo
                    # Determinar color del texto para contraste
                    try:
                         # Simple heurística basada en luminancia (funciona mejor con RGB)
                         rgb_color = color[:3] # Ignorar alpha si existe
                         luminance = 0.299*rgb_color[0] + 0.587*rgb_color[1] + 0.114*rgb_color[2]
                         text_color = 'white' if luminance < 0.5 else 'black'
                    except:
                         text_color = 'black' # Default

                    self.ax_plan_map.text(i + 0.45, 0.5, cultivo_nombre, # Centrado en x=i+0.45
                                          ha='center', va='center', fontsize=8, color=text_color,
                                          rotation=90) # Rotar para nombres largos

                # Configurar ejes del mapa
                self.ax_plan_map.set_xlim(-0.1, num_plots - 0.1 + 0.1) # Ajustar límites X
                self.ax_plan_map.set_ylim(-0.1, 1.1) # Ajustar límites Y
                self.ax_plan_map.set_xticks(np.arange(num_plots) + 0.45) # Posicionar ticks en el centro
                self.ax_plan_map.set_xticklabels([f"P{j+1}" for j in range(num_plots)], fontsize=8)
                self.ax_plan_map.set_yticks([]) # Ocultar eje Y
                self.ax_plan_map.set_title("Mapa del Mejor Plan de Cultivo")
                # Ocultar bordes que no sean el inferior
                self.ax_plan_map.spines['top'].set_visible(False)
                self.ax_plan_map.spines['right'].set_visible(False)
                # self.ax_plan_map.spines['bottom'].set_visible(True) # Dejar el inferior con las etiquetas P1, P2...
                self.ax_plan_map.spines['left'].set_visible(False)


        # --- 3. Ajustar Layout y Dibujar Canvas ---
        self.fig_aptitud.tight_layout(pad=2.0) # Ajustar para evitar solapamientos, con padding
        self.canvas_aptitud.draw_idle() # Dibujar los cambios


    def ejecutar_ag_con_feedback(self):
        if self.ag.datos_csv is None or self.ag.datos_csv.empty:
            messagebox.showerror("Error", "Por favor, carga un archivo CSV cultivable primero.")
            return

        # --- Actualizar parámetros del AG desde la GUI ---
        try:
            pob = self.num_poblacion.get()
            gen = self.num_generaciones.get()
            # Leer variables renombradas y nuevas
            mut_cant = self.prob_mutacion_cantidad_gui.get()
            mut_cult = self.prob_mutar_cultivo_gui.get()
            cross = self.prob_crossover.get()
            elite = self.num_elite.get()
            parcelas = self.num_parcelas_gui.get() # Leer número de parcelas
            agua_tot = self.total_agua_disponible_gui.get() # Leer totales
            fert_tot = self.total_fertilizante_disponible_gui.get()
            pest_tot = self.total_pesticida_disponible_gui.get()

            # Validaciones
            if pob <= 0 or gen <= 0 or parcelas <= 0: raise ValueError("Población, Generaciones y Parcelas deben ser > 0")
            if not (0.0 <= mut_cant <= 1.0 and 0.0 <= mut_cult <= 1.0 and 0.0 <= cross <= 1.0): raise ValueError("Probabilidades entre 0 y 1")
            if elite < 0: raise ValueError("Élite debe ser >= 0")
            if elite >= pob:
                elite = max(0, pob - 1)
                self.num_elite.set(elite) # Actualizar GUI si se ajusta
                messagebox.showwarning("Ajuste", f"Élite ({self.num_elite.get()}) >= Población ({pob}). Ajustado a {elite}.")


            # Actualizar la instancia AG con los atributos correctos
            self.ag.num_poblacion = pob
            self.ag.num_generaciones = gen
            self.ag.prob_mutacion_cantidad = mut_cant # Usar nombre correcto del atributo AG
            self.ag.prob_mutar_cultivo = mut_cult     # Usar nombre correcto del atributo AG
            self.ag.prob_crossover = cross
            self.ag.num_elite = elite
            self.ag.num_parcelas = parcelas             # Usar nombre correcto del atributo AG
            self.ag.total_agua_disponible = agua_tot    # Usar nombre correcto del atributo AG
            self.ag.total_fertilizante_disponible = fert_tot # Usar nombre correcto del atributo AG
            self.ag.total_pesticida_disponible = pest_tot   # Usar nombre correcto del atributo AG

            print("Parámetros AG actualizados desde GUI para optimización de plan.")

        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Error de Parámetro", f"Valor inválido: {e}")
            return

        # --- Ejecución ---
        self.ventana.update_idletasks()
        self.button_run.config(state=tk.DISABLED, text="Optimizando...")
        self.button_load.config(state=tk.DISABLED)
        self.button_clear.config(state=tk.DISABLED)
        self.ventana.config(cursor="watch")

        mejor_plan_encontrado = None
        try:
            print(f"Iniciando AG para Plan ({self.ag.num_generaciones} generaciones, {self.ag.num_parcelas} parcelas)...")
            mejor_plan_encontrado = self.ag.optimizar()
            print("Optimización de Plan AG finalizada.")

            # Actualizar GUI
            self.show_results_plan(mejor_plan_encontrado)
            self.update_plots() # LLAMADA MODIFICADA

            if not mejor_plan_encontrado or mejor_plan_encontrado.get('aptitud', 0.0) <= 0: # Chequeo más robusto
                 messagebox.showinfo("Información", "La optimización no produjo un plan válido con aptitud positiva.")

        except Exception as e:
            messagebox.showerror("Error en Ejecución", f"Ocurrió un error durante la optimización del plan:\n{e}")
            traceback.print_exc() # Imprime detalles en consola
            self.clear_all()

        finally:
            self.button_run.config(state=tk.NORMAL, text="Optimizar Plan")
            self.button_load.config(state=tk.NORMAL)
            self.button_clear.config(state=tk.NORMAL)
            self.ventana.config(cursor="")
            print("Interfaz rehabilitada.")


# --- Bloque principal ---
if __name__ == "__main__":
    mi_ventana = tk.Tk()
    app = GUI(mi_ventana)
    mi_ventana.mainloop()