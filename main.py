import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Descomentar si se usa la gráfica
import tkinter as tk
from tkinter import filedialog, messagebox

class AG:

    def __init__(self, num_poblacion, num_generaciones, prob_mutacion, prob_crossover, num_elite, cantidad_agua_max, cantidad_fertilizante_max, cantidad_pesticida_max): # Renombrados para claridad
        self.num_poblacion = num_poblacion
        self.num_generaciones = num_generaciones
        self.prob_mutacion = prob_mutacion
        self.prob_crossover = prob_crossover
        self.num_elite = num_elite
        # Límites MÁXIMOS para la mutación (pueden ser diferentes a los máx del CSV)
        self.cantidad_agua_max = cantidad_agua_max
        self.cantidad_fertilizante_max = cantidad_fertilizante_max
        self.cantidad_pesticida_max = cantidad_pesticida_max
        self.datos_csv = None
        # Para normalización en la aptitud, calculados después de cargar CSV
        self.max_agua_csv = 1
        self.max_fert_csv = 1
        self.max_pest_csv = 1
        self.max_costo_csv = 1
        self.max_prod_csv = 1
        self.mejor_aptitud_historial = []

    def load_csv(self):
        """Abre un diálogo para seleccionar y cargar el archivo CSV."""
        file_path = filedialog.askopenfilename(
            title="Selecciona el archivo CSV de datos de cultivos",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            print("No se seleccionó ningún archivo.")
            return None

        try:
            # Especificar dtype para columnas potencialmente problemáticas si es necesario
            # dtype_dict = {'ColumnaConflictiva': str} # Ejemplo
            # self.datos_csv = pd.read_csv(file_path, dtype=dtype_dict)
            self.datos_csv = pd.read_csv(file_path)

            # --- Verificación de columnas esenciales ---
            required_cols = ['Cultivo', 'Agua', 'Fertilizante', 'Pesticida', 'Costo',
                             'Produccion', 'EsCultivable', 'CultivoAnterior',
                             'AdaptabilidadClima', 'RetornoEconomico']
            missing_cols = [col for col in required_cols if col not in self.datos_csv.columns]
            if missing_cols:
                 messagebox.showerror("Error", f"Columnas requeridas faltantes en el CSV: {', '.join(missing_cols)}")
                 self.datos_csv = None
                 return None

            # Convertir columnas numéricas, manejando errores
            numeric_cols = ['Agua', 'Fertilizante', 'Pesticida', 'Costo', 'Produccion',
                            'EsCultivable', 'AdaptabilidadClima', 'RetornoEconomico']
            for col in numeric_cols:
                self.datos_csv[col] = pd.to_numeric(self.datos_csv[col], errors='coerce')
                # Opcional: Rellenar NaNs si la conversión falla y quieres permitirlo
                # self.datos_csv[col] = self.datos_csv[col].fillna(0) # O algún otro valor por defecto

            # Comprobar si alguna conversión falló y creó NaNs donde no deberían
            if self.datos_csv[numeric_cols].isnull().any().any():
                 print("Advertencia: Se encontraron valores no numéricos en columnas que deberían serlo. Revise el CSV.")
                 # Podrías decidir detenerte o continuar, aquí continuamos pero filtramos NaNs en EsCultivable
                 # messagebox.showwarning("Advertencia", "Se encontraron valores no numéricos en columnas que deberían serlo. Revise el CSV.")

            # --- Filtrado Crucial ---
            # Asegurarse que EsCultivable sea numérico antes de comparar
            self.datos_csv = self.datos_csv.dropna(subset=['EsCultivable']) # Eliminar filas donde EsCultivable es NaN
            self.datos_csv = self.datos_csv[self.datos_csv["EsCultivable"] == 1].copy()

            if self.datos_csv.empty:
                messagebox.showerror("Error", "El archivo CSV no contiene filas válidas con 'EsCultivable' == 1 después de la limpieza.")
                self.datos_csv = None
                return None

            print("Datos CSV cargados y filtrados (solo cultivables):")
            print(self.datos_csv.head()) # Mostrar solo las primeras filas
            print(f"Número de filas cultivables cargadas: {len(self.datos_csv)}")
            print("-" * 30)

            # Calcular máximos del CSV para normalización DESPUÉS de filtrar
            self.max_agua_csv = self.datos_csv['Agua'].max(skipna=True)
            self.max_fert_csv = self.datos_csv['Fertilizante'].max(skipna=True)
            self.max_pest_csv = self.datos_csv['Pesticida'].max(skipna=True)
            self.max_costo_csv = self.datos_csv['Costo'].max(skipna=True)
            self.max_prod_csv = self.datos_csv['Produccion'].max(skipna=True)

            self.max_agua_csv = 1 if pd.isna(self.max_agua_csv) else self.max_agua_csv
            self.max_fert_csv = 1 if pd.isna(self.max_fert_csv) else self.max_fert_csv
            self.max_pest_csv = 1 if pd.isna(self.max_pest_csv) else self.max_pest_csv
            self.max_costo_csv = 1 if pd.isna(self.max_costo_csv) else self.max_costo_csv
            self.max_prod_csv = 1 if pd.isna(self.max_prod_csv) else self.max_prod_csv
            # Evitar división por cero
            self.max_agua_csv = max(self.max_agua_csv, 1)
            self.max_fert_csv = max(self.max_fert_csv, 1)
            self.max_pest_csv = max(self.max_pest_csv, 1)
            self.max_costo_csv = max(self.max_costo_csv, 1)
            self.max_prod_csv = max(self.max_prod_csv, 1)

    


            return self.datos_csv
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el archivo: {file_path}")
            return None
        except pd.errors.EmptyDataError:
             messagebox.showerror("Error", f"El archivo CSV está vacío: {file_path}")
             return None
        # Quitamos KeyError porque ya lo verificamos manualmente
        # except KeyError as e:
        #      messagebox.showerror("Error", f"Columna requerida no encontrada en el CSV: {e}. Verifica las columnas: {required_cols}")
        #      return None
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado al cargar el archivo: {e}")
            # Imprimir traceback para depuración
            import traceback
            traceback.print_exc()
            return None

    def generar_individuo(self):
        """Genera un individuo seleccionando una fila aleatoria del CSV y almacenando valores originales."""
        if self.datos_csv is None or self.datos_csv.empty:
            raise ValueError("No se han cargado datos válidos del CSV.")

        fila = self.datos_csv.sample().iloc[0]

        cultivo = fila["Cultivo"]

        # *** Guardar valores ORIGINALES del CSV ***
        original_agua = fila["Agua"]
        original_fertilizante = fila["Fertilizante"]
        original_pesticida = fila["Pesticida"]
        original_costo = fila["Costo"]
        original_produccion = fila["Produccion"]
        original_retorno = fila["RetornoEconomico"] 

        # Crear el diccionario del individuo
        individuo = {
            # --- Valores Originales (ancla para evaluación) ---
            "cultivo_nombre": cultivo, # Identificador clave
            "original_agua": original_agua,
            "original_fertilizante": original_fertilizante,
            "original_pesticida": original_pesticida,
            "original_costo": original_costo,
            "original_produccion": original_produccion,
            "original_retorno_economico": original_retorno,
            # --- Valores Mutables (sujetos a evolución) ---
            # Inicializar con los originales, pero pueden cambiar
            "cantidad_agua": original_agua,
            "cantidad_fertilizante": original_fertilizante,
            "cantidad_pesticida": original_pesticida,
            # --- Otros datos (generalmente fijos o para info) ---
            # "cultivos": [cultivo] # Cambiado a cultivo_nombre para evitar listas de un solo elemento
            "aptitud": 0.0 # Inicializar aptitud
        }
        return individuo

    def incializar_poblacion(self):
        """Crea la población inicial."""
        if self.datos_csv is None or self.datos_csv.empty:
             raise ValueError("No se pueden inicializar la población sin datos CSV cargados.")
        poblacion = [self.generar_individuo() for _ in range(self.num_poblacion)]
        return poblacion

    def verificar_rotacion(self, nombre_cultivo_actual):
        """Verifica si el cultivo actual es diferente al anterior requerido, usando el CSV."""
        if self.datos_csv is None:
             return True # No se puede verificar sin datos

        # Busca la fila correspondiente al cultivo actual en los datos cargados
        filas_cultivo = self.datos_csv[self.datos_csv["Cultivo"] == nombre_cultivo_actual]

        if filas_cultivo.empty:
             print(f"Advertencia: Cultivo '{nombre_cultivo_actual}' no encontrado en datos CSV para verificación de rotación.")
             return False # Considerarlo no válido si no se encuentra

        # Usamos la primera fila encontrada para la regla de rotación de este tipo de cultivo
        fila = filas_cultivo.iloc[0]
        cultivo_anterior_requerido = fila["CultivoAnterior"]

        if pd.isna(cultivo_anterior_requerido) or cultivo_anterior_requerido == "Ninguno" or cultivo_anterior_requerido == "":
            return True # No hay restricción específica
        else:
            # La rotación es válida si el cultivo actual NO es igual al requerido como anterior
            return nombre_cultivo_actual != cultivo_anterior_requerido


    # ... (dentro de la clase AG) ...

    def calcular_aptitud(self, individuo):
        """
        Calcula la aptitud. Combina enfoques:
        1. La producción depende de la desviación RELATIVA de recursos (respecto a base).
        2. Se penaliza la desviación ABSOLUTA de recursos (normalizada por disponible).
        """
        if self.datos_csv is None or self.datos_csv.empty:
            raise ValueError("No se pueden calcular aptitudes sin datos CSV cargados.")

        cultivo_actual = individuo["cultivo_nombre"]
        filas_cultivo = self.datos_csv[self.datos_csv["Cultivo"] == cultivo_actual]

        if filas_cultivo.empty:
             print(f"Advertencia: Cultivo '{cultivo_actual}' no encontrado en datos CSV para cálculo de aptitud. Aptitud será 0.")
             return 0.0

        fila_base = filas_cultivo.iloc[0]

        # --- Obtener valores BASE del CSV (Originales) ---
        agua_base = fila_base["Agua"]
        fertilizante_base = fila_base["Fertilizante"]
        pesticida_base = fila_base["Pesticida"]
        costo_base = fila_base["Costo"]
        produccion_base = fila_base["Produccion"] # Producción si se usan recursos base
        adaptabilidad_clima = fila_base["AdaptabilidadClima"]
        retorno_economico = fila_base["RetornoEconomico"]

        # --- Obtener valores EVOLUCIONADOS del individuo ---
        cantidad_agua_ind = individuo["cantidad_agua"]
        cantidad_fertilizante_ind = individuo["cantidad_fertilizante"]
        cantidad_pesticida_ind = individuo["cantidad_pesticida"]

        # --- INICIO: Modelo de Producción Dependiente de Recursos (Como antes) ---
        factor_ajuste_prod = 1.0# Puedes ajustar este factor (e.g., 1.0, 1.5, 2.0)

        dev_agua_rel = abs(cantidad_agua_ind - agua_base) / agua_base if agua_base > 0 else (0 if cantidad_agua_ind == 0 else float('inf'))
        dev_fert_rel = abs(cantidad_fertilizante_ind - fertilizante_base) / fertilizante_base if fertilizante_base > 0 else (0 if cantidad_fertilizante_ind == 0 else float('inf'))
        dev_pest_rel = abs(cantidad_pesticida_ind - pesticida_base) / pesticida_base if pesticida_base > 0 else (0 if cantidad_pesticida_ind == 0 else float('inf'))

        if float('inf') in [dev_agua_rel, dev_fert_rel, dev_pest_rel]:
            avg_rel_dev = float('inf')
        else:
            avg_rel_dev = (dev_agua_rel + dev_fert_rel + dev_pest_rel) / 3.0

        if avg_rel_dev == float('inf'):
             produccion_ajustada = 0.0
        else:
             reduccion_prod = factor_ajuste_prod * (avg_rel_dev ** 2)
             produccion_ajustada = produccion_base * max(0.0, 1.0 - reduccion_prod)
        # --- FIN: Modelo de Producción Dependiente de Recursos ---


        # --- Pesos ---
        peso_produccion = 0.20
        peso_agua_base = 0.10
        peso_fertilizante_base = 0.10
        peso_pesticida_base = 0.10
        peso_costo_base = 0.10
        peso_adaptabilidad = 0.10
        peso_retorno = 0.20
        # *** NUEVO/REINTRODUCIDO: Peso para la penalización por desviación ***
        peso_desviacion_recursos = 0.1 # Ajustar este valor (quizás 0.01, 0.05, 0.1)
        # *** ELIMINADO: peso_uso_absoluto (ya no se usa en esta versión) ***
        peso_uso_absoluto = 0.05

        # --- Normalización para Aptitud Base (Usa Producción Ajustada) ---
        produccion_norm = produccion_ajustada / self.max_prod_csv
        agua_norm_base = 1.0 - (agua_base / self.max_agua_csv)
        fert_norm_base = 1.0 - (fertilizante_base / self.max_fert_csv)
        pest_norm_base = 1.0 - (pesticida_base / self.max_pest_csv)
        costo_norm_base = 1.0 - (costo_base / self.max_costo_csv)
        adaptabilidad_norm = adaptabilidad_clima
        retorno_norm = retorno_economico

        # --- Cálculo de la aptitud BASE ---
        aptitud_base = (
            peso_produccion * produccion_norm +
            peso_agua_base * agua_norm_base +
            peso_fertilizante_base * fert_norm_base +
            peso_pesticida_base * pest_norm_base +
            peso_costo_base * costo_norm_base +
            peso_adaptabilidad * adaptabilidad_norm +
            peso_retorno * retorno_norm
        )

        # --- INICIO: Penalización por Desviación (Solución A - Normalizada por Disponible) ---
        denominador_agua = max(self.cantidad_agua_max, 1)
        denominador_fert = max(self.cantidad_fertilizante_max, 1)
        denominador_pest = max(self.cantidad_pesticida_max, 1)

        # Calcula la desviación absoluta entre evolucionado y base, normalizada por el recurso TOTAL disponible
        desv_agua_norm_disp = abs(cantidad_agua_ind - agua_base) / denominador_agua
        desv_fert_norm_disp = abs(cantidad_fertilizante_ind - fertilizante_base) / denominador_fert
        desv_pest_norm_disp = abs(cantidad_pesticida_ind - pesticida_base) / denominador_pest

        penalizacion_desviacion = (desv_agua_norm_disp + desv_fert_norm_disp + desv_pest_norm_disp) / 3.0
        # --- FIN: Penalización por Desviación ---
        uso_agua_norm = cantidad_agua_ind / denominador_agua
        uso_fert_norm = cantidad_fertilizante_ind / denominador_fert
        uso_pest_norm = cantidad_pesticida_ind / denominador_pest

        penalizacion_uso = (uso_agua_norm + uso_fert_norm + uso_pest_norm) / 3.0
        # --- Penalización por Rotación (como antes) ---
        penalizacion_rotacion = 1.0
        aplica_penal_rot = not self.verificar_rotacion(cultivo_actual)
        if aplica_penal_rot:
            penalizacion_rotacion = 0.06 # O el valor que prefieras

        # --- Cálculo de la Aptitud Final (usando penalizacion_desviacion) ---
        aptitud_antes_max = (aptitud_base
                     - (peso_desviacion_recursos * penalizacion_desviacion)
                     - (peso_uso_absoluto * penalizacion_uso)
                    ) * penalizacion_rotacion
        aptitud_final = max(0.0, aptitud_antes_max)

        # (Puedes volver a añadir prints de DEBUG aquí si lo necesitas)

        return aptitud_final

# ... (resto de la clase AG sin cambios) ...
    def evaluar_poblacion(self, poblacion):
        """Calcula y asigna la aptitud a cada individuo de la población."""
        for individuo in poblacion:
            # Calcular la aptitud usando la función corregida
            individuo["aptitud"] = self.calcular_aptitud(individuo)
            # Asegurar que aptitud sea float
            if not isinstance(individuo["aptitud"], float):
                 individuo["aptitud"] = float(individuo["aptitud"])
        return poblacion

    def seleccion_por_torneo(self, poblacion, tamaño_torneo=3):
        """Selecciona dos padres usando el método de torneo."""
        n_poblacion = len(poblacion)
        if n_poblacion == 0:
            raise ValueError("La población está vacía, no se puede seleccionar.")

        # Asegurarse de no pedir más individuos de los que hay
        tamaño_real_torneo = min(tamaño_torneo, n_poblacion)

        # Seleccionar índices únicos para cada torneo
        indices1 = np.random.choice(n_poblacion, size=tamaño_real_torneo, replace=False)
        indices2 = np.random.choice(n_poblacion, size=tamaño_real_torneo, replace=False)

        # Obtener los individuos participantes
        torneo1 = [poblacion[i] for i in indices1]
        torneo2 = [poblacion[i] for i in indices2]

        # Elegir al mejor de cada torneo (el que tenga mayor aptitud)
        # Manejar el caso de aptitudes iguales seleccionando el primero encontrado
        padre1 = max(torneo1, key=lambda ind: ind.get("aptitud", 0.0))
        padre2 = max(torneo2, key=lambda ind: ind.get("aptitud", 0.0))

        return padre1, padre2

    def cruzar(self, padre1, padre2):
        """Realiza el cruce entre dos padres."""
        # Copiar a los padres para crear hijos
        # Usar deepcopy si los diccionarios fueran más complejos, pero aquí copy() es suficiente
        hijo1 = padre1.copy()
        hijo2 = padre2.copy()

        if np.random.rand() < self.prob_crossover:
            # Punto de cruce simple: intercambiar uno de los genes MUTABLES
            genes_mutables = ["cantidad_agua", "cantidad_fertilizante", "cantidad_pesticida"]
            gen_a_cruzar = np.random.choice(genes_mutables)

            # Intercambiar el valor del gen mutable seleccionado
            temp = hijo1[gen_a_cruzar]
            hijo1[gen_a_cruzar] = hijo2[gen_a_cruzar]
            hijo2[gen_a_cruzar] = temp

            # La aptitud de los hijos deberá ser recalculada en la siguiente evaluación
            hijo1['aptitud'] = 0.0
            hijo2['aptitud'] = 0.0

        return hijo1, hijo2

    def mutar(self, individuo, prob_mutar_cultivo=0.05): # Añadir probabilidad de mutar cultivo
        """Aplica mutación a genes mutables y potencialmente al tipo de cultivo."""
        individuo_mutado = individuo.copy()

        # --- Mutación de Cultivo ---
        if np.random.rand() < prob_mutar_cultivo:
            if self.datos_csv is not None and not self.datos_csv.empty:
                # Seleccionar un NUEVO cultivo aleatorio del CSV (diferente al actual)
                cultivos_disponibles = self.datos_csv['Cultivo'].unique()
                cultivo_actual = individuo_mutado['cultivo_nombre']
                # Asegurarse de no elegir el mismo (a menos que solo haya 1 opción)
                posibles_nuevos = [c for c in cultivos_disponibles if c != cultivo_actual]
                if posibles_nuevos:
                    nuevo_cultivo_nombre = np.random.choice(posibles_nuevos)
                    # Obtener la fila del CSV para el nuevo cultivo
                    # ¡Importante! Decidir cómo manejar múltiples filas para el mismo cultivo. Usar la primera encontrada es lo más simple.
                    nueva_fila = self.datos_csv[self.datos_csv['Cultivo'] == nuevo_cultivo_nombre].iloc[0]

                    print(f"*** MUTACIÓN DE CULTIVO: {cultivo_actual} -> {nuevo_cultivo_nombre} ***") # Para depuración

                    # Actualizar TODOS los valores relevantes en el individuo mutado
                    individuo_mutado['cultivo_nombre'] = nuevo_cultivo_nombre
                    individuo_mutado['original_agua'] = nueva_fila["Agua"]
                    individuo_mutado['original_fertilizante'] = nueva_fila["Fertilizante"]
                    individuo_mutado['original_pesticida'] = nueva_fila["Pesticida"]
                    individuo_mutado['original_costo'] = nueva_fila["Costo"]
                    individuo_mutado['original_produccion'] = nueva_fila["Produccion"]
                    individuo_mutado['original_retorno_economico'] = nueva_fila["RetornoEconomico"]
                    # Resetear también los mutables a los nuevos originales
                    individuo_mutado['cantidad_agua'] = nueva_fila["Agua"]
                    individuo_mutado['cantidad_fertilizante'] = nueva_fila["Fertilizante"]
                    individuo_mutado['cantidad_pesticida'] = nueva_fila["Pesticida"]


        # --- Mutación de Cantidades (como antes) ---
        # Puede ocurrir independientemente de la mutación de cultivo
        if np.random.rand() < self.prob_mutacion:
            individuo_mutado["cantidad_agua"] = np.random.uniform(0, self.cantidad_agua_max)
            # print(f"Mutacion agua: {individuo['cantidad_agua']} -> {individuo_mutado['cantidad_agua']}") # Debug

        if np.random.rand() < self.prob_mutacion:
            individuo_mutado["cantidad_fertilizante"] = np.random.uniform(0, self.cantidad_fertilizante_max)
            # print(f"Mutacion fert: {individuo['cantidad_fertilizante']} -> {individuo_mutado['cantidad_fertilizante']}") # Debug

        if np.random.rand() < self.prob_mutacion:
             individuo_mutado["cantidad_pesticida"] = np.random.uniform(0, self.cantidad_pesticida_max)
             # print(f"Mutacion pest: {individuo['cantidad_pesticida']} -> {individuo_mutado['cantidad_pesticida']}") # Debug

        # La aptitud deberá ser recalculada en la siguiente evaluación
        individuo_mutado['aptitud'] = 0.0 # Resetear aptitud tras cualquier mutación

        return individuo_mutado
    def generar_nueva_poblacion(self, poblacion):
        """Genera la siguiente población."""
        nueva_poblacion = []
        n_poblacion = len(poblacion)

        # Ordenar la población actual por aptitud descendente para elitismo
        # Usar .get con default 0.0 por si alguna aptitud no se calculó
        poblacion_ordenada = sorted(poblacion, key=lambda ind: ind.get("aptitud", 0.0), reverse=True)

        # 1. Elitismo
        num_elite_real = min(self.num_elite, n_poblacion)
        nueva_poblacion.extend(poblacion_ordenada[:num_elite_real])

        # 2. Generar el resto usando Selección, Cruce y Mutación
        while len(nueva_poblacion) < self.num_poblacion:
            if not poblacion: # Chequeo por si la población original era más pequeña que la élite
                break
            padre1, padre2 = self.seleccion_por_torneo(poblacion_ordenada) # Usar ordenada puede dar ligera ventaja
            hijo1, hijo2 = self.cruzar(padre1, padre2)
            hijo1 = self.mutar(hijo1)
            hijo2 = self.mutar(hijo2)

            nueva_poblacion.append(hijo1)
            # Añadir el segundo hijo solo si aún cabe en la población
            if len(nueva_poblacion) < self.num_poblacion:
                nueva_poblacion.append(hijo2)

        return nueva_poblacion

    
    def optimizar(self):
        """Ejecuta el algoritmo genético completo."""
        if self.datos_csv is None or self.datos_csv.empty:
             messagebox.showerror("Error", "No se han cargado datos CSV válidos. Ejecuta load_csv() primero.")
             return None

        print("Iniciando optimización con Algoritmo Genético (versión corregida)...")
        print(f"Parámetros: Población={self.num_poblacion}, Generaciones={self.num_generaciones}, "
              f"Mutación={self.prob_mutacion}, Cruce={self.prob_crossover}, Elite={self.num_elite}")
        print(f"Límites max para mutación: Agua={self.cantidad_agua_max}, Fert={self.cantidad_fertilizante_max}, Pest={self.cantidad_pesticida_max}")
        print("-" * 30)


      

        try:
            # Inicializar población
            poblacion_actual = self.incializar_poblacion()
            self.mejor_aptitud_historial = []
            num_competidores_a_mostrar = 5 # Cuántos competidores (cultivos únicos) mostrar

            # --- Ciclo evolutivo ---
            for generacion in range(self.num_generaciones):
                # 1. Evaluar la población actual (calcula aptitud para todos)
                poblacion_actual = self.evaluar_poblacion(poblacion_actual)

                # --- Mostrar Información de la Generación ---
                if not poblacion_actual:
                    print(f"Generación {generacion + 1}/{self.num_generaciones}: ¡Población vacía!")
                    break

                # Ordenar la población evaluada por aptitud para encontrar al mejor y a los competidores
                poblacion_ordenada_evaluada = sorted(poblacion_actual, key=lambda ind: ind.get("aptitud", 0.0), reverse=True)

                # a) Mostrar el Mejor Absoluto
                mejor_de_generacion = poblacion_ordenada_evaluada[0]
                aptitud_mejor = mejor_de_generacion.get("aptitud", 0.0)
                self.mejor_aptitud_historial.append(aptitud_mejor if isinstance(aptitud_mejor, (int, float)) else 0.0) # Guardar para gráfica

                if isinstance(aptitud_mejor, (int, float)):
                     aptitud_str = f"{aptitud_mejor:.4f}"
                else:
                     aptitud_str = "N/A"
                     print(f"Advertencia: Aptitud no numérica encontrada en el mejor de la generación {generacion + 1}")

                print(f"Generación {generacion + 1}/{self.num_generaciones}: "
                      f"Mejor Aptitud={aptitud_str}, "
                      f"Cultivo='{mejor_de_generacion.get('cultivo_nombre', 'Desconocido')}'")

                # b) Mostrar los Mejores Competidores (Cultivos Únicos)
                print(f"  --- Top {num_competidores_a_mostrar} Competidores (Cultivos Únicos en Población) ---")
                cultivos_mostrados = set()
                competidores_impresos = 0
                for individuo in poblacion_ordenada_evaluada:
                    cultivo = individuo.get('cultivo_nombre', 'Desconocido')
                    if cultivo not in cultivos_mostrados:
                        aptitud_competidor = individuo.get('aptitud', 0.0)
                        if isinstance(aptitud_competidor, (int, float)):
                             aptitud_comp_str = f"{aptitud_competidor:.4f}"
                        else:
                             aptitud_comp_str = "N/A"

                        # Añadir info de rotación para contexto
                        penalizado_str = ""
                        if not self.verificar_rotacion(cultivo):
                            penalizado_str = " (Rot Penal.)"

                        print(f"    - {cultivo:<15}: {aptitud_comp_str}{penalizado_str}")

                        cultivos_mostrados.add(cultivo)
                        competidores_impresos += 1
                        if competidores_impresos >= num_competidores_a_mostrar:
                            break
                # Si se mostraron menos competidores que los solicitados y hubo más individuos
                if competidores_impresos < num_competidores_a_mostrar and len(poblacion_ordenada_evaluada) > competidores_impresos:
                     print("    (No hay más tipos de cultivos únicos en la población actual)")
                elif competidores_impresos == 0:
                     print("    (Población vacía o sin individuos válidos)")


                # 2. Generar la nueva población para la siguiente iteración
                poblacion_actual = self.generar_nueva_poblacion(poblacion_actual)
            # --- Fin del Ciclo Evolutivo ---


            # Evaluar la población final una última vez por si acaso
            poblacion_final = self.evaluar_poblacion(poblacion_actual)

            if not poblacion_final:
                 print("Optimización finalizada pero la población final está vacía.")
                 return None

            # Encontrar y retornar el mejor individuo de la población final
            mejor_individuo_final = max(poblacion_final, key=lambda ind: ind.get("aptitud", 0.0))

            print("-" * 30)
            print("Optimización completada.")
            if mejor_individuo_final: # Asegurarse de que existe
                print("Detalles del Mejor Individuo Encontrado:")
                print(f"  Cultivo: {mejor_individuo_final.get('cultivo_nombre', 'Desconocido')}")
                print(f"  Aptitud Final: {mejor_individuo_final.get('aptitud', 0.0):.4f}")

                print("\n  Valores Originales (del CSV):")
                # Usamos .get() para evitar errores si alguna clave faltara (aunque no debería)
                # y formateamos los números para mejor lectura (ej. 2 decimales)
                original_agua = mejor_individuo_final.get('original_agua', 'N/A')
                original_fert = mejor_individuo_final.get('original_fertilizante', 'N/A')
                original_pest = mejor_individuo_final.get('original_pesticida', 'N/A')
                original_retorno = mejor_individuo_final.get('original_retorno_economico', 'N/A')
                print(f"    Agua Original:       {original_agua}")
                print(f"    Fertilizante Original: {original_fert}")
                print(f"    Pesticida Original:    {original_pest}")
                print(f"    Retorno Original:     {original_retorno}")
                print("\n  Cantidades Finales (Evolucionadas):")
                cantidad_agua = mejor_individuo_final.get('cantidad_agua', 'N/A')
                cantidad_fert = mejor_individuo_final.get('cantidad_fertilizante', 'N/A')
                cantidad_pest = mejor_individuo_final.get('cantidad_pesticida', 'N/A')
                print(f"  Agua Final:       {cantidad_agua}")
                print(f"  Fertilizante Final: {cantidad_fert}")
                print(f"  Pesticida Final:    {cantidad_pest}")


            # (Grafica opcional, etc...)

            return mejor_individuo_final

        # (Manejo de excepciones como estaba...)
        except ValueError as e:
             print(f"Error durante la optimización: {e}")
             messagebox.showerror("Error de Ejecución", f"Ocurrió un error de valor: {e}")
             # import traceback # Descomentar para más detalles
             # traceback.print_exc()
             return None
        except Exception as e:
            print(f"Error inesperado durante la optimización: {e}")
            messagebox.showerror("Error Inesperado", f"Ocurrió un error inesperado: {e}")
            import traceback
            traceback.print_exc() # Imprime detalles del error
            return None