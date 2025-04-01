import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import copy # Necesario para deepcopy en cruce/mutación de planes

class AG:

    def __init__(self, num_poblacion, num_generaciones, prob_mutacion, prob_crossover,
                 num_elite, num_parcelas, # NUEVO: Número de parcelas/slots en el plan
                 total_agua_disponible, total_fertilizante_disponible, total_pesticida_disponible, # RENOMBRADO: Límites GLOBALES
                 prob_mutar_cultivo_en_parcela=0.1): # Probabilidad de cambiar el cultivo en una parcela
        self.num_poblacion = num_poblacion
        self.num_generaciones = num_generaciones
        self.prob_mutacion_cantidad = prob_mutacion # Probabilidad de mutar una cantidad (agua, fert, pest) en una parcela
        self.prob_mutar_cultivo = prob_mutar_cultivo_en_parcela # Probabilidad de mutar el tipo de cultivo en una parcela
        self.prob_crossover = prob_crossover
        self.num_elite = num_elite
        self.num_parcelas = num_parcelas # Número de "slots" en el plan de cultivo

        # Límites TOTALES disponibles para TODO el plan
        self.total_agua_disponible = total_agua_disponible
        self.total_fertilizante_disponible = total_fertilizante_disponible
        self.total_pesticida_disponible = total_pesticida_disponible

        self.datos_csv = None
        # Máximos del CSV (para normalización individual de parcelas)
        self.max_agua_csv = 1
        self.max_fert_csv = 1
        self.max_pest_csv = 1
        self.max_costo_csv = 1
        self.max_prod_csv = 1
        # Máximos teóricos del plan (para normalización del plan total) - Estimación simple
        self.max_retorno_plan_estimado = 1
        self.max_costo_plan_estimado = 1
        self.max_prod_plan_estimado = 1


        self.mejor_aptitud_historial = []
        self.mejor_plan_global = None # Para almacenar el mejor plan encontrado en todas las generaciones

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
            self.datos_csv = pd.read_csv(file_path)

            required_cols = ['Cultivo', 'Agua', 'Fertilizante', 'Pesticida', 'Costo',
                             'Produccion', 'EsCultivable', 'CultivoAnterior',
                             'AdaptabilidadClima', 'RetornoEconomico']
            missing_cols = [col for col in required_cols if col not in self.datos_csv.columns]
            if missing_cols:
                 messagebox.showerror("Error", f"Columnas requeridas faltantes en el CSV: {', '.join(missing_cols)}")
                 self.datos_csv = None
                 return None

            numeric_cols = ['Agua', 'Fertilizante', 'Pesticida', 'Costo', 'Produccion',
                            'EsCultivable', 'AdaptabilidadClima', 'RetornoEconomico']
            for col in numeric_cols:
                self.datos_csv[col] = pd.to_numeric(self.datos_csv[col], errors='coerce')

            if self.datos_csv[numeric_cols].isnull().any().any():
                 print("Advertencia: Se encontraron valores no numéricos en columnas que deberían serlo. Revise el CSV.")

            self.datos_csv = self.datos_csv.dropna(subset=['EsCultivable'])
            self.datos_csv = self.datos_csv[self.datos_csv["EsCultivable"] == 1].copy()

            if self.datos_csv.empty:
                messagebox.showerror("Error", "El archivo CSV no contiene filas válidas con 'EsCultivable' == 1 después de la limpieza.")
                self.datos_csv = None
                return None

            print(f"Datos CSV cargados y filtrados: {len(self.datos_csv)} cultivos cultivables.")
            print("-" * 30)

            # Calcular máximos del CSV para normalización
            self.max_agua_csv = max(self.datos_csv['Agua'].max(skipna=True), 1)
            self.max_fert_csv = max(self.datos_csv['Fertilizante'].max(skipna=True), 1)
            self.max_pest_csv = max(self.datos_csv['Pesticida'].max(skipna=True), 1)
            self.max_costo_csv = max(self.datos_csv['Costo'].max(skipna=True), 1)
            self.max_prod_csv = max(self.datos_csv['Produccion'].max(skipna=True), 1)
            max_retorno_csv = max(self.datos_csv['RetornoEconomico'].max(skipna=True), 1) # Asumiendo que Retorno es > 0

            # Estimaciones simples para máximos del plan completo (para normalizar aptitud total)
            # Se pueden refinar estas estimaciones si se tiene más información
            self.max_retorno_plan_estimado = max_retorno_csv * self.num_parcelas
            self.max_costo_plan_estimado = self.max_costo_csv * self.num_parcelas
            self.max_prod_plan_estimado = self.max_prod_csv * self.num_parcelas


            return self.datos_csv
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado al cargar el archivo: {e}")
            import traceback
            traceback.print_exc()
            self.datos_csv = None
            return None

    def _generar_parcela(self):
        """Genera el diccionario para UNA parcela seleccionando un cultivo aleatorio."""
        if self.datos_csv is None or self.datos_csv.empty:
            raise ValueError("No se han cargado datos válidos del CSV.")

        fila = self.datos_csv.sample().iloc[0]
        cultivo = fila["Cultivo"]

        # Valores ORIGINALES del CSV para este cultivo
        original_agua = fila["Agua"]
        original_fertilizante = fila["Fertilizante"]
        original_pesticida = fila["Pesticida"]
        original_costo = fila["Costo"]
        original_produccion = fila["Produccion"]
        original_retorno = fila["RetornoEconomico"]
        original_adaptabilidad = fila["AdaptabilidadClima"] # Guardar también

        # Crear el diccionario de la parcela
        parcela = {
            "cultivo_nombre": cultivo,
            "original_agua": original_agua,
            "original_fertilizante": original_fertilizante,
            "original_pesticida": original_pesticida,
            "original_costo": original_costo,
            "original_produccion": original_produccion,
            "original_retorno_economico": original_retorno,
            "original_adaptabilidad_clima": original_adaptabilidad,

            # Valores Mutables (inicializados con los originales)
            # Estos representan la asignación de recursos A ESTA PARCELA
            "cantidad_agua": original_agua,
            "cantidad_fertilizante": original_fertilizante,
            "cantidad_pesticida": original_pesticida,
        }
        return parcela

    def generar_individuo(self):
        """Genera un individuo (PLAN), que es una lista de parcelas."""
        if self.datos_csv is None or self.datos_csv.empty:
             raise ValueError("No se pueden generar individuos sin datos CSV cargados.")
        if self.num_parcelas <= 0:
            raise ValueError("El número de parcelas debe ser positivo.")

        plan = [self._generar_parcela() for _ in range(self.num_parcelas)]
        # La aptitud se calculará para el plan completo más tarde
        return {"plan": plan, "aptitud": 0.0} # Envolver la lista en un dict con su aptitud

    def incializar_poblacion(self):
        """Crea la población inicial de planes."""
        if self.datos_csv is None or self.datos_csv.empty:
             raise ValueError("No se pueden inicializar la población sin datos CSV cargados.")
        poblacion = [self.generar_individuo() for _ in range(self.num_poblacion)]
        return poblacion

    def verificar_rotacion_secuencial(self, nombre_cultivo_actual, nombre_cultivo_anterior):
        """
        Verifica si 'nombre_cultivo_actual' puede seguir a 'nombre_cultivo_anterior'
        basado en la columna 'CultivoAnterior' del CSV.
        Asume que 'Ninguno', NaN o vacío significa que no hay restricción desde el anterior.
        """
        if self.datos_csv is None or nombre_cultivo_anterior is None:
             return True # No se puede verificar o es la primera parcela

        # Busca la fila correspondiente al cultivo ACTUAL en los datos cargados
        filas_cultivo_actual = self.datos_csv[self.datos_csv["Cultivo"] == nombre_cultivo_actual]

        if filas_cultivo_actual.empty:
             print(f"Advertencia: Cultivo '{nombre_cultivo_actual}' no encontrado en CSV para verificación de rotación.")
             return False # Considerarlo no válido si no se encuentra

        fila_actual = filas_cultivo_actual.iloc[0]
        cultivo_anterior_requerido = fila_actual["CultivoAnterior"]

        # Interpretar la regla de CultivoAnterior
        if pd.isna(cultivo_anterior_requerido) or cultivo_anterior_requerido.strip() == "" or cultivo_anterior_requerido == "Ninguno":
            return True # No hay restricción sobre qué debe venir antes
        elif cultivo_anterior_requerido == "Diferente":
             return nombre_cultivo_actual != nombre_cultivo_anterior # Debe ser diferente al anterior inmediato
        else:
            # Se requiere un cultivo específico como anterior
            return nombre_cultivo_anterior == cultivo_anterior_requerido


    def calcular_aptitud(self, individuo):
        """
        Calcula la aptitud de un individuo (PLAN).
        Considera el rendimiento/costo total, el uso de recursos globales y la rotación.
        """
        plan = individuo["plan"] # Extraer la lista de parcelas
        if not plan: # Si el plan está vacío por alguna razón
            return 0.0

        # --- Sumar recursos y calcular métricas totales del plan ---
        agua_total_usada = sum(p.get('cantidad_agua', 0) for p in plan)
        fert_total_usado = sum(p.get('cantidad_fertilizante', 0) for p in plan)
        pest_total_usado = sum(p.get('cantidad_pesticida', 0) for p in plan)

        produccion_total_ajustada_plan = 0.0
        costo_total_base_plan = 0.0 # Usando costos base por simplicidad inicial
        retorno_total_base_plan = 0.0 # Usando retornos base por simplicidad inicial
        adaptabilidad_promedio_plan = 0.0
        num_parcelas_validas = 0

        # --- 1. Penalización por Exceso de Recursos GLOBALES (Hard Constraint) ---
        if (agua_total_usada > self.total_agua_disponible or
            fert_total_usado > self.total_fertilizante_disponible or
            pest_total_usado > self.total_pesticida_disponible):
            # print(f"Plan descartado por exceso de recursos: A:{agua_total_usada:.1f}/{self.total_agua_disponible} F:{fert_total_usado:.1f}/{self.total_fertilizante_disponible} P:{pest_total_usado:.1f}/{self.total_pesticida_disponible}")
            return 0.0 # Aptitud cero si se exceden los límites globales

        # --- 2. Calcular contribución de cada parcela y métricas agregadas ---
        factor_ajuste_prod_parcela = 1.5 # Cuán sensible es la producción a la desviación (por parcela)

        for parcela in plan:
            # Obtener valores BASE de la parcela (guardados o buscar en CSV si no)
            agua_base = parcela.get("original_agua", 0)
            fert_base = parcela.get("original_fertilizante", 0)
            pest_base = parcela.get("original_pesticida", 0)
            prod_base = parcela.get("original_produccion", 0)
            costo_base = parcela.get("original_costo", 0)
            retorno_base = parcela.get("original_retorno_economico", 0)
            adapt_base = parcela.get("original_adaptabilidad_clima", 0)

            # Obtener cantidades asignadas a ESTA parcela
            agua_ind = parcela.get("cantidad_agua", 0)
            fert_ind = parcela.get("cantidad_fertilizante", 0)
            pest_ind = parcela.get("cantidad_pesticida", 0)

            # Calcular Producción Ajustada para ESTA parcela (similar a antes)
            dev_agua_rel = abs(agua_ind - agua_base) / agua_base if agua_base > 0 else (0 if agua_ind == 0 else float('inf'))
            dev_fert_rel = abs(fert_ind - fert_base) / fert_base if fert_base > 0 else (0 if fert_ind == 0 else float('inf'))
            dev_pest_rel = abs(pest_ind - pest_base) / pest_base if pest_base > 0 else (0 if pest_ind == 0 else float('inf'))

            # Evitar división por cero o valores infinitos
            valid_devs = [d for d in [dev_agua_rel, dev_fert_rel, dev_pest_rel] if d != float('inf')]
            if not valid_devs: # Si todos son inf o los bases son 0 y las cantidades no
                 produccion_ajustada_parcela = 0.0
            else:
                avg_rel_dev = sum(valid_devs) / len(valid_devs) if valid_devs else 0
                reduccion_prod = factor_ajuste_prod_parcela * (avg_rel_dev ** 2)
                produccion_ajustada_parcela = prod_base * max(0.0, 1.0 - reduccion_prod)

            produccion_total_ajustada_plan += produccion_ajustada_parcela
            costo_total_base_plan += costo_base
            retorno_total_base_plan += retorno_base # Podríamos ajustar el retorno basado en prod_ajustada? Más complejo. Empezar simple.
            adaptabilidad_promedio_plan += adapt_base
            num_parcelas_validas += 1

        if num_parcelas_validas > 0:
            adaptabilidad_promedio_plan /= num_parcelas_validas
        else:
             adaptabilidad_promedio_plan = 0 # Evitar división por cero

        # --- 3. Penalización por Rotación (Secuencial) ---
        factor_penalizacion_rotacion = 0.8 # Multiplicador si falla la rotación (reduce la aptitud)
        penalizacion_rotacion_acumulada = 1.0
        for i in range(1, len(plan)):
            cultivo_actual = plan[i].get('cultivo_nombre')
            cultivo_anterior = plan[i-1].get('cultivo_nombre')
            if not self.verificar_rotacion_secuencial(cultivo_actual, cultivo_anterior):
                penalizacion_rotacion_acumulada *= factor_penalizacion_rotacion
                # print(f"Penalización rotación: {cultivo_anterior} -> {cultivo_actual}")


        # --- 4. Combinar métricas en Aptitud Final ---
        # Normalizar métricas totales del plan (usando estimaciones max del plan)
        prod_norm_plan = produccion_total_ajustada_plan / self.max_prod_plan_estimado if self.max_prod_plan_estimado else 0
        costo_norm_plan = costo_total_base_plan / self.max_costo_plan_estimado if self.max_costo_plan_estimado else 0
        retorno_norm_plan = retorno_total_base_plan / self.max_retorno_plan_estimado if self.max_retorno_plan_estimado else 0
        # Adaptabilidad ya está entre 0 y 1 (asumimos)

        # Definir Pesos para la aptitud del plan
        peso_produccion = 0.30
        peso_retorno = 0.40
        peso_costo = 0.15 # Minimizar costo -> resta
        peso_adaptabilidad = 0.10
        # Penalización por uso de recursos (queremos favorecer planes más eficientes incluso si no exceden el límite)
        peso_eficiencia_recursos = 0.05
        agua_usada_norm = agua_total_usada / self.total_agua_disponible if self.total_agua_disponible else 0
        fert_usado_norm = fert_total_usado / self.total_fertilizante_disponible if self.total_fertilizante_disponible else 0
        pest_usado_norm = pest_total_usado / self.total_pesticida_disponible if self.total_pesticida_disponible else 0
        penalizacion_uso_recursos = (agua_usada_norm + fert_usado_norm + pest_usado_norm) / 3.0


        aptitud_bruta = (
            (peso_produccion * prod_norm_plan) +
            (peso_retorno * retorno_norm_plan) +
            (peso_adaptabilidad * adaptabilidad_promedio_plan) -
            (peso_costo * costo_norm_plan) -
            (peso_eficiencia_recursos * penalizacion_uso_recursos)
        )

        # Aplicar penalización por rotación
        aptitud_con_rotacion = aptitud_bruta * penalizacion_rotacion_acumulada

        # Asegurar que la aptitud sea no negativa
        aptitud_final = max(0.0, aptitud_con_rotacion)

        # print(f"Plan: {[p['cultivo_nombre'] for p in plan]} | ProdNorm:{prod_norm_plan:.2f} RetNorm:{retorno_norm_plan:.2f} CostNorm:{costo_norm_plan:.2f} Adap:{adaptabilidad_promedio_plan:.2f} UsoRes:{penalizacion_uso_recursos:.2f} PenRot:{penalizacion_rotacion_acumulada:.2f} -> Aptitud: {aptitud_final:.4f}")

        return aptitud_final

    def evaluar_poblacion(self, poblacion):
        """Calcula y asigna la aptitud a cada individuo (plan) de la población."""
        for individuo in poblacion:
            # Calcular la aptitud usando la función corregida
            individuo["aptitud"] = self.calcular_aptitud(individuo)
            # Asegurar que aptitud sea float
            if not isinstance(individuo["aptitud"], float):
                 individuo["aptitud"] = float(individuo["aptitud"])
        return poblacion

    def seleccion_por_torneo(self, poblacion, tamaño_torneo=3):
        """Selecciona dos padres (planes) usando el método de torneo."""
        n_poblacion = len(poblacion)
        if n_poblacion == 0:
            raise ValueError("La población está vacía, no se puede seleccionar.")
        tamaño_real_torneo = min(tamaño_torneo, n_poblacion)
        indices1 = np.random.choice(n_poblacion, size=tamaño_real_torneo, replace=False)
        indices2 = np.random.choice(n_poblacion, size=tamaño_real_torneo, replace=False)
        torneo1 = [poblacion[i] for i in indices1]
        torneo2 = [poblacion[i] for i in indices2]
        padre1 = max(torneo1, key=lambda ind: ind.get("aptitud", 0.0))
        padre2 = max(torneo2, key=lambda ind: ind.get("aptitud", 0.0))
        return padre1, padre2

    def cruzar(self, padre1, padre2):
        """Realiza el cruce entre dos padres (planes) usando cruce de un punto en la lista."""
        # Usar deepcopy porque los individuos contienen listas de diccionarios
        hijo1_ind = copy.deepcopy(padre1)
        hijo2_ind = copy.deepcopy(padre2)
        plan1 = hijo1_ind["plan"]
        plan2 = hijo2_ind["plan"]

        if np.random.rand() < self.prob_crossover and self.num_parcelas > 1:
            # Cruce de un punto en la lista de parcelas
            punto_cruce = np.random.randint(1, self.num_parcelas) # Entre 1 y num_parcelas-1
            # print(f"Cruce en punto: {punto_cruce}")
            nuevo_plan1 = plan1[:punto_cruce] + plan2[punto_cruce:]
            nuevo_plan2 = plan2[:punto_cruce] + plan1[punto_cruce:]
            hijo1_ind["plan"] = nuevo_plan1
            hijo2_ind["plan"] = nuevo_plan2

            # La aptitud de los hijos deberá ser recalculada
            hijo1_ind['aptitud'] = 0.0
            hijo2_ind['aptitud'] = 0.0

        return hijo1_ind, hijo2_ind

    def mutar(self, individuo):
        """Aplica mutación a un individuo (plan). Puede mutar el cultivo o las cantidades en parcelas."""
        individuo_mutado = copy.deepcopy(individuo) # Necesario por la estructura anidada
        plan_mutado = individuo_mutado["plan"]
        reset_aptitud = False

        for i in range(len(plan_mutado)):
            parcela_actual = plan_mutado[i]

            # --- Mutación de Cultivo en Parcela ---
            if np.random.rand() < self.prob_mutar_cultivo:
                if self.datos_csv is not None and not self.datos_csv.empty:
                    cultivos_disponibles = self.datos_csv['Cultivo'].unique()
                    cultivo_actual_nombre = parcela_actual['cultivo_nombre']
                    posibles_nuevos = [c for c in cultivos_disponibles if c != cultivo_actual_nombre]

                    if posibles_nuevos:
                        nuevo_cultivo_nombre = np.random.choice(posibles_nuevos)
                        nueva_fila = self.datos_csv[self.datos_csv['Cultivo'] == nuevo_cultivo_nombre].iloc[0]

                        print(f"*** MUTACIÓN CULTIVO en Parcela {i}: {cultivo_actual_nombre} -> {nuevo_cultivo_nombre} ***")

                        # Actualizar TODOS los valores relevantes en ESTA parcela
                        parcela_actual['cultivo_nombre'] = nuevo_cultivo_nombre
                        parcela_actual['original_agua'] = nueva_fila["Agua"]
                        parcela_actual['original_fertilizante'] = nueva_fila["Fertilizante"]
                        parcela_actual['original_pesticida'] = nueva_fila["Pesticida"]
                        parcela_actual['original_costo'] = nueva_fila["Costo"]
                        parcela_actual['original_produccion'] = nueva_fila["Produccion"]
                        parcela_actual['original_retorno_economico'] = nueva_fila["RetornoEconomico"]
                        parcela_actual['original_adaptabilidad_clima'] = nueva_fila["AdaptabilidadClima"]
                        # Resetear también los mutables a los nuevos originales
                        parcela_actual['cantidad_agua'] = nueva_fila["Agua"]
                        parcela_actual['cantidad_fertilizante'] = nueva_fila["Fertilizante"]
                        parcela_actual['cantidad_pesticida'] = nueva_fila["Pesticida"]
                        reset_aptitud = True


            # --- Mutación de Cantidades en Parcela ---
            # Definir límites seguros para la mutación de cantidad (ej. 1.5x el máx del CSV para ese recurso?)
            # O usar los límites globales divididos entre parcelas como guía muy laxa?
            # Usemos los max del CSV como guía para la escala del cambio.
            # Podríamos también definir max_mutacion_agua_parcela etc. en __init__
            max_agua_mut_parcela = self.max_agua_csv * 1.5 # Ejemplo: permitir hasta 150% del max visto
            max_fert_mut_parcela = self.max_fert_csv * 1.5
            max_pest_mut_parcela = self.max_pest_csv * 1.5

            if np.random.rand() < self.prob_mutacion_cantidad:
                 # Mutar cantidad de agua para esta parcela
                 # Usar una distribución normal centrada en el valor actual? O uniforme?
                 # Uniforme es más simple por ahora.
                 cantidad_anterior = parcela_actual["cantidad_agua"]
                 parcela_actual["cantidad_agua"] = np.random.uniform(0, max_agua_mut_parcela)
                 # print(f"Mutacion agua parcela {i}: {cantidad_anterior:.1f} -> {parcela_actual['cantidad_agua']:.1f}")
                 reset_aptitud = True

            if np.random.rand() < self.prob_mutacion_cantidad:
                 cantidad_anterior = parcela_actual["cantidad_fertilizante"]
                 parcela_actual["cantidad_fertilizante"] = np.random.uniform(0, max_fert_mut_parcela)
                 # print(f"Mutacion fert parcela {i}: {cantidad_anterior:.1f} -> {parcela_actual['cantidad_fertilizante']:.1f}")
                 reset_aptitud = True

            if np.random.rand() < self.prob_mutacion_cantidad:
                 cantidad_anterior = parcela_actual["cantidad_pesticida"]
                 parcela_actual["cantidad_pesticida"] = np.random.uniform(0, max_pest_mut_parcela)
                 # print(f"Mutacion pest parcela {i}: {cantidad_anterior:.1f} -> {parcela_actual['cantidad_pesticida']:.1f}")
                 reset_aptitud = True

        if reset_aptitud:
            individuo_mutado['aptitud'] = 0.0 # Resetear aptitud si hubo alguna mutación

        return individuo_mutado

    def generar_nueva_poblacion(self, poblacion_actual):
        """Genera la siguiente población de planes."""
        nueva_poblacion = []
        n_poblacion = len(poblacion_actual)

        if n_poblacion == 0:
            return [] # Retornar lista vacía si la población actual está vacía

        # Ordenar la población actual por aptitud descendente para elitismo
        poblacion_ordenada = sorted(poblacion_actual, key=lambda ind: ind.get("aptitud", 0.0), reverse=True)

        # 1. Elitismo
        num_elite_real = min(self.num_elite, n_poblacion)
        # Asegurar que los élite se copian profundamente para evitar modificaciones accidentales
        for i in range(num_elite_real):
             nueva_poblacion.append(copy.deepcopy(poblacion_ordenada[i]))
             # print(f"Elite {i+1}: Aptitud {poblacion_ordenada[i]['aptitud']:.4f}")


        # 2. Generar el resto usando Selección, Cruce y Mutación
        while len(nueva_poblacion) < self.num_poblacion:
            padre1, padre2 = self.seleccion_por_torneo(poblacion_ordenada) # Usar ordenada puede dar ligera ventaja
            hijo1, hijo2 = self.cruzar(padre1, padre2)
            hijo1 = self.mutar(hijo1)
            hijo2 = self.mutar(hijo2) # Mutar ambos hijos

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < self.num_poblacion:
                nueva_poblacion.append(hijo2)

        return nueva_poblacion


    def optimizar(self):
        """Ejecuta el algoritmo genético completo para encontrar el mejor plan de cultivo."""
        if self.datos_csv is None or self.datos_csv.empty:
             messagebox.showerror("Error", "No se han cargado datos CSV válidos. Ejecuta load_csv() primero.")
             return None

        print("Iniciando optimización del PLAN DE CULTIVO...")
        print(f"Parámetros: Población={self.num_poblacion}, Generaciones={self.num_generaciones}, Parcelas={self.num_parcelas}")
        print(f"Probs: Cruce={self.prob_crossover}, Mut.Cantidad={self.prob_mutacion_cantidad}, Mut.Cultivo={self.prob_mutar_cultivo}, Elite={self.num_elite}")
        print(f"Recursos Totales Disponibles: Agua={self.total_agua_disponible}, Fert={self.total_fertilizante_disponible}, Pest={self.total_pesticida_disponible}")
        print("-" * 40)

        try:
            poblacion_actual = self.incializar_poblacion()
            self.mejor_aptitud_historial = []
            self.mejor_plan_global = None # Reiniciar el mejor global
            mejor_aptitud_global = -1.0

            for generacion in range(self.num_generaciones):
                # 1. Evaluar la población actual
                poblacion_actual = self.evaluar_poblacion(poblacion_actual)

                if not poblacion_actual:
                    print(f"Generación {generacion + 1}/{self.num_generaciones}: ¡Población vacía!")
                    break

                # Ordenar para encontrar el mejor de esta generación
                poblacion_ordenada_evaluada = sorted(poblacion_actual, key=lambda ind: ind.get("aptitud", 0.0), reverse=True)
                mejor_plan_generacion = poblacion_ordenada_evaluada[0]
                aptitud_mejor_generacion = mejor_plan_generacion.get("aptitud", 0.0)
                self.mejor_aptitud_historial.append(aptitud_mejor_generacion if isinstance(aptitud_mejor_generacion, float) else 0.0)

                # Actualizar el mejor plan global encontrado hasta ahora
                if aptitud_mejor_generacion > mejor_aptitud_global:
                    mejor_aptitud_global = aptitud_mejor_generacion
                    self.mejor_plan_global = copy.deepcopy(mejor_plan_generacion) # Guardar copia profunda
                    print(f"*** Nuevo Mejor Plan Global encontrado en Gen {generacion + 1} - Aptitud: {mejor_aptitud_global:.5f} ***")


                # Mostrar info de la generación
                aptitud_str = f"{aptitud_mejor_generacion:.5f}" if isinstance(aptitud_mejor_generacion, float) else "N/A"
                print(f"Generación {generacion + 1}/{self.num_generaciones}: Mejor Aptitud={aptitud_str}")
                # Opcional: Mostrar los cultivos del mejor plan de la generación
                cultivos_mejor_gen = [p.get('cultivo_nombre', '?') for p in mejor_plan_generacion.get('plan', [])]
                print(f"  Mejor Plan Gen: {' -> '.join(cultivos_mejor_gen)}")


                # 2. Generar la nueva población
                poblacion_actual = self.generar_nueva_poblacion(poblacion_actual)

            # --- Fin del Ciclo Evolutivo ---

            print("-" * 40)
            print("Optimización completada.")

            if self.mejor_plan_global is None:
                print("No se encontró ningún plan válido durante la optimización.")
                # Intentar encontrar el mejor de la última población si existe
                if poblacion_actual:
                     poblacion_final_evaluada = self.evaluar_poblacion(poblacion_actual)
                     poblacion_final_ordenada = sorted(poblacion_final_evaluada, key=lambda ind: ind.get("aptitud", 0.0), reverse=True)
                     if poblacion_final_ordenada and poblacion_final_ordenada[0].get("aptitud", 0.0) > 0:
                           self.mejor_plan_global = copy.deepcopy(poblacion_final_ordenada[0])
                           mejor_aptitud_global = self.mejor_plan_global.get("aptitud", 0.0)
                           print("Usando el mejor plan encontrado en la última generación.")
                     else:
                           print("La población final tampoco contenía planes válidos.")
                           return None
                else:
                     return None


            print("=== Mejor Plan de Cultivo Encontrado ===")
            print(f"Aptitud Final del Plan: {mejor_aptitud_global:.5f}")

            agua_total_final = 0
            fert_total_final = 0
            pest_total_final = 0
            costo_total_final = 0
            retorno_total_final = 0 # Estimado base

            plan_final = self.mejor_plan_global.get("plan", [])
            print("\nDetalles por Parcela:")
            for i, parcela in enumerate(plan_final):
                print(f"  --- Parcela {i+1} ---")
                print(f"    Cultivo:          {parcela.get('cultivo_nombre', 'N/A')}")
                print(f"    Agua Asignada:    {parcela.get('cantidad_agua', 0):.2f} (Original: {parcela.get('original_agua', 0):.2f})")
                print(f"    Fertiliz. Asignado: {parcela.get('cantidad_fertilizante', 0):.2f} (Original: {parcela.get('original_fertilizante', 0):.2f})")
                print(f"    Pesticida Asignado: {parcela.get('cantidad_pesticida', 0):.2f} (Original: {parcela.get('original_pesticida', 0):.2f})")
                print(f"    Costo Base:       {parcela.get('original_costo', 0):.2f}")
                print(f"    Retorno Base:     {parcela.get('original_retorno_economico', 0):.2f}")
                print(f"    Adaptabilidad:    {parcela.get('original_adaptabilidad_clima', 0):.2f}")

                agua_total_final += parcela.get('cantidad_agua', 0)
                fert_total_final += parcela.get('cantidad_fertilizante', 0)
                pest_total_final += parcela.get('cantidad_pesticida', 0)
                costo_total_final += parcela.get('original_costo', 0) # Costo base sumado
                retorno_total_final += parcela.get('original_retorno_economico', 0) # Retorno base sumado

            print("\nResumen del Plan Completo:")
            print(f"  Agua Total Usada:        {agua_total_final:.2f} / {self.total_agua_disponible:.2f}")
            print(f"  Fertilizante Total Usado:{fert_total_final:.2f} / {self.total_fertilizante_disponible:.2f}")
            print(f"  Pesticida Total Usado:   {pest_total_final:.2f} / {self.total_pesticida_disponible:.2f}")
            print(f"  Costo Total Estimado:    {costo_total_final:.2f}")
            print(f"  Retorno Total Estimado:  {retorno_total_final:.2f}")





            return self.mejor_plan_global

        except ValueError as e:
             print(f"Error durante la optimización: {e}")
             messagebox.showerror("Error de Ejecución", f"Ocurrió un error de valor: {e}")
             return None
        except Exception as e:
            print(f"Error inesperado durante la optimización: {e}")
            messagebox.showerror("Error Inesperado", f"Ocurrió un error inesperado: {e}")
            import traceback
            traceback.print_exc()
            return None