import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

class AG:

    def __init__(self, num_poblacion, num_generaciones, prob_mutacion, prob_crossover, num_elite, cantidad_agua, cantidad_fertilizante, cantidad_pesticida):
        self.num_poblacion = num_poblacion
        self.num_generaciones = num_generaciones    
        self.prob_mutacion = prob_mutacion
        self.prob_crossover = prob_crossover
        self.num_elite = num_elite
        self.cantidad_agua = cantidad_agua
        self.cantidad_fertilizante = cantidad_fertilizante
        self.cantidad_pesticida = cantidad_pesticida
        self.datos_csv = None
    
    def load_csv(self):
        
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return None

        try:
            self.datos_csv = pd.read_csv(file_path)
            self.datos_csv = self.datos_csv[self.datos_csv["EsCultivable"] == 1]
            print(self.datos_csv)
            return self.datos_csv
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error al cargar el archivo: {e}")
            return None
        

    def generar_individuo(self):
        if self.datos_csv is None or self.datos_csv.empty:
            raise ValueError("No se han cargado datos del CSV.")
       

        fila = self.datos_csv.sample().iloc[0]

         
        cantidad_agua = fila["Agua"]
        cantidad_fertilizante = fila["Fertilizante"]
        cantidad_pesticida = fila["Pesticida"]
        costo_total = fila["Costo"]
        produccion_total = fila["Produccion"]
        cultivo = fila["Cultivo"]
     
        individuo = {
        "cantidad_agua": cantidad_agua,
        "cantidad_fertilizante": cantidad_fertilizante,
        "cantidad_pesticida": cantidad_pesticida,
        "costo_total": costo_total,
        "produccion_total": produccion_total,
        "cultivos": [cultivo]
    }
        return individuo
    

    def incializar_poblacion(self):
        poblacion = [self.generar_individuo() for _ in range(self.num_poblacion)]
        return poblacion

    def verificar_rotacion(self, cultivos):
       
        cultivo_actual = cultivos[0]
        fila = self.datos_csv[self.datos_csv["Cultivo"] == cultivo_actual].iloc[0]
        cultivo_anterior = fila["CultivoAnterior"]

        return cultivo_actual != cultivo_anterior

    def calcular_aptitud(self, individuo):
        if self.datos_csv is None or self.datos_csv.empty:
            raise ValueError("No se han cargado datos del CSV.")
        
        
        fila = self.datos_csv[self.datos_csv["Cultivo"] == individuo["cultivos"][0]].iloc[0]

        adaptabilidad_clima = fila["AdaptabilidadClima"]
        retorno_economico = fila["RetornoEconomico"]

        peso_produccion = 0.4
        peso_agua = 0.2
        peso_fertilizante = 0.2
        peso_pesticida = 0.1
        peso_costo = 0.1
        peso_adaptabilidad = 0.1
        peso_retorno = 0.1

        
        produccion_normalizada = individuo["produccion_total"]
        agua_normalizada = 1 - (individuo["cantidad_agua"])
        fertilizante_normalizado = 1 - (individuo["cantidad_fertilizante"])
        pesticida_normalizado = 1 - (individuo["cantidad_pesticida"])
        costo_normalizado = 1 - (individuo["costo_total"]) 

       
        aptitud = (
            peso_produccion * produccion_normalizada +
            peso_agua * agua_normalizada +
            peso_fertilizante * fertilizante_normalizado +
            peso_pesticida * pesticida_normalizado +
            peso_costo * costo_normalizado +
            peso_adaptabilidad * adaptabilidad_clima +
            peso_retorno * retorno_economico
    )

       
        if not self.verificar_rotacion(individuo["cultivos"]):
            aptitud *= 0.5 

        return aptitud
    
    def evaluar_poblacion(self, poblacion):
        for individuo in poblacion:
            individuo["aptitud"] = self.calcular_aptitud(individuo)
        return poblacion
    

    def seleccion_por_torneo(self, poblacion, tamaño_torneo= 3):

        torneo1 = np.random.choice(poblacion, size=tamaño_torneo, replace=False)
        torneo2 = np.random.choice(poblacion, size=tamaño_torneo, replace=False)

        padre1 = max(torneo1, key=lambda ind: ind["aptitud"])
        padre2 = max(torneo2, key=lambda ind: ind["aptitud"])


        return padre1, padre2
    

    def cruzar(self, padre1, padre2):

        if np.random.rand() > self.prob_crossover:
            return padre1.copy(), padre2.copy()

        hijo1 = padre1.copy()
        hijo2 = padre2.copy()  

        punto_cruce = np.random.randint(1, 4)

        if punto_cruce == 1:
            hijo1["cantidad_agua"] = padre2["cantidad_agua"]
            hijo2["cantidad_agua"] = padre1["cantidad_agua"]
        elif punto_cruce == 2:
            hijo1["cantidad_fertilizante"] = padre2["cantidad_fertilizante"]
            hijo2["cantidad_fertilizante"] = padre1["cantidad_fertilizante"]
        elif punto_cruce == 3:
            hijo1["cantidad_pesticida"] = padre2["cantidad_pesticida"]
            hijo2["cantidad_pesticida"] = padre1["cantidad_pesticida"]

        return hijo1, hijo2
    
    def mutar(self, individuo):
        
        if np.random.rand() < self.prob_mutacion:
            individuo["cantidad_agua"] = np.random.randint(0, self.cantidad_agua)
        if np.random.rand() < self.prob_mutacion:
            individuo["cantidad_fertilizante"] = np.random.randint(0, self.cantidad_fertilizante)
        if np.random.rand() < self.prob_mutacion:
            individuo["cantidad_pesticida"] = np.random.randint(0, self.cantidad_pesticida)
        return individuo
    
    def generar_nueva_poblacion(self, poblacion):
        nueva_poblacion = []

        poblacion = sorted(poblacion, key=lambda ind: ind["aptitud"], reverse=True)
        nueva_poblacion.extend(poblacion[:self.num_elite])

        while len(nueva_poblacion) < self.num_poblacion:
           
            padre1, padre2 = self.seleccion_por_torneo(poblacion)
            hijo1, hijo2 = self.cruzar(padre1, padre2)

            hijo1 = self.mutar(hijo1)
            hijo2 = self.mutar(hijo2)

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < self.num_poblacion:
                nueva_poblacion.append(hijo2)

        return nueva_poblacion
    

    def optimizar(self):

        
        poblacion = self.incializar_poblacion()

        for generacion in range(self.num_generaciones):
            
            poblacion = self.evaluar_poblacion(poblacion)

            
            poblacion = self.generar_nueva_poblacion(poblacion)

            mejor_individuo = max(poblacion, key=lambda ind: ind["aptitud"])
            print(f"Generación {generacion + 1}: Mejor individuo: {mejor_individuo}")

        # Retornar el mejor individuo encontrado
        mejor_individuo = max(poblacion, key=lambda ind: ind["aptitud"])
        return mejor_individuo