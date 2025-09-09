import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
import random
import glob
from tqdm import tqdm

def aplicar_transformaciones_color(imagen):
    """
    Aplica diferentes transformaciones de color a la imagen
    """
    # Convertir a PIL para mejor manejo de colores
    if isinstance(imagen, np.ndarray):
        if len(imagen.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(imagen, mode='L')
    else:
        pil_img = imagen
    
    transformaciones = []
    
    # 1. Cambiar brillo (±30%)
    brillo = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Brightness(pil_img)
    img_brillo = enhancer.enhance(brillo)
    transformaciones.append(('brillo', img_brillo))
    
    # 2. Cambiar contraste (±40%)
    contraste = random.uniform(0.6, 1.4)
    enhancer = ImageEnhance.Contrast(pil_img)
    img_contraste = enhancer.enhance(contraste)
    transformaciones.append(('contraste', img_contraste))
    
    # 3. Cambiar saturación (solo para imágenes a color)
    if pil_img.mode != 'L':
        saturacion = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Color(pil_img)
        img_saturacion = enhancer.enhance(saturacion)
        transformaciones.append(('saturacion', img_saturacion))
    
    # 4. Ajustar gamma
    gamma = random.uniform(0.8, 1.2)
    img_array = np.array(pil_img)
    img_gamma = np.power(img_array / 255.0, gamma) * 255.0
    img_gamma = Image.fromarray(img_gamma.astype(np.uint8))
    transformaciones.append(('gamma', img_gamma))
    
    return random.choice(transformaciones)[1]

def aplicar_transformaciones_bordes(imagen):
    """
    Aplica diferentes efectos de bordes y filtros
    """
    if isinstance(imagen, np.ndarray):
        pil_img = Image.fromarray(imagen)
    else:
        pil_img = imagen
    
    transformaciones = []
    
    # 1. Desenfoque gaussiano
    blur_radius = random.uniform(0.5, 2.0)
    img_blur = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    transformaciones.append(('blur', img_blur))
    
    # 2. Afilar imagen
    img_sharp = pil_img.filter(ImageFilter.SHARPEN)
    transformaciones.append(('sharpen', img_sharp))
    
    # 3. Detección de bordes suave
    img_edges = pil_img.filter(ImageFilter.FIND_EDGES)
    transformaciones.append(('edges', img_edges))
    
    # 4. Suavizado
    img_smooth = pil_img.filter(ImageFilter.SMOOTH)
    transformaciones.append(('smooth', img_smooth))
    
    # 5. Contorno
    img_contour = pil_img.filter(ImageFilter.CONTOUR)
    transformaciones.append(('contour', img_contour))
    
    return random.choice(transformaciones)[1]

def aplicar_transformaciones_geometricas(imagen):
    """
    Aplica transformaciones geométricas (rotación, escala, etc.)
    """
    if isinstance(imagen, np.ndarray):
        img_cv = imagen
    else:
        img_cv = np.array(imagen)
    
    altura, ancho = img_cv.shape[:2]
    transformaciones = []
    
    # 1. Rotación pequeña (-15° a +15°)
    # angulo = random.uniform(-15, 15)
    # centro = (ancho // 2, altura // 2)
    # matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    # img_rotada = cv2.warpAffine(img_cv, matriz, (ancho, altura), 
    #                            borderMode=cv2.BORDER_REFLECT_101)
    # transformaciones.append(('rotacion', img_rotada))
    
    # 2. Escalado (90% a 110%)
    escala = random.uniform(0.9, 1.1)
    nuevo_ancho = int(ancho * escala)
    nueva_altura = int(altura * escala)
    img_escalada = cv2.resize(img_cv, (nuevo_ancho, nueva_altura))
    
    # Recortar o rellenar para mantener tamaño original
    if escala > 1.0:
        # Recortar desde el centro
        start_x = (nuevo_ancho - ancho) // 2
        start_y = (nueva_altura - altura) // 2
        img_escalada = img_escalada[start_y:start_y+altura, start_x:start_x+ancho]
    else:
        # Agregar padding
        pad_x = (ancho - nuevo_ancho) // 2
        pad_y = (altura - nueva_altura) // 2
        img_escalada = cv2.copyMakeBorder(img_escalada, pad_y, altura-nueva_altura-pad_y,
                                        pad_x, ancho-nuevo_ancho-pad_x, 
                                        cv2.BORDER_REFLECT_101)
    
    transformaciones.append(('escala', img_escalada))
    
    # 3. Desplazamiento (shift)
    shift_x = random.randint(-5, 5)
    shift_y = random.randint(-5, 5)
    matriz_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img_shifted = cv2.warpAffine(img_cv, matriz_shift, (ancho, altura),
                               borderMode=cv2.BORDER_REFLECT_101)
    transformaciones.append(('shift', img_shifted))
    
    return random.choice(transformaciones)[1]

def aplicar_ruido(imagen):
    """
    Agrega diferentes tipos de ruido a la imagen
    """
    if isinstance(imagen, Image.Image):
        img_array = np.array(imagen)
    else:
        img_array = imagen.copy()
    
    transformaciones = []
    
    # 1. Ruido gaussiano
    ruido = np.random.normal(0, random.uniform(5, 15), img_array.shape)
    img_ruido = np.clip(img_array.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    transformaciones.append(('ruido_gaussiano', img_ruido))
    
    # 2. Ruido sal y pimienta
    probabilidad = random.uniform(0.001, 0.01)
    img_sp = img_array.copy()
    
    # Sal (píxeles blancos)
    coords_sal = tuple([np.random.randint(0, i - 1, int(probabilidad * img_array.size * 0.5)) 
                       for i in img_array.shape])
    img_sp[coords_sal] = 255
    
    # Pimienta (píxeles negros)
    coords_pimienta = tuple([np.random.randint(0, i - 1, int(probabilidad * img_array.size * 0.5)) 
                           for i in img_array.shape])
    img_sp[coords_pimienta] = 0
    
    transformaciones.append(('sal_pimienta', img_sp))
    
    return random.choice(transformaciones)[1]

def generar_imagen_aumentada(imagen_original):
    """
    Aplica una combinación aleatoria de transformaciones a una imagen
    """
    imagen = imagen_original.copy()
    
    # Lista de transformaciones disponibles
    transformaciones_disponibles = [
        aplicar_transformaciones_color,
        aplicar_transformaciones_bordes,
        aplicar_transformaciones_geometricas,
        aplicar_ruido
    ]
    
    # Aplicar 1-3 transformaciones aleatoriamente
    num_transformaciones = random.randint(1, 3)
    transformaciones_elegidas = random.sample(transformaciones_disponibles, num_transformaciones)
    
    for transformacion in transformaciones_elegidas:
        imagen = transformacion(imagen)
    
    return imagen

def generar_imagenes_aumentadas(carpeta_input, num_imagenes_objetivo=2000):
    """
    Genera imágenes aumentadas a partir de las imágenes en una carpeta
    
    Args:
        carpeta_input: Ruta de la carpeta con las imágenes originales
        num_imagenes_objetivo: Número total de imágenes que queremos generar
    """
    # Extensiones de imagen soportadas
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Encontrar todas las imágenes en la carpeta
    imagenes_originales = []
    for extension in extensiones:
        imagenes_originales.extend(glob.glob(os.path.join(carpeta_input, extension)))
        imagenes_originales.extend(glob.glob(os.path.join(carpeta_input, extension.upper())))
    
    if not imagenes_originales:
        print(f"No se encontraron imágenes en la carpeta: {carpeta_input}")
        return
    
    print(f"Encontradas {len(imagenes_originales)} imágenes originales")
    print(f"Generando {num_imagenes_objetivo} imágenes aumentadas...")
    
    contador_generadas = 0
    
    with tqdm(total=num_imagenes_objetivo, desc="Generando imágenes") as pbar:
        while contador_generadas < num_imagenes_objetivo:
            # Elegir imagen original aleatoria
            imagen_original_path = random.choice(imagenes_originales)
            
            try:
                # Cargar imagen
                imagen = cv2.imread(imagen_original_path)
                if imagen is None:
                    continue
                
                # Generar imagen aumentada
                imagen_aumentada = generar_imagen_aumentada(imagen)
                
                # Convertir a array numpy si es necesario
                if isinstance(imagen_aumentada, Image.Image):
                    imagen_aumentada = np.array(imagen_aumentada)
                    if len(imagen_aumentada.shape) == 3:
                        imagen_aumentada = cv2.cvtColor(imagen_aumentada, cv2.COLOR_RGB2BGR)
                
                # Generar nombre único
                nombre_original = os.path.splitext(os.path.basename(imagen_original_path))[0]
                extension = os.path.splitext(imagen_original_path)[1]
                nombre_nuevo = f"{nombre_original}_aug_{contador_generadas:04d}{extension}"
                ruta_nueva = os.path.join(carpeta_input, nombre_nuevo)
                
                # Guardar imagen aumentada
                cv2.imwrite(ruta_nueva, imagen_aumentada)
                
                contador_generadas += 1
                pbar.update(1)
                
                # Mostrar progreso cada 50 imágenes
                if contador_generadas % 50 == 0:
                    print(f"Generadas {contador_generadas}/{num_imagenes_objetivo} imágenes")
                    
            except Exception as e:
                print(f"Error procesando {imagen_original_path}: {e}")
                continue
    
    print(f"¡Proceso completado! Se generaron {contador_generadas} imágenes en: {carpeta_input}")

def mostrar_estadisticas_carpeta(carpeta):
    """
    Muestra estadísticas de las imágenes en la carpeta
    """
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    todas_las_imagenes = []
    for extension in extensiones:
        todas_las_imagenes.extend(glob.glob(os.path.join(carpeta, extension)))
        todas_las_imagenes.extend(glob.glob(os.path.join(carpeta, extension.upper())))
    
    originales = [img for img in todas_las_imagenes if '_aug_' not in os.path.basename(img)]
    aumentadas = [img for img in todas_las_imagenes if '_aug_' in os.path.basename(img)]
    
    print(f"\n=== ESTADÍSTICAS DE LA CARPETA ===")
    print(f"Carpeta: {carpeta}")
    print(f"Imágenes originales: {len(originales)}")
    print(f"Imágenes aumentadas: {len(aumentadas)}")
    print(f"Total de imágenes: {len(todas_las_imagenes)}")

# EJEMPLO DE USO
def ejemplo_uso():
    """
    Ejemplo de cómo usar el generador
    """
    carpeta_imagenes = "mi_carpeta_imagenes"  # Cambia por tu carpeta
    
    # Mostrar estadísticas antes
    mostrar_estadisticas_carpeta(carpeta_imagenes)
    
    # Generar 2000 imágenes aumentadas
    generar_imagenes_aumentadas(carpeta_imagenes, num_imagenes_objetivo=2000)
    
    # Mostrar estadísticas después
    mostrar_estadisticas_carpeta(carpeta_imagenes)

# Para usar el script:
if __name__ == "__main__":
    # CAMBIA ESTA RUTA POR LA TUYA
    carpeta_imagenes = input("Ingresa la ruta de la carpeta con imágenes: ").strip()
    
    if os.path.exists(carpeta_imagenes):
        num_imagenes = int(input("¿Cuántas imágenes quieres generar? (por defecto 2000): ") or "2000")
        generar_imagenes_aumentadas(carpeta_imagenes, num_imagenes)
        mostrar_estadisticas_carpeta(carpeta_imagenes)
    else:
        print(f"La carpeta {carpeta_imagenes} no existe.")