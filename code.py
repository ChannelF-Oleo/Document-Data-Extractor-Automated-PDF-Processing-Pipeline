import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageEnhance
from io import BytesIO
from google.cloud import vision
import pymysql
import json
try:
    import cv2  # Nuevo: Para detección de rostros
except Exception:
    cv2 = None
import numpy as np # Nuevo: Para cv2
try:
    import pytesseract  # Nuevo: Para OCR de teléfonos
except Exception:
    pytesseract = None
from concurrent.futures import ThreadPoolExecutor, as_completed # Nuevo: Para paralelizar
# --- Añadido para PyMuPDF ---
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False
# ------------------------------

# --- MEJORA: Config Tesseract ---
# Usa una variable de entorno preferida o detecta instalación en la ruta estándar de Windows
env_cmd = os.environ.get('TESSERACT_CMD')
default_exe = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
detected = None
if env_cmd:
    detected = env_cmd
elif os.path.exists(default_exe):
    detected = default_exe
# Si se detectó una ruta válida, configúrala en pytesseract; si no, dejaremos el valor por defecto
try:
    if detected:
        pytesseract.pytesseract.tesseract_cmd = detected
        os.environ['TESSERACT_CMD'] = detected
    else:
        # mantener el comportamiento previo pero advertir
        print("ADVERTENCIA: No se encontró Tesseract en PATH ni en la ruta estándar. La extracción de teléfonos podría fallar.")
except Exception as e:
    print(f"ADVERTENCIA: No se pudo configurar Tesseract ({detected}). La extracción de teléfonos fallará. {e}")
# ------------------------------

# Configura la variable de entorno para las credenciales de Google Cloud Vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'tu-key.json'

# --- MEJORA: Config Optimizada ---
DPI = 300  # MEJORA: Subido a 300 DPI para mejor calidad de OCR
PHOTO_SIZE = (150, 200)
# --- FIN DE MEJORA ---

# --- MEJORA: Carga del clasificador de rostros OpenCV ---
if cv2 is not None:
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        print(f"ERROR: No se pudo cargar el clasificador de rostros de OpenCV. {e}")
        print("Asegúrate de tener OpenCV instalado: pip install opencv-python")
        face_cascade = None
else:
    print("ADVERTENCIA: OpenCV (cv2) no está instalado. La detección de rostros se desactivará.")
    face_cascade = None
# ----------------------------------------------------

MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASS = ''
MYSQL_DB = 'padron_electoral'

vision_client = vision.ImageAnnotatorClient()

# --- FUNCIÓN DE LIMPIEZA (Mejorada para nulos) ---
def clean_string(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Limpia caracteres nulos o problemáticos para MySQL/JSON
    return text.encode('utf-8', 'replace').decode('utf-8').replace('\x00', '')
# --- FIN DE FUNCIÓN ---

# --- MEJORA: OCR por Bloque (Más barato y rápido) ---
def ocr_block(block_img: Image.Image) -> Dict:
    """Envía una imagen PEQUEÑA (bloque o header) a la API de Vision."""
    img_buffer = BytesIO()
    block_img.save(img_buffer, format='PNG')
    content = img_buffer.getvalue()
    image = vision.Image(content=content)

    # Usamos TEXT_DETECTION, más eficiente para bloques densos
    response = vision_client.text_detection(image=image)

    full_text = response.full_text_annotation.text if response.full_text_annotation else ''
    entities = []
    # response.text_annotations[0] es el texto completo, [1:] son las palabras
    for entity in response.text_annotations[1:]:
        entities.append({
            'text': entity.description.strip(),
            'bounds': entity.bounding_poly.vertices
        })
    return {'full_text': full_text, 'entities': entities}
# --- FIN DE MEJORA ---

def get_y_center(bounds: List) -> float:
    top_y = min(v.y for v in bounds)
    bottom_y = max(v.y for v in bounds)
    return (top_y + bottom_y) / 2

def get_x_center(bounds: List) -> float:
    left_x = min(v.x for v in bounds)
    right_x = max(v.x for v in bounds)
    return (left_x + right_x) / 2

# --- MEJORA: Funciones de parseo aisladas ---
def parse_headers_from_text(full_text: str) -> Dict:
    """Extrae los datos del encabezado del texto de un bloque de header."""
    headers = {
        'province': '',
        'municipality': '',
        'circunscripcion': '',
        'colegio_electoral': '',
        'recinto': ''
    }
    try:
        # --- ¡CORRECCIÓN REGEX! --- (de una versión anterior, para más robustez)
        province_search = re.search(r'Prov:\s*(\d+\s*-\s*.+?)(?=\n|$)', full_text, re.I)
        headers['province'] = clean_string(province_search.group(1).strip()) if province_search else ''

        municipality_search = re.search(r'Mun:\s*(\d+\s*-\s*.+?)(?=\n|$)', full_text, re.I)
        headers['municipality'] = clean_string(municipality_search.group(1).strip()) if municipality_search else ''

        circunscripcion_search = re.search(r'Circ\.\s*:\s*(\d+)', full_text)
        headers['circunscripcion'] = clean_string(circunscripcion_search.group(1).strip()) if circunscripcion_search else ''

        colegio_electoral_search = re.search(r'Elect\.\s*(\d+\s*\w*)', full_text, re.I)
        headers['colegio_electoral'] = clean_string(colegio_electoral_search.group(1).strip()) if colegio_electoral_search else ''

        recinto_search = re.search(r'Rec:\s*(\d+\s*-\s*.+?)(?=\n|$)', full_text, re.I)
        headers['recinto'] = clean_string(recinto_search.group(1).strip()) if recinto_search else ''
    except Exception as e:
        print(f" ⚠️ Error parseando headers: {e}")
    return headers

def parse_block_to_entry(block_img: Image.Image, ocr_result: Dict, block_idx: int, page_num: int,
                         block_crop_box: tuple, headers_dict: Dict) -> Dict:
    """Parsea el resultado de OCR de un SOLO bloque de persona."""
    entities = ocr_result['entities']
    block_text = ocr_result['full_text']

    cedula_entities = [e for e in entities if re.match(r'^\d{3}-\d{7}-\d$', e['text'])]
    if not cedula_entities:
        return None

    cedula = cedula_entities[0]['text']

    sorted_block = sorted(entities, key=lambda e: (get_y_center(e['bounds']), get_x_center(e['bounds'])))

    nombre, telefono, direccion = '', '', ''

    # Lógica de extracción de tu script original (bucles while)
    i = 0
    while i < len(sorted_block):
        ent = sorted_block[i]
        text = ent['text']
        if nombre == '' and text.isupper() and len(text) > 2 and cedula not in text and text not in ['EXT', 'JCE', 'CE', 'PA', 'ICE', 'JDE', 'ACE', 'UCE', 'VICE', 'JLE']:
            nombre = text
            j = i + 1
            while j < len(sorted_block):
                next_ent = sorted_block[j]
                delta_y = abs(get_y_center(next_ent['bounds']) - get_y_center(ent['bounds']))
                if delta_y < 10 and (next_ent['text'].isupper() or next_ent['text'] in [',', ' ']):
                    nombre += ' ' + next_ent['text'].strip()
                    j += 1
                else:
                    break
            i = j - 1
        
        elif direccion == '' and (text.startswith('Dir:') or text.startswith('C/') or '#' in text):
            direccion = text.replace('Dir:', '').strip()
            j = i + 1
            while j < len(sorted_block):
                next_ent = sorted_block[j]
                delta_y = abs(get_y_center(next_ent['bounds']) - get_y_center(ent['bounds']))
                if delta_y < 15 and any(char in next_ent['text'] for char in ['#', ',', 'C/', ' ']):
                    direccion += ' ' + next_ent['text']
                    j += 1
                else:
                    break
            i = j - 1
        i += 1

    # --- MEJORA: Extracción de Teléfono con Tesseract ---
    telefono = extract_tel_from_region(entities, sorted_block, block_img)

    if not telefono:  # Fallback a tu regex si Tesseract falla
        
        # --- ¡CAMBIO DE TELÉFONO 1! ---
        # Regex más precisa para buscar 'Tel: 1(809)...'
        tel_match = re.search(r'((?:Tel:|Cel:)\s*(?:1[\s-]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', block_text, re.I)
        # --- FIN DEL CAMBIO ---
        
        telefono = clean_string(tel_match.group(0)) if tel_match else ''
    # --- FIN DE MEJORA ---

    # Fallbacks para nombre y dirección
    if not nombre:
        nombre_match = re.search(r'([A-ZÁÉÍÓÚ ,.-]{5,50})', block_text)
        nombre = nombre_match.group(1) if nombre_match else ''
    if not direccion:
        dir_match = re.search(r'Dir:.*?(?=\sPRM|\(809\))', block_text, re.DOTALL | re.I)
        direccion = clean_string(dir_match.group(0).replace('Dir:', '').strip()) if dir_match else ''

    # --- NUEVA FUNCIONALIDAD: Extraer status de Voto ---
    voto_status = 'NO'  # Default
    voto_match = re.search(r'Vot[oó]:\s*(PA|PC|EXT)', block_text, re.I)
    if voto_match:
        voto_status = voto_match.group(1).upper()
    # --------------------------------------------------

    entry = {
        'cedula': cedula,
        'nombre': clean_string(nombre.strip()),
        'telefono': clean_string(telefono.strip()),
        'direccion': clean_string(direccion.strip()),
        'voto_status': voto_status,  # <-- NUEVO CAMPO
        'colegio_electoral': headers_dict['colegio_electoral'],
        'recinto': headers_dict['recinto'],
        'zona': '',
        'numero': str(block_idx + 1),  # 1-10
        'pagina': page_num,
        'foto_path': '',
        'foto_blob': None,
        'block_crop_box': block_crop_box  # Coordenadas (l, t, r, b) del bloque
    }
    return entry


# --- MEJORA: Función de Tesseract para Teléfonos ---
def extract_tel_from_region(entities: List[Dict], block_entities: List[Dict], block_img: Image.Image) -> str:
    """Usa Tesseract en una región específica para extraer números de teléfono."""
    tel_entities = [e for e in block_entities if any(kw in e['text'].upper() for kw in ['TEL:', 'CEL', '('])]
    if not tel_entities:
        return ''
    # Si pytesseract no está disponible, intentamos un fallback simple con el texto detectado
    if pytesseract is None:
        simple_text = tel_entities[0].get('text', '')
        # --- ¡CAMBIO DE TELÉFONO 2a! ---
        tels = re.findall(r'(?:1[\s-]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', simple_text)
        # --- FIN DEL CAMBIO ---
        return ' '.join(tels) if tels else ''

    try:
        tel_ent = tel_entities[0]
        bounds = tel_ent['bounds']
        x1, y1 = min(v.x for v in bounds), min(v.y for v in bounds)
        x2, y2 = max(v.x for v in bounds), max(v.y for v in bounds)

        w_block, h_block = block_img.size
        crop_x1 = max(0, x1 - 10)
        crop_y1 = max(0, y1 - 10)
        crop_x2 = min(w_block, x2 + 150)  # Expandir 150px a la derecha
        crop_y2 = min(h_block, y2 + 10)

        tel_region = block_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Config de Tesseract: psm 7 = tratar como una sola línea de texto
        tel_text = pytesseract.image_to_string(tel_region, config='--psm 7 -c tessedit_char_whitelist=0123456789()-+')

        # --- ¡CAMBIO DE TELÉFONO 2b! ---
        # Regex actualizada para capturar el '1' opcional
        tels = re.findall(r'(?:1[\s-]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', tel_text)
        # --- FIN DEL CAMBIO ---
        
        return ' '.join(tels) if tels else ''
    except Exception as e:
        print(f" ⚠️ Error en Tesseract (extract_tel_from_region): {e}")
        return ''
    # --- FIN DE MEJORA ---

# --- REFACTORIZACIÓN COMPLETA: process_pdf ---
def process_pdf(pdf_path: str) -> List[Dict]:
    """
    Procesa un PDF usando la estrategia de pre-corte:
    1. Renderiza con PyMuPDF.
    2. Corta el Header -> OCR (1 llamada) -> Parsea Headers.
    3. Corta 10 Bloques -> OCR en paralelo (10 llamadas) -> Parsea Entradas.
    4. Detecta Rostros en la página completa.
    5. Asocia Rostros a Entradas por contención espacial.
    """
    print(f"🔍 Procesando {os.path.basename(pdf_path)}")
    pages = []
    if HAVE_PYMUPDF:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                mat = fitz.Matrix(DPI / 72.0, DPI / 72.0)  # Usando DPI=300
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes(output="png")
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
                pages.append(img)
            doc.close()
        except Exception as e:
            print(f"⚠️ PyMuPDF falló: {e}")
            raise
    else:
        print("❌ PyMuPDF no está instalado. No se puede continuar.")
        return []

    all_entries = []
    for page_num, page_img in enumerate(pages, 1):
        
        # He quitado el guardado de DEBUG_page_1.png para limpiar el log,
        # pero puedes volver a ponerlo si lo necesitas para calibrar.
        # if page_num == 1:
        #     print(f"DEBUG: Guardando página 1 en DEBUG_page_1.png para calibración...")
        #     page_img.save('DEBUG_page_1.png')
        #     print(f"DEBUG: Dimensiones de la imagen: {page_img.size}")
    
        print(f" Procesando página {page_num}...")

        W, H = page_img.size

        # --- 1. OCR al Header ---
        
        # Mantenemos el 10% para evitar "sangrado" del header
        HEADER_HEIGHT = int(H * 0.10) 
        
        print(f" [Debug] Dimensiones pág: {W}x{H} | Header Height (10%): {HEADER_HEIGHT}px")

        HEADER_BOX = (0, 0, W, HEADER_HEIGHT)
        header_img = page_img.crop(HEADER_BOX)
        header_result = ocr_block(header_img)
        # Usamos la función de parseo de headers mejorada
        headers_dict = parse_headers_from_text(header_result['full_text'])
        print(f" [Header] Recinto: {headers_dict['recinto']}, Colegio: {headers_dict['colegio_electoral']}")

        # --- 2. Definición dinámica de 10 Cajas (Bloques) ---
        CROP_BOXES_10 = []
        col_width = W // 2
        start_y = HEADER_HEIGHT  # Empezar después del header
        block_height = (H - start_y) // 5  # 5 filas

        for i in range(5):  # 5 filas
            top = start_y + (i * block_height)
            bottom = top + block_height
            CROP_BOXES_10.append((0, top, col_width, bottom))  # Col 1
            CROP_BOXES_10.append((col_width, top, W, bottom))  # Col 2

        CROP_BOXES_10.sort(key=lambda box: (box[0], box[2]))  # Ordenar por col, luego fila

        blocks_to_process = []  # Tuplas (idx, box, img)
        for idx, box in enumerate(CROP_BOXES_10):
            block_img = page_img.crop(box)
            blocks_to_process.append((idx, box, block_img))

        # --- 3. OCR en Paralelo para los 10 Bloques ---
        page_entries = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_block = {
                executor.submit(ocr_block, block_img): (idx, box, block_img)
                for (idx, box, block_img) in blocks_to_process
            }

            for future in as_completed(future_to_block):
                idx, box, block_img = future_to_block[future]
                try:
                    ocr_result = future.result()
                    entry = parse_block_to_entry(block_img, ocr_result, idx, page_num,
                                                 box, headers_dict)
                    if entry:
                        page_entries.append(entry)
                except Exception as e:
                    print(f" ❌ Error en bloque {idx}: {e}")

        # --- 4. Detección de Rostros y Asociación Espacial ---
        page_entries.sort(key=lambda e: int(e['numero']))
        page_entries = detect_and_crop_faces_spatial(page_img, page_entries, CROP_BOXES_10)

        all_entries.extend(page_entries)
        print(f" [Page {page_num}] {len(page_entries)} entradas procesadas.")

    return all_entries
    # --- FIN DE REFACTORIZACIÓN ---

# --- MEJORA: Detección de Rostros (OpenCV) y Match Espacial ---
def detect_and_crop_faces_spatial(page_img: Image.Image, entries: List[Dict], crop_boxes: List[Dict]) -> List[Dict]:
    """Detecta todos los rostros en la página y los asocia al bloque que los contiene."""
    if not entries or face_cascade is None:
        return entries

    img_cv = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Parámetros flexibles (de la corrección anterior)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
    
    print(f" [Faces] Rostros detectados: {len(faces)}")

    if len(faces) == 0:
        print(" ⚠️ No se detectaron rostros. Usando fallback de crop fijo.")
        return _crop_fixed_fallback(page_img, entries, crop_boxes)

    os.makedirs('fotos', exist_ok=True)

    matched_entries = set()
    for x, y, w, h in faces:
        face_center_x = x + w / 2
        face_center_y = y + h / 2

        best_match_entry = None

        # Encontrar en qué 'entry' (y su 'block_crop_box') cae este rostro
        for entry in entries:
            if entry['cedula'] in matched_entries:
                continue

            left, top, right, bottom = entry['block_crop_box']

            # El rostro está DENTRO de este bloque
            if (left < face_center_x < right) and (top < face_center_y < bottom):
                best_match_entry = entry
                break

        if not best_match_entry:
            continue

        matched_entries.add(best_match_entry['cedula'])

        # Crop y save
        pad = int(0.1 * w)
        x_pad, y_pad = max(0, x - pad), max(0, y - pad)
        w_pad, h_pad = w + 2 * pad, h + 2 * pad

        face_cropped = img_cv[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        face_pil = Image.fromarray(cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB))
        face_pil = ImageEnhance.Sharpness(face_pil).enhance(1.5)

        cedula = best_match_entry['cedula']
        photo_path = os.path.join('fotos', f"{cedula}.jpg")
        resized = face_pil.resize(PHOTO_SIZE)
        resized.save(photo_path, 'JPEG', quality=90)

        img_buffer = BytesIO()
        resized.save(img_buffer, format='JPEG')
        best_match_entry['foto_blob'] = img_buffer.getvalue()
        best_match_entry['foto_path'] = photo_path
        print(f" 📸 {cedula} -> {photo_path} (Match espacial OK)")

    return entries


def _crop_fixed_fallback(page_img: Image.Image, entries: List[Dict], crop_boxes: List[Dict]) -> List[Dict]:
    """Fallback si OpenCV falla. Usa coordenadas relativas."""
    print(" [Fallback] Usando coordenadas fijas relativas para fotos...")
    os.makedirs('fotos', exist_ok=True)

    for entry in entries:
        try:
            # Obtener el índice (numero) del entry para saber qué crop_box usar
            idx = int(entry['numero']) - 1
            if idx >= len(crop_boxes):
                continue

            block_left, block_top, block_right, block_bottom = crop_boxes[idx]

            # --- ¡CALIBRA ESTOS VALORES! ---
            # (Estos son los valores que calibraste en el paso anterior)
            relative_x = 55      # Reemplaza con tu 'relative_x'
            relative_y = 40      # Reemplaza con tu 'relative_y'
            photo_width = 110    # Reemplaza con tu 'photo_width'
            photo_height = 120   # Reemplaza con tu 'photo_height'
            # ---------------------------------

            photo_left = block_left + relative_x
            photo_right = photo_left + photo_width
            photo_top = block_top + relative_y
            photo_bottom = photo_top + photo_height

            box = (photo_left, photo_top, photo_right, photo_bottom)

            photo = page_img.crop(box)
            cedula = entry['cedula']
            photo_path = os.path.join('fotos', f"{cedula}.jpg")
            resized = photo.resize(PHOTO_SIZE)
            resized.save(photo_path, 'JPEG', quality=90)
            img_buffer = BytesIO()
            resized.save(img_buffer, format='JPEG')
            entry['foto_blob'] = img_buffer.getvalue()
            entry['foto_path'] = photo_path
        except Exception as e:
            print(f" ❌ ERROR en fallback de foto para {entry.get('cedula', 'N/A')}: {e}")

    return entries
# --- FIN DE MEJORA ---

# --- MEJORA: create_mysql_db (más robusto) ---
def create_mysql_db(all_data: List[Dict]):
    if not all_data:
        print("💾 No hay datos para insertar en MySQL.")
        return
    try:
        conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASS, database=MYSQL_DB, charset='utf8mb4')
        cursor = conn.cursor()

        insert_count = 0
        update_count = 0

        for entry in all_data:
            foto_blob_data = entry['foto_blob'] if entry['foto_blob'] else pymysql.NULL

            # SQL con el nuevo campo voto_status
            sql = """
            INSERT INTO personas (
                cedula, nombre, telefono, direccion, voto_status,
                colegio_electoral, recinto, zona, numero, pagina, foto_path, foto
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                nombre=VALUES(nombre),
                telefono=VALUES(telefono),
                direccion=VALUES(direccion),
                voto_status=VALUES(voto_status),
                colegio_electoral=VALUES(colegio_electoral),
                recinto=VALUES(recinto),
                zona=VALUES(zona),
                numero=VALUES(numero),
                pagina=VALUES(pagina),
                foto_path=VALUES(foto_path),
                foto=VALUES(foto)
            """

            # `result` es 1 para INSERT, 2 para UPDATE
            result = cursor.execute(sql, (
                entry['cedula'], entry['nombre'], entry['telefono'], entry['direccion'],
                entry.get('voto_status'),  # Nuevo campo (usar get por seguridad)
                entry['colegio_electoral'], entry['recinto'], entry['zona'],
                entry['numero'], entry['pagina'], entry['foto_path'], foto_blob_data
            ))

            if result == 1:
                insert_count += 1
            elif result == 2:
                update_count += 1

        conn.commit()
        print(f"💾 MySQL listo: {insert_count} nuevos registros, {update_count} actualizados.")
    except pymysql.err.OperationalError as e:
        print(f"❌ ERROR de conexión MySQL: {e}")
        print(" Asegúrate de que el servidor MySQL esté corriendo y la DB 'padron_electoral' exista.")
    except Exception as e:
        print(f"❌ ERROR en MySQL: {e}")
    finally:
        if 'conn' in locals() and getattr(conn, 'open', True):
            conn.close()
# --- FIN DE MEJORA ---


# --- ¡NUEVA FUNCIÓN DE GUARDADO! ---
def save_data(all_data: List[Dict]):
    """Guarda los datos en CSV, JSON y MySQL."""
    if not all_data:
        print("📄 No hay datos para guardar.")
        return

    print("\n💾 Guardando progreso...")
    os.makedirs('outputs', exist_ok=True)
    
    # Quitar columnas internas antes de guardar
    df = pd.DataFrame(all_data)
    df_to_save = df.drop(columns=['block_crop_box', 'foto_blob'], errors='ignore')

    try:
        print("📄 Guardando en formato JSON...")
        data_list = df_to_save.to_dict('records')
        # Usamos 'a' (append) para JSON, aunque esto puede crear un JSON inválido si son múltiples listas.
        # Mejor: guardamos un JSON por zona.
        # O, sobrescribimos el JSON principal cada vez. Sobrescribir es más seguro.
        with open('outputs/padron_limpio.json', 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        print(" ...JSON guardado exitosamente (sobrescrito).")
    except Exception as e:
        print(f"❌ Error al guardar JSON: {e}")

    try:
        print("📄 Guardando en formato CSV...")
        # Usamos 'a' (append) para CSV si el archivo ya existe, y quitamos el header
        file_exists = os.path.exists('outputs/padron_limpio.csv')
        df_to_save.to_csv('outputs/padron_limpio.csv', 
                          mode='a' if file_exists else 'w', 
                          header=not file_exists, 
                          index=False, 
                          encoding='utf-8-sig')
        print(" ...CSV guardado exitosamente (añadido).")
    except Exception as e:
        print(f"❌ Error al guardar CSV: {e}")

    # Guardar en DB (pasa all_data completo, con 'foto_blob')
    # La función create_mysql_db ya maneja la lógica de INSERT/UPDATE
    create_mysql_db(all_data)
# --- FIN DE LA NUEVA FUNCIÓN ---


# --- ¡FUNCIÓN process_folder MODIFICADA! ---
def process_folder(root_folder: str = 'PADRON ELECTORAL POR ZONA'):
    total_processed_data = [] # Para el conteo final

    if not os.path.exists(root_folder):
        print(f"❌ No se encontró la carpeta raíz: {root_folder}")
        return

    # Obtenemos la lista de zonas primero
    try:
        all_zonas = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    except Exception as e:
        print(f"❌ Error leyendo la carpeta raíz: {e}")
        return
        
    for zona_folder in all_zonas:
        zona_path = os.path.join(root_folder, zona_folder)
        
        zona = zona_folder.split('-')[0].strip() if '-' in zona_folder else zona_folder
        print(f"\n🏢 Procesando Zona: {zona}")

        current_zone_data = [] # Aquí guardamos los datos SÓLO de esta zona

        for escuela_folder in os.listdir(zona_path):
            escuela_path = os.path.join(zona_path, escuela_folder)
            if not os.path.isdir(escuela_path):
                continue
            recinto_local = escuela_folder.replace('Carpeta ', '').strip()

            pdf_files = [f for f in os.listdir(escuela_path) if f.endswith('.pdf')]
            if not pdf_files:
                continue

            print(f" 🏫 Recinto: {recinto_local} ({len(pdf_files)} PDFs)")

            for pdf_file in pdf_files:
                
                # (Puedes descomentar tus líneas de prueba si las necesitas)
                # if '1312A' not in pdf_file and '1256A' not in pdf_file:
                #    continue

                pdf_path = os.path.join(escuela_path, pdf_file)

                try:
                    entries = process_pdf(pdf_path)
                    for entry in entries:
                        entry['zona'] = zona
                        entry['recinto'] = entry['recinto'] or recinto_local
                        current_zone_data.append(entry) # Añadimos a los datos de la zona actual
                except Exception as e:
                    print(f"❌❌❌ ERROR PROCESANDO EL ARCHIVO {pdf_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # --- ¡NUEVA LÓGICA DE GUARDADO POR ZONA! ---
        if current_zone_data:
            print(f"\n💾 Zona '{zona}' completada. Guardando {len(current_zone_data)} registros...")
            save_data(current_zone_data) # Llamamos a la nueva función de guardado
            total_processed_data.extend(current_zone_data) # Añadimos al conteo total
        else:
            print(f"\n🤷 Zona '{zona}' completada sin nuevos datos.")

        # --- ¡NUEVA LÓGICA DE PREGUNTA! ---
        try:
            respuesta = input(f"¿Desea continuar con la siguiente zona? (s/n): ").strip().lower()
            if respuesta not in ['s', 'si', 'y', 'yes', '']: # Aceptar Enter (vacío) como 'sí'
                print("🛑 Proceso detenido por el usuario.")
                break # Sale del bucle de 'zona_folder'
        except KeyboardInterrupt:
            print("\n🛑 Proceso detenido por el usuario (Ctrl+C).")
            break
        # --- FIN DE LA NUEVA LÓGICA ---

    # --- LÓGICA FINAL MODIFICADA ---
    if total_processed_data:
        print(f"\n🎉 ¡Proceso finalizado! {len(total_processed_data)} registros totales procesados en esta sesión.")
    else:
        print(f"\n❌ Proceso finalizado sin errores, pero no se extrajo ningún dato.")
# --- FIN DE LA FUNCIÓN MODIFICADA ---


if __name__ == '__main__':
    # --- Verificaciones Previas ---
    if not os.path.exists('tu-key.json'):
        print("❌ ADVERTENCIA: No se encontró 'tu-key.json'. La API de Google Cloud Vision fallará.")
    
    # Verificación de Tesseract actualizada
    tesseract_path_ok = False
    if detected: # 'detected' es la variable que definimos al inicio
        if os.path.exists(detected):
            tesseract_path_ok = True
        else:
             print(f"❌ ADVERTENCIA: Tesseract CMD está configurado como '{detected}' pero no se encontró.")
    else:
        print(f"❌ ADVERTENCIA: No se detectó Tesseract. La extracción de teléfono podría fallar.")

    if not HAVE_PYMUPDF:
        print(f"❌ ADVERTENCIA: PyMuPDF (fitz) no está instalado. El renderizado de PDF fallará.")
    if face_cascade is None:
        print(f"❌ ADVERTENCIA: El clasificador de rostros de OpenCV no se cargó. La detección de rostros fallará.")
    # --- Fin de Verificaciones ---

    process_folder()
