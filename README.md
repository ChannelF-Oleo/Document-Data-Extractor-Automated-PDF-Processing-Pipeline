# Document Data Extractor: Automated PDF Processing Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Visión General

**Document Data Extractor** es un framework modular en Python diseñado para automatizar la extracción de datos estructurados de documentos PDF escaneados o digitales. Este pipeline combina tecnologías avanzadas de visión por computadora, reconocimiento óptico de caracteres (OCR) y procesamiento paralelo para transformar documentos complejos en datos utilizables, listos para análisis, almacenamiento o integración con sistemas empresariales.

Ideal para escenarios donde se manejan volúmenes altos de documentos semi-estructurados (como formularios, registros administrativos, informes o catálogos), el script divide el procesamiento en etapas eficientes: renderizado de alta resolución, segmentación de regiones de interés, extracción de texto y entidades, detección de elementos visuales (como imágenes o firmas), y exportación a múltiples formatos. 

Con un enfoque en la escalabilidad y la reducción de costos (minimizando llamadas a APIs externas mediante procesamiento local), este proyecto es perfecto para desarrolladores, analistas de datos y equipos de automatización que buscan una solución robusta y personalizable. Soporta flujos de trabajo por lotes, con pausas interactivas para procesamiento iterativo, y maneja errores gracefully para entornos de producción.

## 🔧 Características Clave

- **Renderizado Optimizado de PDFs**: Utiliza PyMuPDF para convertir páginas en imágenes de alta resolución (configurable hasta 300 DPI), preservando detalles finos sin sobrecargar la memoria.
  
- **Segmentación Inteligente**: Divide automáticamente las páginas en bloques/regiones (e.g., headers, columnas, filas) usando coordenadas dinámicas, adaptable a layouts variables como tablas o formularios de múltiples columnas.

- **OCR Híbrido y Eficiente**:
  - API principal: Google Cloud Vision para texto denso y entidades (con llamadas por bloque para ahorrar costos).
  - Fallback local: Tesseract para tareas específicas (e.g., números de contacto o códigos), con configuración automática de rutas.
  - Procesamiento paralelo: ThreadPoolExecutor para OCR simultáneo en múltiples bloques, acelerando el throughput.

- **Extracción de Entidades Personalizable**:
  - Regex y heurísticas para campos como identificadores (e.g., códigos alfanuméricos), nombres, direcciones o metadatos.
  - Soporte para estados o categorías (e.g., "activo/inactivo") vía patrones configurables.
  - Limpieza robusta: Manejo de caracteres nulos, codificación UTF-8 y normalización para compatibilidad con bases de datos.

- **Visión por Computadora Integrada**:
  - Detección de elementos visuales (e.g., rostros, logos o firmas) usando OpenCV con clasificadores pre-entrenados.
  - Asociación espacial: Vincula detecciones a regiones específicas mediante coordenadas de bounding boxes.
  - Recorte y mejora: Ajuste automático de imágenes (e.g., resizing, sharpening) y guardado como blobs o archivos.

- **Almacenamiento Flexible**:
  - **CSV/JSON**: Exportación incremental (append por lote) para análisis rápido con Pandas.
  - **Base de Datos**: Integración con MySQL (o adaptable) para INSERT/UPDATE idempotente, evitando duplicados.
  - **Archivos Multimedia**: Guardado de imágenes extraídas en carpetas dedicadas.

- **Escalabilidad y UX**:
  - Procesamiento por carpetas/jerarquías (e.g., por "categorías" o "subgrupos").
  - Verificaciones previas: Detecta dependencias faltantes (e.g., Tesseract, credenciales API).
  - Logs detallados y pausas interactivas para sesiones largas.
  - Fallbacks automáticos: Coordenadas fijas o modos degradados si componentes fallan.

## 📋 Requisitos del Sistema

- **Python 3.8+** con entornos virtuales recomendados.
- **Dependencias Core** (instala vía `pip install -r requirements.txt`):
  - `pandas`, `Pillow`: Manejo de datos e imágenes.
  - `google-cloud-vision`: OCR en la nube (requiere credenciales JSON).
  - `PyMuPDF` (fitz): Renderizado de PDFs.
  - `opencv-python`, `numpy`: Detección visual.
  - `pytesseract`: OCR local (opcional).
  - `pymysql`: Conexión a DB.
- **Herramientas Externas**:
  - Tesseract OCR: Descarga desde [GitHub](https://github.com/tesseract-ocr/tesseract); el script lo detecta automáticamente.
  - Google Cloud: Habilita Vision API y coloca `credentials.json` en la raíz.
  - MySQL: Servidor local (configurable); crea tabla base con el schema proporcionado.

### Schema de Ejemplo para DB (Adaptable)
```sql
CREATE TABLE documents (
    id_field VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    contact VARCHAR(100),
    address TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    category VARCHAR(100),
    subcategory VARCHAR(100),
    group VARCHAR(50),
    position VARCHAR(10),
    page_num INT,
    media_path VARCHAR(255),
    media_blob LONGBLOB
) CHARACTER SET utf8mb4;
```

## 🚀 Guía de Instalación y Uso

1. **Clonación**:
   ```bash
   git clone https://github.com/tu-usuario/document-data-extractor.git
   cd document-data-extractor
   ```

2. **Setup**:
   ```bash
   pip install -r requirements.txt
   # Configura env vars: export GOOGLE_APPLICATION_CREDENTIALS=credentials.json
   # Opcional: export TESSERACT_CMD=/path/to/tesseract
   ```

3. **Estructura de Entrada** (Flexible):
   ```
   INPUT_FOLDER/
   ├── Category-A/
   │   ├── Subgroup-1/
   │   │   ├── file1.pdf
   │   │   └── file2.pdf
   │   └── Subgroup-2/
   │       └── file3.pdf
   └── Category-B/
       └── Subgroup-3/
           └── file4.pdf
   ```
   - Adapta `process_folder()` para tus jerarquías (e.g., categorías como "zonas", subgrupos como "recintos").

4. **Ejecución**:
   ```bash
   python main.py  # O 'code.py' en el repo
   ```
   - Procesa por categoría, con prompts para continuar/pausar.
   - Salidas: `./outputs/data_export.csv/json`, `./media/`, y DB actualizada.

5. **Personalización**:
   - Edita regex en `parse_block_to_entry()` para tus entidades.
   - Ajusta DPI, tamaños de crop o parámetros de detección en constantes globales.
   - Extiende para otros OCR (e.g., AWS Textract) o DBs (e.g., PostgreSQL).

Ejemplo de Salida (JSON):
```json
[
  {
    "id_field": "ABC-1234567-DEF",
    "name": "Ejemplo Nombre Completo",
    "contact": "Tel: (555) 123-4567",
    "address": "Dirección Detallada #123, Ciudad",
    "status": "approved",
    "category": "A",
    "subcategory": "Subgroup-1",
    "group": "Main",
    "position": "1",
    "page_num": 1,
    "media_path": "media/ABC-1234567-DEF.jpg"
  }
]
```

## 🛡️ Solución de Problemas Comunes

| Issue | Posible Causa | Fix |
|-------|---------------|-----|
| **OCR Inexacto** | Resolución baja o API no configurada. | Aumenta DPI; verifica credenciales. |
| **Detección Visual Falla** | OpenCV no instalado o lighting pobre en scans. | `pip install opencv-python`; usa fallback fijo. |
| **Memoria Excedida** | PDFs grandes/multi-página. | Baja DPI o procesa por lotes pequeños. |
| **DB Errores** | Credenciales inválidas. | Edita vars MySQL; prueba conexión manual. |
| **Tesseract No Encontrado** | Instalación faltante. | Instala y setea `TESSERACT_CMD`. |

- **Debug Mode**: Habilita saves de imágenes intermedias (e.g., crops) para calibrar.

## 🤝 Contribución y Comunidad

¡Colabora! Abre issues para bugs, features o adaptaciones (e.g., soporte para invoices o CVs). Sigue el flujo estándar: fork → branch → PR.

- **Roadmap**: Soporte multi-idioma OCR, UI web para previews, integración con ML para clasificación de docs.

## 📄 Licencia

MIT - Libre para uso, modificación y distribución.

## 🙌 Créditos

Inspirado en herramientas de automatización de docs. Gracias a Google Vision, OpenCV y comunidades OSS por las bases sólidas.

¡Automatiza tu flujo de documentos hoy! Si necesitas adaptaciones, ¡házmelo saber! 📊
