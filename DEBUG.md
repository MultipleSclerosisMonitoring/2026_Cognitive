# Guía de Depuración - 2026_Cognitive

## 🎯 Configuraciones de Debugging Disponibles

### 1. **🐛 Debug - Configuración Estándar** (Recomendada)
- **Nivel de verbosidad**: INFO (3)
- **Uso**: Depuración normal del programa
- **Características**:
  - Carga automáticamente `.env`
  - Pasa configuración desde `calibration/config.yaml`
  - Valida que `.env` exista antes de ejecutar
  - Muestra valores de retorno

**Comando equivalente:**
```bash
python -m calibration.main --config calibration/config.yaml --verbose 3
```

---

### 2. **🐛 Debug - Verbose (DEBUG completo)**
- **Nivel de verbosidad**: DEBUG (4)
- **Uso**: Depuración profunda, incluye librerías de scikit-learn
- **Características**:
  - Muestra TODOS los mensajes de logging
  - No excluye código de librerías (`justMyCode=false`)
  - Útil para debuggear problemas complejos

**Comando equivalente:**
```bash
DEBUG=1 python -m calibration.main --config calibration/config.yaml --verbose 4
```

---

### 3. **⚠️ Debug - Warning Level**
- **Nivel de verbosidad**: WARNING (2)
- **Uso**: Ejecución más silenciosa, solo advertencias y errores
- **Características**:
  - Menos ruido en consola
  - Enfoque en problemas críticos

---

### 4. **🧪 Debug - Con Parámetros Personalizados**
- **Uso**: Modificar argumentos manualmente
- **Cómo usarlo**:
  1. Abre `.vscode/launch.json`
  2. Edita el array `args` en esta configuración
  3. Inicia el debugging (F5)

**Ejemplo - Para cambiar el archivo de configuración:**
```json
"args": [
    "--config",
    "calibration/config_custom.yaml",
    "--verbose",
    "3"
]
```

---

### 5. **🔍 Debug - unittest (Pruebas Unitarias)**
- **Uso**: Ejecutar suite de pruebas con pytest
- **Características**:
  - Modo verbose con salida corta de errores
  - Carga `.env` para pruebas
  - Ideal para validar cambios

**Comando equivalente:**
```bash
pytest -v --tb=short tests/
```

---

### 6. **📊 Debug - Archivo YAML Específico**
- **Uso**: Seleccionar dinámicamente un archivo YAML
- **Características**:
  - Abre diálogo de selección de archivo
  - Útil para probar múltiples configuraciones

---

### 7. **🔐 Debug - Sin Cargar .env**
- **Uso**: Probar fallback a credenciales en `config.yaml`
- **Características**:
  - No carga variables de `.env`
  - Valida que el fallback funciona correctamente
  - Seguridad: verifica que no se filtren credenciales

---

## 🔨 Tareas Pre-Launch (Validación Automática)

Antes de ejecutar cualquier debugging, VS Code puede validar el entorno:

### Tareas Disponibles (Ctrl+Shift+B):

1. **validar-env**: Verifica que `.env` exista
2. **instalar-dependencias**: `pip install -e .`
3. **limpiar-cache**: Limpia `__pycache__` y `.pyc`
4. **ejecutar-test**: Ejecuta pytest
5. **listar-configuraciones**: Muestra el YAML cargado
6. **validar-config-yaml**: Valida sintaxis de YAML
7. **crear-env-desde-ejemplo**: Crea `.env` desde `.env.example`

---

## 🚀 Flujo de Debugging Típico

### Opción A: Depuración Rápida
```
1. Presiona F5
2. Selecciona "🐛 Debug - Configuración Estándar"
3. VS Code valida .env automáticamente
4. El programa se inicia en modo depuración
5. Los breakpoints funcionan normalmente
```

### Opción B: Depuración Profunda
```
1. Presiona F5
2. Selecciona "🐛 Debug - Verbose (DEBUG completo)"
3. Ahora ves todos los logs incluyendo librerías
4. Puedes debuggear problemas en scikit-learn, pandas, etc.
```

### Opción C: Depuración con Configuración Personalizada
```
1. Edita .vscode/launch.json
2. Modifica el array "args" de tu configuración
3. Presiona F5 y selecciona la configuración
4. El programa usa tus argumentos personalizados
```

---

## 🔍 Variables de Entorno Automáticas

Las siguientes variables se inyectan automáticamente:

```
PYTHONPATH=${workspaceFolder}       # Ruta raíz del proyecto
PYTHONUNBUFFERED=1                 # Output sin buffer
PYTHONDONTWRITEBYTECODE=0          # Permite .pyc para debugging
```

## 📋 Requisitos Previos

### 1. Crear archivo `.env`
```bash
cp .env.example .env
# Edita .env con tus credenciales reales
```

### 2. Instalar dependencias
```bash
pip install -e .
# O usar la tarea: Ctrl+Shift+B → instalar-dependencias
```

### 3. Validar configuración
```bash
# Usar la tarea: Ctrl+Shift+B → validar-config-yaml
```

---

## 🎯 Breakpoints y Stepping

### Punto de Entrada Automático
El debugger inicia en `calibration/main.py:main()` por defecto.

### Agregar Breakpoints
1. Haz clic en el área gris a la izquierda del número de línea
2. Debería aparecer un punto rojo
3. Durante debugging, la ejecución se pausará en ese punto

### Navegación
- **F10**: Step Over (salta a siguiente línea)
- **F11**: Step Into (entra en función)
- **Shift+F11**: Step Out (sale de función actual)
- **F5**: Continue (continúa ejecución)

---

## 📊 Monitoreando Variables

### Variables de Watch
En el panel de debugging (lado izquierdo):
1. Abre la pestaña "Watch"
2. Haz clic en "+" y escribe el nombre de una variable
3. VS Code mostrará su valor en tiempo real

### Variables Locales
Automáticamente visible en el panel "Variables" cuando pausas ejecución.

---

## 🐛 Debugging de Excepciones

### Pausar en Excepciones
1. En el panel "Debug Console" (abajo)
2. Click en el icono de engranaje ("Breakpoint Settings")
3. Activa "Caught Exceptions" o "Uncaught Exceptions"

### Inspeccionar el Traceback Completo
Cuando una excepción ocurre:
1. El debugging pausa automáticamente
2. En "Call Stack" ves la cadena de llamadas
3. Haz click en cada frame para inspeccionar variables locales

---

## 🔐 Seguridad: Protegiendo Credenciales

### ✅ Buenas Prácticas

**El archivo `.env` NO se commitea** (está en `.gitignore`):
```
.env          ← Variables de entorno reales
.env.example  ← Template (seguro de commitear)
```

**Debugging sin filtrar secretos:**
```bash
# ✓ Seguro: credenciales en .env (no visible en logs por defecto)
python -m calibration.main --config calibration/config.yaml

# ✗ Inseguro: credenciales en command line
python -m calibration.main --db-uri "postgresql://user:pass@host/db"
```

**En config.yaml:**
```yaml
# ✓ Seguro: usa env vars
data:
  db_host: "localhost"  # fallback solo
  # DB_HOST env var toma precedencia

# ✗ Inseguro: credenciales hardcodeadas
data:
  db_uri: "postgresql://lectura:password123@host/db"
```

---

## 🆘 Troubleshooting

### Problema: ".env no encontrado"
```bash
# Solución:
cp .env.example .env
# Luego edita .env con tus valores reales
```

### Problema: "ModuleNotFoundError: No module named 'calibration'"
```bash
# Solución:
pip install -e .
# O usar tarea: Ctrl+Shift+B → instalar-dependencias
```

### Problema: "YAML parsing error"
```bash
# Solución:
# Usar tarea: Ctrl+Shift+B → validar-config-yaml
# Verifica indentación en config.yaml (YAML usa espacios, no tabs)
```

### Problema: Breakpoints no se detienen
```bash
# Soluciones:
1. Limpia cache: Ctrl+Shift+B → limpiar-cache
2. Reinicia VS Code
3. Verifica que el archivo fue guardado (Ctrl+S)
```

### Problema: Variables no se muestran en Watch
```bash
# Solución:
1. La ejecución debe estar pausada
2. La variable debe estar en scope actual
3. Asegúrate de usar el nombre exacto de la variable
```

---

## 🔧 Personalizaciones Avanzadas

### Modificar Nivel de Verbosidad Global
En `.vscode/launch.json`, cambia `--verbose 3`:
```json
"args": [
    "--config", "calibration/config.yaml",
    "--verbose", "4"  ← Cambiar este número (0-4)
]
```

Niveles:
- 0 = CRITICAL (solo errores fatales)
- 1 = ERROR (solo errores)
- 2 = WARNING (advertencias)
- 3 = INFO (información normal)
- 4 = DEBUG (todo, incluyendo librerías)

### Agregar Nueva Configuración de Debug
Edita `.vscode/launch.json` y agrega:
```json
{
    "name": "Mi Configuración Personalizada",
    "type": "python",
    "request": "launch",
    "module": "calibration.main",
    "args": ["--config", "calibration/config.yaml"],
    "console": "integratedTerminal",
    "envFile": "${workspaceFolder}/.env"
}
```

---

## 📚 Recursos Adicionales

- [VS Code Python Debugging](https://code.visualstudio.com/docs/python/debugging)
- [Python logging módulo](https://docs.python.org/3/library/logging.html)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)

---

**Última actualización**: Mayo 27, 2026
