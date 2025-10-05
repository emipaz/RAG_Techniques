# Configuración del Entorno de Desarrollo - RAG Techniques

Esta guía te ayudará a configurar un entorno virtual optimizado para trabajar con las técnicas RAG, utilizando `venv`, `uv` para acelerar instalaciones, e `ipykernel` para Jupyter.

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip actualizado
- Git configurado

## 🚀 Paso 1: Crear el Entorno Virtual

### Windows (PowerShell)
```powershell
# Navegar al directorio del proyecto
cd D:\RAG_Techniques

# Crear entorno virtual
python -m venv rag_env

# Activar el entorno virtual
.\rag_env\Scripts\Activate.ps1
```

### Linux/Mac
```bash
# Navegar al directorio del proyecto
cd /path/to/RAG_Techniques

# Crear entorno virtual
python -m venv rag_env

# Activar el entorno virtual
source rag_env/bin/activate
```

## ⚡ Paso 2: Instalar UV para Acelerar Instalaciones

UV es un instalador de paquetes Python ultrarrápido que puede acelerar significativamente las instalaciones.

```powershell
# Con el entorno virtual activado
pip install uv

# Verificar la instalación
uv --version
```

## 📦 Paso 3: Instalar Dependencias Base

```powershell
# Actualizar pip usando uv
uv pip install --upgrade pip

# Instalar dependencias básicas para Jupyter
uv pip install jupyter jupyterlab ipykernel

# Instalar dependencias del proyecto (si existe requirements.txt)
uv pip install -r requirements.txt
```

## 🔧 Paso 4: Configurar IPython Kernel

Para que Jupyter pueda usar nuestro entorno virtual como kernel:

```powershell
# Instalar el kernel en Jupyter
python -m ipykernel install --user --name=rag_env --display-name="RAG Techniques (Python)"
```

### Verificar la instalación del kernel
```powershell
# Listar kernels disponibles
jupyter kernelspec list
```

Deberías ver algo como:
```
Available kernels:
  python3    /Users/username/.../python3
  rag_env    /Users/username/.../rag_env
```

## 📚 Paso 5: Instalar Dependencias Específicas para RAG

```powershell
# Dependencias comunes para RAG
uv pip install langchain langchain-openai
uv pip install chromadb faiss-cpu
uv pip install openai tiktoken
uv pip install transformers sentence-transformers
uv pip install pandas numpy matplotlib seaborn
uv pip install python-dotenv
uv pip install streamlit gradio  # Para demos interactivas

# Para trabajar con PDFs y documentos
uv pip install pypdf2 python-docx
uv pip install unstructured[local-inference]

# Para evaluación
uv pip install ragas deepeval

# Dependencias adicionales para técnicas avanzadas
uv pip install llama-index neo4j
```

## 🔐 Paso 6: Configurar Variables de Entorno

Crear archivo `.env` para las API keys:

```powershell
# Crear archivo .env (si no existe)
New-Item -ItemType File -Path ".env" -Force
```

Agregar al archivo `.env`:
```env
# OpenAI
OPENAI_API_KEY=tu_api_key_aqui

# Otras APIs que puedas necesitar
ANTHROPIC_API_KEY=tu_anthropic_key
HUGGINGFACE_API_TOKEN=tu_hf_token

# Configuraciones locales
PYTHONPATH=D:\RAG_Techniques
```

## 🚀 Paso 7: Iniciar Jupyter

```powershell
# Iniciar Jupyter Lab
jupyter lab

# O Jupyter Notebook clásico
jupyter notebook
```

## 📋 Paso 8: Verificar la Configuración

Crear un notebook de prueba para verificar que todo funciona:

```python
# Celda 1: Verificar imports básicos
import sys
print(f"Python: {sys.version}")
print(f"Entorno virtual activo: {sys.prefix}")

# Celda 2: Verificar dependencias RAG
try:
    import langchain
    import openai
    import chromadb
    print("✅ Dependencias RAG instaladas correctamente")
except ImportError as e:
    print(f"❌ Error en dependencias: {e}")

# Celda 3: Verificar variables de entorno
import os
from dotenv import load_dotenv

load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    print("✅ Variables de entorno configuradas")
else:
    print("⚠️ Revisar configuración de .env")
```

## 🔄 Comandos Útiles para el Día a Día

### Activar/Desactivar Entorno
```powershell
# Activar
.\rag_env\Scripts\Activate.ps1

# Desactivar
deactivate
```

### Gestión de Paquetes con UV
```powershell
# Instalar paquete específico
uv pip install nombre_paquete

# Actualizar paquete
uv pip install --upgrade nombre_paquete

# Listar paquetes instalados
uv pip list

# Generar requirements.txt
uv pip freeze > requirements.txt
```

### Gestión de Kernels
```powershell
# Listar kernels
jupyter kernelspec list

# Eliminar kernel (si es necesario)
jupyter kernelspec remove rag_env

# Reinstalar kernel
python -m ipykernel install --user --name=rag_env --display-name="RAG Techniques (Python)"
```

## 🐛 Solución de Problemas Comunes

### Error: "No module named 'xyz'"
```powershell
# Verificar que el entorno está activado
where python
# Debería mostrar la ruta del entorno virtual

# Reinstalar el paquete
uv pip install xyz
```

### Jupyter no encuentra el kernel
```powershell
# Reinstalar ipykernel
uv pip install --force-reinstall ipykernel
python -m ipykernel install --user --name=rag_env --display-name="RAG Techniques (Python)"
```

### UV no funciona correctamente
```powershell
# Volver a pip tradicional temporalmente
pip install nombre_paquete

# O reinstalar uv
pip uninstall uv
pip install uv
```

## 📁 Estructura de Archivos Recomendada

```
RAG_Techniques/
├── rag_env/                    # Entorno virtual (en .gitignore)
├── .env                        # Variables de entorno (en .gitignore)
├── .gitignore                  # Ignorar archivos sensibles
├── requirements.txt            # Dependencias del proyecto
├── CONFIGURACION_ENTORNO.md    # Este archivo
├── notebooks_traducidos/       # Tus notebooks en español
├── data/                       # Datos de ejemplo
└── scripts/                    # Scripts de utilidad
```

## 🔒 Seguridad

Asegúrate de que tu `.gitignore` incluya:
```gitignore
# Entorno virtual
rag_env/
venv/
env/

# Variables de entorno
.env
.env.local

# Archivos de Python
__pycache__/
*.pyc
*.pyo

# Jupyter
.ipynb_checkpoints/

# Datos sensibles
*.key
*.pem
```

## 🎯 Próximos Pasos

1. ✅ Configurar entorno virtual
2. ✅ Instalar dependencias
3. ✅ Configurar Jupyter kernel
4. 🔄 Comenzar a traducir notebooks
5. 🔄 Probar técnicas RAG
6. 🔄 Documentar ejemplos en español

---

**Nota**: Este entorno está optimizado para trabajar con las técnicas RAG del repositorio original, manteniendo compatibilidad y añadiendo mejoras para el desarrollo en español.