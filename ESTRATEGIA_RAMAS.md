# Estrategia de Gestión de Ramas - RAG Techniques ES

## Configuración de Repositorio

### Remotos
- `origin`: Repositorio original (NirDiamant/RAG_Techniques)
- `emi-rag`: Mi fork personal

### Ramas
- `main`: Rama principal del original (NO MODIFICAR)
- `desarrollo`: Mi rama de trabajo con traducciones y mejoras

## Flujo de Trabajo Recomendado

### 1. Mantener sincronizado con el original
```bash
git checkout main
git pull origin main
```

### 2. Actualizar tu rama de desarrollo
```bash
git checkout desarrollo
git merge main  # Solo si hay cambios importantes
```

### 3. Trabajar en tus cambios
```bash
git checkout desarrollo
# Hacer cambios...
git add .
git commit -m "Descripción de cambios"
```

### 4. Subir a tu fork
```bash
git push mifork desarrollo
```

## Archivos a NO Modificar

Para evitar conflictos con el repositorio original:

- `README.md` (mantener original)
- Estructura de carpetas principal
- Archivos de licencia originales

## Archivos Propios

Crear con prefijos/sufijos distintivos:
- `README_EMI.md`
- `*_es.ipynb` para notebooks traducidos
- Carpeta `traduciones/` para contenido específico

## Estrategia de Merge

1. **NUNCA** hacer merge directo de `desarrollo` a `main`
2. Mantener `main` limpio como referencia al original
3. Trabajar siempre en `desarrollo`
4. Si necesitas algo del original, hacer cherry-pick específico

## Resolución de Conflictos

Si hay conflictos:
1. Identificar si el conflicto afecta archivos originales
2. Si es así, priorizar la versión original
3. Mantener cambios solo en archivos propios

## Backup Strategy

Regularmente hacer backup de:
- Commits importantes: `git tag v1.0-es`
- Push frecuente a tu fork
- Mantener copias locales de archivos críticos