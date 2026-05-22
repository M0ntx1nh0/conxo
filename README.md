# Conxo Analytics

App local en Streamlit para analizar el rendimiento del Conxo a partir de `conxo.xlsx`.

## Arranque rápido

1. Crear entorno virtual.
2. Instalar dependencias con `requirements.txt`.
3. Lanzar la app con `streamlit run app.py`.

## Estructura inicial

- `General`: clasificación actual del Conxo, selector de jornada y ficha técnica del partido.
- `Equipo`: evolución de posición, diferencia de goles y producción ofensiva/defensiva por franjas de tiempo.
- `Plantilla`: tabla base de jugadores con convocatorias, titularidades, partidos, minutos y goles.

## Archivos necesarios para desplegar

Para que la app funcione correctamente en Streamlit deben subirse al repositorio:

- `app.py`
- `requirements.txt`
- `conxo.xlsx`
- carpeta `assets/`

No hace falta subir:

- `.venv/`
- `__pycache__/`
- `Plantilla Conxo.numbers`
- archivos temporales como `~$conxo.xlsx`

## Despliegue en Streamlit

1. Sube el proyecto a GitHub con los archivos necesarios.
2. Entra en Streamlit Community Cloud.
3. Crea una nueva app apuntando a este repositorio.
4. Selecciona:
   - rama: la que uses en GitHub
   - archivo principal: `app.py`
5. Lanza el despliegue.

La app leerá directamente `conxo.xlsx` y la carpeta `assets` desde el propio repositorio.
