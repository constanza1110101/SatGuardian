Contribuyendo a SatGuardian
¡Gracias por tu interés en contribuir a SatGuardian! Este documento proporciona las directrices para contribuir al proyecto de manera efectiva, asegurando que SatGuardian siga siendo una herramienta de alta calidad para la investigación en ciberseguridad satelital.

Consideraciones Éticas y Legales
Antes de contribuir, recuerda que este proyecto está destinado exclusivamente para uso educativo e investigación autorizada. Todas las contribuciones deben:

No facilitar la interceptación ilegal de comunicaciones privadas
No implementar capacidades para interferir activamente con sistemas satelitales
Mantener el enfoque educativo y de investigación en ciberseguridad defensiva
Respetar las regulaciones internacionales sobre comunicaciones espaciales
Entorno de Desarrollo
Prerrequisitos
Python 3.8+
Git
Dispositivo SDR compatible (para pruebas)
Dependencias listadas en requirements.txt
Configuración del Entorno
Clona el repositorio:

bash

Hide
git clone https://github.com/usuario/satguardian.git
cd satguardian
Crea y activa un entorno virtual:

bash

Hide
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
Instala las dependencias:

bash

Hide
pip install -r requirements.txt
Instala las dependencias de desarrollo:

bash

Hide
pip install -r requirements-dev.txt
Flujo de Trabajo de Contribución
1. Seleccionar o Crear un Issue
Antes de comenzar a trabajar en una contribución:

Revisa los issues existentes para ver si tu idea ya ha sido propuesta
Si encuentras un issue relacionado, comenta en él para expresar tu interés
Si es una nueva idea, crea un nuevo issue describiendo:
El problema o mejora propuesta
La solución que planeas implementar
Cualquier consideración técnica relevante
2. Bifurcar (Fork) y Clonar
Haz un fork del repositorio en GitHub
Clona tu fork localmente:
bash

Hide
git clone https://github.com/TU-USUARIO/satguardian.git
cd satguardian
3. Crear una Rama
Crea una rama específica para tu contribución:

bash

Hide
git checkout -b tipo/descripcion-breve
Utiliza prefijos descriptivos como:

feature/ para nuevas características
fix/ para correcciones de errores
docs/ para mejoras en la documentación
refactor/ para refactorizaciones de código
test/ para añadir o mejorar pruebas
4. Desarrollo
Durante el desarrollo:

Sigue las convenciones de código Python (PEP 8)
Escribe pruebas unitarias para tu código
Mantén la compatibilidad con las versiones de Python soportadas (3.8+)
Documenta las nuevas funcionalidades con docstrings
Asegúrate de que tu código funcione con los dispositivos SDR soportados
5. Pruebas
Antes de enviar tu contribución:

Ejecuta las pruebas unitarias:

bash

Hide
pytest
Verifica el estilo de código:

bash

Hide
flake8
Si tu contribución implica procesamiento de señales o nuevos algoritmos, incluye pruebas específicas y, si es posible, datos de muestra.

6. Enviar un Pull Request
Asegúrate de que tu rama esté actualizada con la rama principal:

bash

Hide
git fetch origin
git rebase origin/main
Empuja tus cambios a tu fork:

bash

Hide
git push origin tipo/descripcion-breve
Crea un Pull Request en GitHub

Describe detalladamente los cambios realizados

Referencia cualquier issue relacionado

Espera la revisión y responde a los comentarios

Convenciones de Código
Estilo de Código Python
Sigue PEP 8 para el estilo de código Python
Utiliza 4 espacios para la indentación (no tabulaciones)
Limita las líneas a 88 caracteres (compatible con Black)
Utiliza nombres descriptivos para variables y funciones
Docstrings
Utiliza el formato de Google para docstrings:

python

Hide
def funcion(param1, param2):
    """Breve descripción de la función.
    
    Descripción más detallada que explica el propósito
    y comportamiento de la función.
    
    Args:
        param1 (tipo): Descripción del parámetro.
        param2 (tipo): Descripción del parámetro.
        
    Returns:
        tipo: Descripción de lo que devuelve.
        
    Raises:
        ExcepcionTipo: Cuando y por qué se lanza esta excepción.
    """
Comentarios
Utiliza comentarios para explicar "por qué", no "qué" o "cómo"
Mantén los comentarios actualizados cuando cambies el código
Evita comentarios obvios o redundantes
Manejo de Errores
Utiliza manejo de excepciones apropiado
Registra errores con mensajes descriptivos
Evita capturar excepciones genéricas sin reaccionar adecuadamente
Estructura del Proyecto
Familiarízate con la estructura del proyecto antes de contribuir:

plaintext

Hide
satguardian/
├── satguardian.py        # Script principal y punto de entrada
├── modules/              # Módulos principales
│   ├── signal_processor.py  # Procesamiento de señales
│   ├── satellite_tracker.py # Seguimiento de satélites
│   ├── threat_detector.py   # Detección de amenazas
│   └── database.py          # Gestión de base de datos
├── ui/                   # Componentes de interfaz de usuario
│   ├── main_window.py       # Ventana principal
│   ├── spectrum_view.py     # Visualización de espectro
│   └── dialogs.py           # Diálogos adicionales
├── utils/                # Utilidades generales
│   ├── config.py            # Gestión de configuración
│   ├── sdr.py               # Abstracción de dispositivos SDR
│   └── tle.py               # Gestión de datos TLE
├── models/               # Modelos de IA
├── tests/                # Pruebas unitarias y de integración
└── data/                 # Datos y recursos
Áreas de Contribución
Estas son algunas áreas específicas donde las contribuciones son especialmente bienvenidas:

Procesamiento de Señales: Mejoras en algoritmos de detección y clasificación
Soporte de Hardware: Compatibilidad con más dispositivos SDR
Detección de Amenazas: Nuevos algoritmos para identificar señales anómalas
Modelos de IA: Mejoras en la clasificación de señales y detección de anomalías
Visualización: Mejoras en la representación de datos y usabilidad
Documentación: Tutoriales, ejemplos y mejoras en la documentación existente
Optimización: Mejoras de rendimiento para procesamiento en tiempo real
Pruebas: Ampliación de la cobertura de pruebas y casos de prueba
Directrices para Contribuciones Específicas
Nuevos Algoritmos de Procesamiento de Señales
Incluye referencias a papers o documentación técnica relevante
Proporciona métricas de rendimiento y comparaciones
Asegúrate de que funcionen con los dispositivos SDR soportados
Considera el rendimiento en hardware con recursos limitados
Modelos de Machine Learning
Documenta el proceso de entrenamiento y las métricas de evaluación
Proporciona scripts para reentrenar el modelo si es posible
Considera el tamaño del modelo y los requisitos de memoria
Asegúrate de que pueda ejecutarse en tiempo real
Mejoras en la Interfaz de Usuario
Mantén la coherencia con el diseño existente
Considera la accesibilidad y la experiencia de usuario
Prueba en diferentes resoluciones de pantalla
Documenta cualquier nueva característica de la interfaz
Proceso de Revisión
Después de enviar un pull request:

Las pruebas automatizadas se ejecutarán en CI
Al menos un mantenedor revisará tu código
Es posible que se soliciten cambios o aclaraciones
Una vez aprobado, tu código será fusionado en la rama principal
Comunicación
Para discusiones sobre el desarrollo:

Utiliza los issues de GitHub para preguntas específicas
Para discusiones más amplias, utiliza las Discusiones de GitHub
Mantén un tono respetuoso y constructivo en todas las comunicaciones
Reconocimiento
Los contribuyentes serán reconocidos en:

El archivo CONTRIBUTORS.md
Las notas de la versión
La documentación del proyecto cuando sea apropiado
Al contribuir a este proyecto, confirmas que tus contribuciones se ajustan a los propósitos educativos y de investigación del proyecto, y que no tienes la intención de facilitar actividades no autorizadas o ilegales.

Gracias por ayudar a mejorar SatGuardian y por tu compromiso con el desarrollo ético de herramientas de ciberseguridad.
