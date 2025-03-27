#!/usr/bin/env python3
# SatGuardian - Advanced Satellite Security Monitoring and Analysis System
# SOLO PARA USO EDUCATIVO E INVESTIGACIÓN AUTORIZADA
# Requiere: Python 3.8+, NumPy, SciPy, PySDR, Matplotlib, TensorFlow, PyQt5, Pandas

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtCore, QtWidgets, QtGui
import tensorflow as tf
import pandas as pd
import json
import sqlite3
import threading
import queue
import time
import datetime
import os
import sys
import argparse
import logging
import warnings
import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union, Any
import folium
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time
import ephem
import gpsd
import serial
from enum import Enum, auto
import configparser
import socket

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("satguardian.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SatGuardian")

# Constantes globales
VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = "config/satguardian.ini"
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
TLE_DIR = "tle"
DEFAULT_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"

# Tipos de satélites y señales
class SatelliteType(Enum):
    COMMUNICATIONS = auto()
    NAVIGATION = auto()
    WEATHER = auto()
    MILITARY = auto()
    EARTH_OBSERVATION = auto()
    EXPERIMENTAL = auto()
    UNKNOWN = auto()

class SignalType(Enum):
    TELEMETRY = auto()
    COMMAND = auto()
    PAYLOAD_DATA = auto()
    BEACON = auto()
    NAVIGATION = auto()
    UNKNOWN = auto()

class ModulationType(Enum):
    BPSK = auto()
    QPSK = auto()
    OQPSK = auto()
    MSK = auto()
    GMSK = auto()
    FSK = auto()
    QAM = auto()
    APSK = auto()
    UNKNOWN = auto()

class ThreatLevel(Enum):
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

@dataclass
class SatelliteInfo:
    norad_id: int
    name: str
    type: SatelliteType = SatelliteType.UNKNOWN
    country: str = "Unknown"
    launch_date: str = "Unknown"
    orbital_period: float = 0.0
    inclination: float = 0.0
    apogee: float = 0.0
    perigee: float = 0.0
    tle_line1: str = ""
    tle_line2: str = ""
    last_update: datetime.datetime = field(default_factory=datetime.datetime.now)
    known_frequencies: List[float] = field(default_factory=list)
    known_modulations: List[ModulationType] = field(default_factory=list)
    notes: str = ""

@dataclass
class Signal:
    id: str
    center_frequency: float
    bandwidth: float
    power: float
    timestamp: datetime.datetime
    satellite_id: Optional[int] = None
    modulation: ModulationType = ModulationType.UNKNOWN
    signal_type: SignalType = SignalType.UNKNOWN
    snr: float = 0.0
    duration: float = 0.0
    is_encrypted: bool = False
    is_anomalous: bool = False
    threat_level: ThreatLevel = ThreatLevel.NONE
    detection_confidence: float = 0.0
    raw_data_path: Optional[str] = None
    spectrum_path: Optional[str] = None
    notes: str = ""

class SatelliteDatabase:
    """Gestiona la base de datos de satélites y señales detectadas"""
    
    def __init__(self, db_path="data/satguardian.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._initialize_database()
        
    def _initialize_database(self):
        """Crea las tablas necesarias si no existen"""
        # Tabla de satélites
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS satellites (
            norad_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT,
            country TEXT,
            launch_date TEXT,
            orbital_period REAL,
            inclination REAL,
            apogee REAL,
            perigee REAL,
            tle_line1 TEXT,
            tle_line2 TEXT,
            last_update TEXT,
            known_frequencies TEXT,
            known_modulations TEXT,
            notes TEXT
        )
        ''')
        
        # Tabla de señales detectadas
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id TEXT PRIMARY KEY,
            center_frequency REAL,
            bandwidth REAL,
            power REAL,
            timestamp TEXT,
            satellite_id INTEGER,
            modulation TEXT,
            signal_type TEXT,
            snr REAL,
            duration REAL,
            is_encrypted INTEGER,
            is_anomalous INTEGER,
            threat_level TEXT,
            detection_confidence REAL,
            raw_data_path TEXT,
            spectrum_path TEXT,
            notes TEXT,
            FOREIGN KEY (satellite_id) REFERENCES satellites (norad_id)
        )
        ''')
        
        # Tabla de eventos de seguridad
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS security_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event_type TEXT,
            description TEXT,
            satellite_id INTEGER,
            signal_id TEXT,
            severity TEXT,
            is_resolved INTEGER,
            resolution_notes TEXT,
            FOREIGN KEY (satellite_id) REFERENCES satellites (norad_id),
            FOREIGN KEY (signal_id) REFERENCES signals (id)
        )
        ''')
        
        self.conn.commit()
    
    def add_satellite(self, satellite: SatelliteInfo) -> bool:
        """Añade o actualiza información de un satélite en la base de datos"""
        try:
            # Convertir listas a JSON para almacenamiento
            known_frequencies = json.dumps(satellite.known_frequencies)
            known_modulations = json.dumps([m.name for m in satellite.known_modulations])
            
            self.cursor.execute('''
            INSERT OR REPLACE INTO satellites 
            (norad_id, name, type, country, launch_date, orbital_period, inclination, 
             apogee, perigee, tle_line1, tle_line2, last_update, known_frequencies, 
             known_modulations, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                satellite.norad_id,
                satellite.name,
                satellite.type.name,
                satellite.country,
                satellite.launch_date,
                satellite.orbital_period,
                satellite.inclination,
                satellite.apogee,
                satellite.perigee,
                satellite.tle_line1,
                satellite.tle_line2,
                satellite.last_update.isoformat(),
                known_frequencies,
                known_modulations,
                satellite.notes
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error al añadir satélite a la base de datos: {e}")
            return False
    
    def get_satellite(self, norad_id: int) -> Optional[SatelliteInfo]:
        """Recupera información de un satélite por su ID de NORAD"""
        try:
            self.cursor.execute("SELECT * FROM satellites WHERE norad_id = ?", (norad_id,))
            row = self.cursor.fetchone()
            if not row:
                return None
                
            # Convertir de JSON a listas
            known_frequencies = json.loads(row[12]) if row[12] else []
            known_modulations = [ModulationType[m] for m in json.loads(row[13])] if row[13] else []
            
            return SatelliteInfo(
                norad_id=row[0],
                name=row[1],
                type=SatelliteType[row[2]] if row[2] else SatelliteType.UNKNOWN,
                country=row[3],
                launch_date=row[4],
                orbital_period=row[5],
                inclination=row[6],
                apogee=row[7],
                perigee=row[8],
                tle_line1=row[9],
                tle_line2=row[10],
                last_update=datetime.datetime.fromisoformat(row[11]) if row[11] else datetime.datetime.now(),
                known_frequencies=known_frequencies,
                known_modulations=known_modulations,
                notes=row[14]
            )
        except Exception as e:
            logger.error(f"Error al recuperar satélite de la base de datos: {e}")
            return None
    
    def add_signal(self, signal: Signal) -> bool:
        """Añade una señal detectada a la base de datos"""
        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO signals 
            (id, center_frequency, bandwidth, power, timestamp, satellite_id, modulation, 
             signal_type, snr, duration, is_encrypted, is_anomalous, threat_level, 
             detection_confidence, raw_data_path, spectrum_path, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.id,
                signal.center_frequency,
                signal.bandwidth,
                signal.power,
                signal.timestamp.isoformat(),
                signal.satellite_id,
                signal.modulation.name,
                signal.signal_type.name,
                signal.snr,
                signal.duration,
                1 if signal.is_encrypted else 0,
                1 if signal.is_anomalous else 0,
                signal.threat_level.name,
                signal.detection_confidence,
                signal.raw_data_path,
                signal.spectrum_path,
                signal.notes
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error al añadir señal a la base de datos: {e}")
            return False
    
    def log_security_event(self, timestamp: datetime.datetime, event_type: str, 
                          description: str, satellite_id: Optional[int] = None, 
                          signal_id: Optional[str] = None, 
                          severity: str = "MEDIUM", 
                          is_resolved: bool = False, 
                          resolution_notes: str = "") -> bool:
        """Registra un evento de seguridad en la base de datos"""
        try:
            self.cursor.execute('''
            INSERT INTO security_events 
            (timestamp, event_type, description, satellite_id, signal_id, severity, 
             is_resolved, resolution_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp.isoformat(),
                event_type,
                description,
                satellite_id,
                signal_id,
                severity,
                1 if is_resolved else 0,
                resolution_notes
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error al registrar evento de seguridad: {e}")
            return False
    
    def get_recent_signals(self, limit: int = 100) -> List[Signal]:
        """Recupera las señales más recientes"""
        try:
            self.cursor.execute(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", 
                (limit,)
            )
            rows = self.cursor.fetchall()
            
            signals = []
            for row in rows:
                signals.append(Signal(
                    id=row[0],
                    center_frequency=row[1],
                    bandwidth=row[2],
                    power=row[3],
                    timestamp=datetime.datetime.fromisoformat(row[4]),
                    satellite_id=row[5],
                    modulation=ModulationType[row[6]] if row[6] else ModulationType.UNKNOWN,
                    signal_type=SignalType[row[7]] if row[7] else SignalType.UNKNOWN,
                    snr=row[8],
                    duration=row[9],
                    is_encrypted=bool(row[10]),
                    is_anomalous=bool(row[11]),
                    threat_level=ThreatLevel[row[12]] if row[12] else ThreatLevel.NONE,
                    detection_confidence=row[13],
                    raw_data_path=row[14],
                    spectrum_path=row[15],
                    notes=row[16]
                ))
            
            return signals
        except Exception as e:
            logger.error(f"Error al recuperar señales recientes: {e}")
            return []
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        self.conn.close()

class TLEManager:
    """Gestiona la descarga y actualización de datos TLE (Two-Line Element) para satélites"""
    
    def __init__(self, tle_dir=TLE_DIR, tle_url=DEFAULT_TLE_URL):
        self.tle_dir = tle_dir
        self.tle_url = tle_url
        os.makedirs(tle_dir, exist_ok=True)
        self.satellites = {}
        self.last_update = None
    
    def update_tle_data(self, force=False) -> bool:
        """Descarga los datos TLE más recientes"""
        now = datetime.datetime.now()
        
        # Comprobar si es necesaria una actualización
        if not force and self.last_update and (now - self.last_update).days < 1:
            logger.info("Datos TLE ya actualizados recientemente")
            return True
            
        try:
            logger.info(f"Descargando datos TLE desde {self.tle_url}")
            response = requests.get(self.tle_url)
            response.raise_for_status()
            
            # Guardar los datos TLE
            tle_file = os.path.join(self.tle_dir, f"tle_{now.strftime('%Y%m%d')}.txt")
            with open(tle_file, 'w') as f:
                f.write(response.text)
            
            # Actualizar el enlace simbólico al archivo más reciente
            latest_link = os.path.join(self.tle_dir, "latest.txt")
            if os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(tle_file, latest_link)
            
            # Cargar los satélites
            self._load_satellites(tle_file)
            self.last_update = now
            
            logger.info(f"Datos TLE actualizados correctamente. {len(self.satellites)} satélites cargados.")
            return True
        except Exception as e:
            logger.error(f"Error al actualizar datos TLE: {e}")
            return False
    
    def _load_satellites(self, tle_file):
        """Carga los satélites desde un archivo TLE"""
        self.satellites = {}
        
        with open(tle_file, 'r') as f:
            lines = f.readlines()
            
        # Procesar los datos TLE (grupos de 3 líneas)
        for i in range(0, len(lines) - 2, 3):
            try:
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                
                # Extraer NORAD ID
                norad_id = int(line1[2:7])
                
                # Crear satélite usando skyfield
                satellite = EarthSatellite(line1, line2, name)
                
                # Almacenar en el diccionario
                self.satellites[norad_id] = {
                    'name': name,
                    'line1': line1,
                    'line2': line2,
                    'satellite': satellite
                }
            except Exception as e:
                logger.warning(f"Error al procesar TLE para satélite: {e}")
    
    def get_satellite_position(self, norad_id, timestamp=None) -> Optional[Tuple[float, float, float]]:
        """Obtiene la posición actual de un satélite (latitud, longitud, altitud)"""
        if not timestamp:
            timestamp = datetime.datetime.now()
            
        if norad_id not in self.satellites:
            logger.warning(f"Satélite con NORAD ID {norad_id} no encontrado en datos TLE")
            return None
            
        try:
            # Convertir timestamp a tiempo skyfield
            ts = load.timescale()
            t = ts.from_datetime(timestamp)
            
            # Obtener posición geocéntrica
            satellite = self.satellites[norad_id]['satellite']
            geocentric = satellite.at(t)
            
            # Convertir a coordenadas geográficas
            subpoint = wgs84.subpoint_of(geocentric)
            
            return (subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km)
        except Exception as e:
            logger.error(f"Error al calcular posición del satélite {norad_id}: {e}")
            return None
    
    def get_visible_satellites(self, lat, lon, alt=0, min_elevation=10) -> List[int]:
        """Obtiene lista de satélites visibles desde una ubicación dada"""
        visible = []
        
        if not self.satellites:
            logger.warning("No hay datos TLE cargados")
            return visible
            
        try:
            # Crear objeto observador
            observer = ephem.Observer()
            observer.lat = str(lat)
            observer.lon = str(lon)
            observer.elevation = alt
            observer.date = ephem.now()
            
            # Comprobar cada satélite
            for norad_id, sat_data in self.satellites.items():
                try:
                    # Crear objeto PyEphem
                    sat = ephem.readtle(sat_data['name'], sat_data['line1'], sat_data['line2'])
                    sat.compute(observer)
                    
                    # Comprobar si es visible (por encima del horizonte con elevación mínima)
                    elevation_deg = math.degrees(sat.alt)
                    if elevation_deg > min_elevation:
                        visible.append(norad_id)
                except Exception as e:
                    logger.debug(f"Error al calcular visibilidad para satélite {norad_id}: {e}")
            
            return visible
        except Exception as e:
            logger.error(f"Error al calcular satélites visibles: {e}")
            return []
    
    def export_satellite_info(self, norad_id) -> Optional[SatelliteInfo]:
        """Exporta información detallada de un satélite basada en TLE"""
        if norad_id not in self.satellites:
            logger.warning(f"Satélite con NORAD ID {norad_id} no encontrado en datos TLE")
            return None
            
        try:
            sat_data = self.satellites[norad_id]
            
            # Extraer información del TLE
            line1 = sat_data['line1']
            line2 = sat_data['line2']
            
            # Inclination (degrees)
            inclination = float(line2[8:16])
            
            # Orbital period calculation from mean motion
            mean_motion = float(line2[52:63])  # Revolutions per day
            orbital_period = 24.0 * 60.0 / mean_motion  # Period in minutes
            
            # Eccentricity
            eccentricity = float("0." + line2[26:33])
            
            # Semi-major axis (km)
            # Using formula: a = (mu/(n*2*pi/86400)^2)^(1/3)
            mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
            n = mean_motion * 2 * math.pi / 86400  # Mean motion in radians/second
            semi_major_axis = (mu / (n*n))**(1/3)
            
            # Apogee and perigee
            apogee = semi_major_axis * (1 + eccentricity) - 6378.137  # km above Earth's surface
            perigee = semi_major_axis * (1 - eccentricity) - 6378.137  # km above Earth's surface
            
            # Create SatelliteInfo object
            sat_info = SatelliteInfo(
                norad_id=norad_id,
                name=sat_data['name'],
                type=self._guess_satellite_type(sat_data['name']),
                country=self._guess_satellite_country(sat_data['name']),
                launch_date="Unknown",  # TLE doesn't contain launch date
                orbital_period=orbital_period,
                inclination=inclination,
                apogee=apogee,
                perigee=perigee,
                tle_line1=line1,
                tle_line2=line2,
                last_update=datetime.datetime.now()
            )
            
            return sat_info
        except Exception as e:
            logger.error(f"Error al exportar información del satélite {norad_id}: {e}")
            return None
    
    def _guess_satellite_type(self, name) -> SatelliteType:
        """Intenta adivinar el tipo de satélite basado en su nombre"""
        name_lower = name.lower()
        
        if any(term in name_lower for term in ['iridium', 'intelsat', 'inmarsat', 'globalstar', 'telesat', 'comms']):
            return SatelliteType.COMMUNICATIONS
        elif any(term in name_lower for term in ['gps', 'navstar', 'glonass', 'galileo', 'beidou', 'compass']):
            return SatelliteType.NAVIGATION
        elif any(term in name_lower for term in ['noaa', 'goes', 'meteosat', 'meteor', 'himawari']):
            return SatelliteType.WEATHER
        elif any(term in name_lower for term in ['nro', 'kh', 'keyhole', 'vela', 'dsp', 'milstar', 'sbirs']):
            return SatelliteType.MILITARY
        elif any(term in name_lower for term in ['landsat', 'worldview', 'geoeye', 'spot', 'sentinel']):
            return SatelliteType.EARTH_OBSERVATION
        elif any(term in name_lower for term in ['test', 'exp', 'demo', 'cubesat']):
            return SatelliteType.EXPERIMENTAL
        
        return SatelliteType.UNKNOWN
    
    def _guess_satellite_country(self, name) -> str:
        """Intenta adivinar el país de origen del satélite basado en su nombre"""
        name_lower = name.lower()
        
        if any(term in name_lower for term in ['usa', 'noaa', 'goes', 'gps', 'navstar', 'dmsp']):
            return "USA"
        elif any(term in name_lower for term in ['cosmos', 'glonass', 'meteor', 'elektro']):
            return "Russia"
        elif any(term in name_lower for term in ['beidou', 'fengyun', 'yaogan', 'gaofen']):
            return "China"
        elif any(term in name_lower for term in ['galileo', 'sentinel', 'meteosat']):
            return "EU"
        elif 'jaxa' in name_lower or 'himawari' in name_lower:
            return "Japan"
        elif 'isro' in name_lower or 'cartosat' in name_lower:
            return "India"
        
        return "Unknown"

class SignalProcessor:
    """Procesa señales de RF para detectar y analizar transmisiones satelitales"""
    
    def __init__(self, model_dir=MODELS_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Cargar modelos de ML
        self._load_models()
        
        # Inicializar parámetros
        self.sample_rate = 2.048e6  # Tasa de muestreo predeterminada
        self.fft_size = 4096        # Tamaño de FFT predeterminado
        
    def _load_models(self):
        """Carga los modelos de machine learning para clasificación de señales"""
        try:
            # Modelo de clasificación de modulación
            modulation_model_path = os.path.join(self.model_dir, "modulation_classifier.h5")
            if os.path.exists(modulation_model_path):
                self.modulation_model = tf.keras.models.load_model(modulation_model_path)
                logger.info("Modelo de clasificación de modulación cargado correctamente")
            else:
                logger.warning("Modelo de clasificación de modulación no encontrado")
                self.modulation_model = None
            
            # Modelo de detección de anomalías
            anomaly_model_path = os.path.join(self.model_dir, "anomaly_detector.h5")
            if os.path.exists(anomaly_model_path):
                self.anomaly_model = tf.keras.models.load_model(anomaly_model_path)
                logger.info("Modelo de detección de anomalías cargado correctamente")
            else:
                logger.warning("Modelo de detección de anomalías no encontrado")
                self.anomaly_model = None
                
            # Modelo de clasificación de tipo de señal
            signal_type_model_path = os.path.join(self.model_dir, "signal_type_classifier.h5")
            if os.path.exists(signal_type_model_path):
                self.signal_type_model = tf.keras.models.load_model(signal_type_model_path)
                logger.info("Modelo de clasificación de tipo de señal cargado correctamente")
            else:
                logger.warning("Modelo de clasificación de tipo de señal no encontrado")
                self.signal_type_model = None
                
        except Exception as e:
            logger.error(f"Error al cargar modelos de ML: {e}")
            self.modulation_model = None
            self.anomaly_model = None
            self.signal_type_model = None
    
    def process_signal(self, samples, center_freq, sample_rate=None) -> Signal:
        """Procesa una señal y extrae sus características"""
        if sample_rate is not None:
            self.sample_rate = sample_rate
            
        # Generar ID único para la señal
        signal_id = hashlib.md5(f"{center_freq}_{time.time()}".encode()).hexdigest()
        
        # Analizar espectro
        spectrum = self._compute_spectrum(samples)
        
        # Detectar características básicas
        power = self._estimate_power(samples)
        bandwidth = self._estimate_bandwidth(spectrum)
        snr = self._estimate_snr(spectrum)
        
        # Clasificar modulación
        modulation, mod_confidence = self._classify_modulation(samples)
        
        # Detectar si la señal está encriptada
        is_encrypted = self._detect_encryption(samples)
        
        # Clasificar tipo de señal
        signal_type = self._classify_signal_type(samples, spectrum)
        
        # Detectar anomalías
        is_anomalous = self._detect_anomalies(samples, spectrum)
        
        # Evaluar nivel de amenaza
        threat_level = self._evaluate_threat(is_encrypted, is_anomalous, modulation, signal_type)
        
        # Guardar datos en bruto y espectro si es necesario
        raw_data_path = None
        spectrum_path = None
        
        if is_anomalous or threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # Guardar datos para análisis posterior
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_data_path = os.path.join(DATA_DIR, f"signal_{signal_id}_{timestamp_str}.npy")
            spectrum_path = os.path.join(DATA_DIR, f"spectrum_{signal_id}_{timestamp_str}.npy")
            
            os.makedirs(DATA_DIR, exist_ok=True)
            np.save(raw_data_path, samples)
            np.save(spectrum_path, spectrum)
        
        # Crear objeto Signal
        signal = Signal(
            id=signal_id,
            center_frequency=center_freq,
            bandwidth=bandwidth,
            power=power,
            timestamp=datetime.datetime.now(),
            modulation=modulation,
            signal_type=signal_type,
            snr=snr,
            duration=len(samples) / self.sample_rate,
            is_encrypted=is_encrypted,
            is_anomalous=is_anomalous,
            threat_level=threat_level,
            detection_confidence=mod_confidence,
            raw_data_path=raw_data_path,
            spectrum_path=spectrum_path
        )
        
        return signal
    
    def _compute_spectrum(self, samples):
        """Calcula el espectro de potencia de las muestras"""
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples, n=self.fft_size)))**2
        return spectrum / np.max(spectrum)  # Normalizar
    
    def _estimate_power(self, samples):
        """Estima la potencia de la señal en dB"""
        return 10 * np.log10(np.mean(np.abs(samples)**2))
    
    def _estimate_bandwidth(self, spectrum):
        """Estima el ancho de banda de la señal en Hz"""
        # Usar método de -3dB
        max_val = np.max(spectrum)
        threshold = max_val / 2  # -3dB = factor de 1/2 en potencia
      # Encontrar índices donde el espectro cruza el umbral
        mask = spectrum > threshold
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return 0.0
            
        # Calcular ancho de banda
        lower_idx = indices[0]
        upper_idx = indices[-1]
        bandwidth = (upper_idx - lower_idx) * (self.sample_rate / self.fft_size)
        
        return bandwidth
    
    def _estimate_snr(self, spectrum):
        """Estima la relación señal-ruido en dB"""
        # Ordenar espectro
        sorted_spectrum = np.sort(spectrum)
        
        # Estimar nivel de ruido (mediana del 10% inferior)
        noise_floor = np.median(sorted_spectrum[:int(len(sorted_spectrum) * 0.1)])
        
        # Estimar nivel de señal (promedio del 1% superior)
        signal_level = np.mean(sorted_spectrum[-int(len(sorted_spectrum) * 0.01):])
        
        # Calcular SNR
        if noise_floor > 0:
            snr = 10 * np.log10(signal_level / noise_floor)
        else:
            snr = 100.0  # Valor arbitrario alto si el ruido es muy bajo
            
        return snr
    
    def _classify_modulation(self, samples):
        """Clasifica el tipo de modulación de la señal"""
        if self.modulation_model is None:
            # Clasificación básica basada en estadísticas si no hay modelo ML
            return self._basic_modulation_classification(samples)
        
        try:
            # Preprocesar muestras para el modelo
            features = self._extract_modulation_features(samples)
            features = np.expand_dims(features, axis=0)  # Añadir dimensión de batch
            
            # Realizar predicción
            prediction = self.modulation_model.predict(features)[0]
            
            # Obtener clase con mayor probabilidad
            class_idx = np.argmax(prediction)
            confidence = float(prediction[class_idx])
            
            # Mapear índice a tipo de modulación
            modulation_types = [
                ModulationType.BPSK,
                ModulationType.QPSK,
                ModulationType.OQPSK,
                ModulationType.MSK,
                ModulationType.GMSK,
                ModulationType.FSK,
                ModulationType.QAM,
                ModulationType.APSK
            ]
            
            if class_idx < len(modulation_types):
                return modulation_types[class_idx], confidence
            else:
                return ModulationType.UNKNOWN, 0.0
                
        except Exception as e:
            logger.error(f"Error en clasificación de modulación: {e}")
            return self._basic_modulation_classification(samples)
    
    def _basic_modulation_classification(self, samples):
        """Clasificación básica de modulación basada en estadísticas de la señal"""
        # Calcular características básicas
        amplitude = np.abs(samples)
        phase = np.angle(samples)
        
        # Calcular estadísticas
        amp_var = np.var(amplitude)
        phase_var = np.var(phase)
        amp_mean = np.mean(amplitude)
        
        # Normalizar varianza
        norm_amp_var = amp_var / (amp_mean ** 2) if amp_mean > 0 else 0
        
        # Decisión basada en características
        if norm_amp_var < 0.05 and phase_var > 2.0:
            return ModulationType.PSK, 0.7
        elif norm_amp_var > 0.5 and phase_var < 0.5:
            return ModulationType.ASK, 0.7
        elif 0.05 < norm_amp_var < 0.5 and 0.5 < phase_var < 2.0:
            return ModulationType.QAM, 0.6
        elif norm_amp_var < 0.05 and phase_var < 0.5:
            # Calcular derivada de fase para distinguir FM de PM
            phase_diff = np.diff(np.unwrap(phase))
            if np.var(phase_diff) > 0.1:
                return ModulationType.FSK, 0.6
            else:
                return ModulationType.BPSK, 0.5
        
        return ModulationType.UNKNOWN, 0.3
    
    def _extract_modulation_features(self, samples):
        """Extrae características para clasificación de modulación"""
        # Calcular características en dominio del tiempo
        amplitude = np.abs(samples)
        phase = np.angle(samples)
        
        # Características de amplitud
        amp_mean = np.mean(amplitude)
        amp_var = np.var(amplitude)
        amp_skew = np.mean(((amplitude - amp_mean) / np.std(amplitude)) ** 3) if np.std(amplitude) > 0 else 0
        
        # Características de fase
        phase_mean = np.mean(phase)
        phase_var = np.var(phase)
        phase_diff = np.diff(np.unwrap(phase))
        phase_diff_var = np.var(phase_diff)
        
        # Características de frecuencia
        spectrum = np.abs(np.fft.fft(samples))
        spectrum_peak = np.max(spectrum) / np.sum(spectrum)
        
        # Calcular momentos espectrales
        freqs = np.fft.fftfreq(len(spectrum), 1/self.sample_rate)
        spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0
        
        # Combinar características
        features = np.array([
            amp_mean, amp_var, amp_skew,
            phase_mean, phase_var, phase_diff_var,
            spectrum_peak, abs(spectral_centroid)
        ])
        
        # Normalizar
        features = (features - np.mean(features)) / np.std(features) if np.std(features) > 0 else features
        
        return features
    
    def _detect_encryption(self, samples):
        """Detecta si una señal parece estar encriptada"""
        # Calcular entropía de la señal
        hist, _ = np.histogram(np.real(samples), bins=50, density=True)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calcular autocorrelación
        autocorr = np.correlate(samples, samples, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalizar
        
        # Calcular estadísticas de autocorrelación
        autocorr_peak = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
        
        # Las señales encriptadas tienen alta entropía y baja autocorrelación
        if entropy > 4.5 and autocorr_peak < 0.3:
            return True
        
        return False
    
    def _classify_signal_type(self, samples, spectrum):
        """Clasifica el tipo de señal (telemetría, comando, etc.)"""
        if self.signal_type_model is None:
            # Clasificación básica si no hay modelo ML
            return self._basic_signal_type_classification(samples, spectrum)
        
        try:
            # Preprocesar muestras para el modelo
            features = self._extract_signal_type_features(samples, spectrum)
            features = np.expand_dims(features, axis=0)  # Añadir dimensión de batch
            
            # Realizar predicción
            prediction = self.signal_type_model.predict(features)[0]
            
            # Obtener clase con mayor probabilidad
            class_idx = np.argmax(prediction)
            
            # Mapear índice a tipo de señal
            signal_types = [
                SignalType.TELEMETRY,
                SignalType.COMMAND,
                SignalType.PAYLOAD_DATA,
                SignalType.BEACON,
                SignalType.NAVIGATION
            ]
            
            if class_idx < len(signal_types):
                return signal_types[class_idx]
            else:
                return SignalType.UNKNOWN
                
        except Exception as e:
            logger.error(f"Error en clasificación de tipo de señal: {e}")
            return self._basic_signal_type_classification(samples, spectrum)
    
    def _basic_signal_type_classification(self, samples, spectrum):
        """Clasificación básica del tipo de señal basada en características"""
        # Calcular periodicidad (para detectar balizas)
        autocorr = np.correlate(samples, samples, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalizar
        
        # Buscar picos periódicos en la autocorrelación
        peaks, _ = signal.find_peaks(autocorr, height=0.4, distance=100)
        
        if len(peaks) > 3:
            # Comprobar si los picos son equidistantes (balizas)
            intervals = np.diff(peaks)
            if np.std(intervals) / np.mean(intervals) < 0.1:
                return SignalType.BEACON
        
        # Analizar ancho de banda y forma espectral
        bandwidth = self._estimate_bandwidth(spectrum)
        
        if bandwidth < 10000:  # Señales estrechas suelen ser balizas o telemetría
            return SignalType.TELEMETRY
        elif bandwidth > 100000:  # Señales anchas suelen ser datos de carga útil
            return SignalType.PAYLOAD_DATA
        
        # Buscar patrones de navegación (señales muy estructuradas)
        # Esto requeriría análisis más específicos para cada sistema de navegación
        
        # Valor predeterminado
        return SignalType.UNKNOWN
    
    def _extract_signal_type_features(self, samples, spectrum):
        """Extrae características para clasificación del tipo de señal"""
        # Características temporales
        autocorr = np.correlate(samples, samples, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalizar
        
        # Buscar picos en la autocorrelación
        peaks, _ = signal.find_peaks(autocorr, height=0.3)
        peak_count = len(peaks)
        
        if peak_count > 0:
            peak_heights = autocorr[peaks]
            peak_mean = np.mean(peak_heights)
            peak_var = np.var(peak_heights)
            if peak_count > 1:
                peak_intervals = np.diff(peaks)
                interval_std = np.std(peak_intervals) / np.mean(peak_intervals) if np.mean(peak_intervals) > 0 else 0
            else:
                interval_std = 0
        else:
            peak_mean = 0
            peak_var = 0
            interval_std = 0
        
        # Características espectrales
        bandwidth = self._estimate_bandwidth(spectrum)
        spectrum_peak = np.max(spectrum)
        spectrum_mean = np.mean(spectrum)
        spectrum_var = np.var(spectrum)
        spectrum_skew = np.mean(((spectrum - spectrum_mean) / np.std(spectrum)) ** 3) if np.std(spectrum) > 0 else 0
        
        # Análisis tiempo-frecuencia
        # Calcular espectrograma
        f, t, Sxx = signal.spectrogram(samples, fs=self.sample_rate, nperseg=256)
        
        # Variabilidad temporal del espectro
        spectral_var = np.mean(np.var(Sxx, axis=1))
        
        # Combinar características
        features = np.array([
            peak_count, peak_mean, peak_var, interval_std,
            bandwidth / self.sample_rate, spectrum_peak, spectrum_var, spectrum_skew,
            spectral_var
        ])
        
        # Normalizar
        features = (features - np.mean(features)) / np.std(features) if np.std(features) > 0 else features
        
        return features
    
    def _detect_anomalies(self, samples, spectrum):
        """Detecta anomalías en la señal"""
        if self.anomaly_model is None:
            # Detección básica si no hay modelo ML
            return self._basic_anomaly_detection(samples, spectrum)
        
        try:
            # Preprocesar muestras para el modelo
            features = self._extract_anomaly_features(samples, spectrum)
            features = np.expand_dims(features, axis=0)  # Añadir dimensión de batch
            
            # Para modelos de autoencoder, reconstruir y calcular error
            reconstructed = self.anomaly_model.predict(features)
            mse = np.mean(np.square(features - reconstructed))
            
            # Umbral para considerar anomalía
            threshold = 0.15
            
            return mse > threshold
                
        except Exception as e:
            logger.error(f"Error en detección de anomalías: {e}")
            return self._basic_anomaly_detection(samples, spectrum)
    
    def _basic_anomaly_detection(self, samples, spectrum):
        """Detección básica de anomalías basada en estadísticas"""
        # Calcular características
        amplitude = np.abs(samples)
        phase = np.angle(samples)
        
        # Estadísticas de amplitud
        amp_mean = np.mean(amplitude)
        amp_var = np.var(amplitude)
        amp_skew = np.mean(((amplitude - amp_mean) / np.std(amplitude)) ** 3) if np.std(amplitude) > 0 else 0
        amp_kurtosis = np.mean(((amplitude - amp_mean) / np.std(amplitude)) ** 4) if np.std(amplitude) > 0 else 0
        
        # Estadísticas de fase
        phase_var = np.var(phase)
        phase_diff = np.diff(np.unwrap(phase))
        phase_diff_var = np.var(phase_diff)
        
        # Detectar anomalías basadas en estadísticas
        # Valores extremos pueden indicar señales anómalas
        if amp_kurtosis > 10 or abs(amp_skew) > 2:
            return True
        
        # Cambios abruptos en la fase
        if phase_diff_var > 5.0:
            return True
        
        # Analizar espectro para detectar anomalías
        # Picos inusuales o formas espectrales extrañas
        spectrum_sorted = np.sort(spectrum)
        spectrum_ratio = spectrum_sorted[-1] / np.mean(spectrum_sorted[:-1]) if np.mean(spectrum_sorted[:-1]) > 0 else 0
        
        if spectrum_ratio > 100:  # Pico extremadamente alto
            return True
        
        return False
    
    def _extract_anomaly_features(self, samples, spectrum):
        """Extrae características para detección de anomalías"""
        # Combinar características temporales y espectrales
        amplitude = np.abs(samples)
        phase = np.angle(samples)
        
        # Estadísticas de amplitud
        amp_mean = np.mean(amplitude)
        amp_var = np.var(amplitude)
        amp_skew = np.mean(((amplitude - amp_mean) / np.std(amplitude)) ** 3) if np.std(amplitude) > 0 else 0
        amp_kurtosis = np.mean(((amplitude - amp_mean) / np.std(amplitude)) ** 4) if np.std(amplitude) > 0 else 0
        
        # Estadísticas de fase
        phase_mean = np.mean(phase)
        phase_var = np.var(phase)
        phase_diff = np.diff(np.unwrap(phase))
        phase_diff_var = np.var(phase_diff)
        
        # Estadísticas espectrales
        spectrum_mean = np.mean(spectrum)
        spectrum_var = np.var(spectrum)
        spectrum_skew = np.mean(((spectrum - spectrum_mean) / np.std(spectrum)) ** 3) if np.std(spectrum) > 0 else 0
        
        # Entropía espectral
        spectrum_norm = spectrum / np.sum(spectrum) if np.sum(spectrum) > 0 else spectrum
        entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10))
        
        # Características de autocorrelación
        autocorr = np.correlate(samples, samples, mode='full')
        autocorr = autocorr[len(autocorr)//2:len(autocorr)//2+100]
        autocorr = autocorr / autocorr[0]  # Normalizar
        autocorr_decay = np.mean(np.abs(autocorr[1:]))
        
        # Combinar características
        features = np.array([
            amp_mean, amp_var, amp_skew, amp_kurtosis,
            phase_mean, phase_var, phase_diff_var,
            spectrum_mean, spectrum_var, spectrum_skew,
            entropy, autocorr_decay
        ])
        
        # Normalizar
        features = (features - np.mean(features)) / np.std(features) if np.std(features) > 0 else features
        
        return features
    
    def _evaluate_threat(self, is_encrypted, is_anomalous, modulation, signal_type):
        """Evalúa el nivel de amenaza basado en características de la señal"""
        # Iniciar con nivel bajo
        threat_level = ThreatLevel.LOW
        
        # Señales encriptadas aumentan el nivel de amenaza
        if is_encrypted:
            threat_level = ThreatLevel.MEDIUM
        
        # Señales anómalas aumentan aún más el nivel
        if is_anomalous:
            if threat_level == ThreatLevel.MEDIUM:
                threat_level = ThreatLevel.HIGH
            else:
                threat_level = ThreatLevel.MEDIUM
        
        # Combinaciones específicas de características
        if is_encrypted and is_anomalous:
            threat_level = ThreatLevel.HIGH
        
        # Tipos de señal específicos
        if signal_type == SignalType.COMMAND and is_encrypted:
            threat_level = ThreatLevel.HIGH
        
        # Modulaciones militares conocidas
        if modulation in [ModulationType.GMSK, ModulationType.APSK] and is_encrypted:
            threat_level = ThreatLevel.HIGH
        
        # Nivel crítico solo en casos extremos
        if is_encrypted and is_anomalous and signal_type == SignalType.COMMAND:
            threat_level = ThreatLevel.CRITICAL
        
        return threat_level

class SatelliteTracker:
    """Gestiona el seguimiento de satélites y correlación con señales"""
    
    def __init__(self, tle_manager, db):
        self.tle_manager = tle_manager
        self.db = db
        self.current_location = None
        self.visible_satellites = []
        self.last_location_update = None
        
        # Intentar inicializar GPS
        self.gps_available = self._init_gps()
    
    def _init_gps(self):
        """Inicializa la conexión GPS si está disponible"""
        try:
            gpsd.connect()
            packet = gpsd.get_current()
            logger.info(f"GPS inicializado. Posición actual: {packet.position()}")
            return True
        except Exception as e:
            logger.warning(f"GPS no disponible: {e}")
            return False
    
    def update_location(self, lat=None, lon=None, alt=None):
        """Actualiza la ubicación actual del sistema"""
        if lat is not None and lon is not None:
            # Usar coordenadas proporcionadas
            self.current_location = (lat, lon, alt if alt is not None else 0)
            self.last_location_update = datetime.datetime.now()
            logger.info(f"Ubicación actualizada manualmente: {self.current_location}")
            return True
            
        elif self.gps_available:
            try:
                # Obtener ubicación del GPS
                packet = gpsd.get_current()
                if packet.mode >= 2:  # 2D o 3D fix
                    lat, lon = packet.position()
                    alt = packet.altitude() if packet.mode >= 3 else 0
                    self.current_location = (lat, lon, alt)
                    self.last_location_update = datetime.datetime.now()
                    logger.info(f"Ubicación actualizada desde GPS: {self.current_location}")
                    return True
                else:
                    logger.warning("GPS no tiene fix válido")
                    return False
            except Exception as e:
                logger.error(f"Error al actualizar ubicación desde GPS: {e}")
                return False
        else:
            logger.warning("No se puede actualizar ubicación: GPS no disponible y no se proporcionaron coordenadas")
            return False
    
    def update_visible_satellites(self):
        """Actualiza la lista de satélites visibles desde la ubicación actual"""
        if not self.current_location:
            logger.warning("No se puede actualizar satélites visibles: ubicación desconocida")
            return False
            
        # Comprobar si los datos TLE están cargados
        if not self.tle_manager.satellites:
            logger.warning("No hay datos TLE cargados")
            return False
            
        # Obtener satélites visibles
        try:
            lat, lon, alt = self.current_location
            self.visible_satellites = self.tle_manager.get_visible_satellites(lat, lon, alt)
            logger.info(f"Satélites visibles actualizados: {len(self.visible_satellites)} satélites")
            return True
        except Exception as e:
            logger.error(f"Error al actualizar satélites visibles: {e}")
            return False
    
    def correlate_signal(self, signal: Signal) -> Optional[int]:
        """Correlaciona una señal detectada con un satélite visible"""
        if not self.visible_satellites:
            logger.warning("No hay satélites visibles para correlacionar")
            return None
            
        best_match = None
        best_score = 0
        
        for norad_id in self.visible_satellites:
            # Obtener información del satélite
            satellite = self.db.get_satellite(norad_id)
            if not satellite:
                # Intentar obtener información básica del TLE
                satellite = self.tle_manager.export_satellite_info(norad_id)
                if satellite:
                    self.db.add_satellite(satellite)
                else:
                    continue
            
            # Calcular puntuación de correlación
            score = self._calculate_correlation_score(signal, satellite)
            
            if score > best_score and score > 0.6:  # Umbral mínimo
                best_match = norad_id
                best_score = score
        
        if best_match:
            logger.info(f"Señal correlacionada con satélite NORAD ID {best_match} (puntuación: {best_score:.2f})")
            
            # Actualizar señal con ID de satélite
            signal.satellite_id = best_match
            
            return best_match
        else:
            logger.info("No se encontró correlación con ningún satélite visible")
            return None
    
    def _calculate_correlation_score(self, signal: Signal, satellite: SatelliteInfo) -> float:
        """Calcula una puntuación de correlación entre una señal y un satélite"""
        score = 0.0
        
        # Comprobar si la frecuencia coincide con frecuencias conocidas
        if satellite.known_frequencies:
            for freq in satellite.known_frequencies:
                # Permitir cierto margen de error en la frecuencia
                if abs(signal.center_frequency - freq) < 100000:  # 100 kHz
                    score += 0.6
                    break
                elif abs(signal.center_frequency - freq) < 1000000:  # 1 MHz
                    score += 0.3
                    break
        
        # Comprobar si la modulación coincide
        if satellite.known_modulations and signal.modulation in satellite.known_modulations:
            score += 0.2
        
        # Ajustar puntuación según el tipo de satélite y señal
        if satellite.type == SatelliteType.COMMUNICATIONS and signal.signal_type in [SignalType.PAYLOAD_DATA, SignalType.TELEMETRY]:
            score += 0.1
        elif satellite.type == SatelliteType.NAVIGATION and signal.signal_type == SignalType.NAVIGATION:
            score += 0.2
        elif satellite.type == SatelliteType.WEATHER and signal.signal_type == SignalType.PAYLOAD_DATA:
            score += 0.1
        
        # Normalizar puntuación
        return min(score, 1.0)
    
    def generate_satellite_map(self, output_file="satellite_map.html"):
        """Genera un mapa con los satélites visibles y sus trayectorias"""
        if not self.current_location:
            logger.warning("No se puede generar mapa: ubicación desconocida")
            return False
            
        if not self.visible_satellites:
            logger.warning("No hay satélites visibles para mostrar en el mapa")
            return False
            
        try:
            # Crear mapa centrado en la ubicación actual
            lat, lon, _ = self.current_location
            m = folium.Map(location=[lat, lon], zoom_start=4)
            
            # Añadir marcador para la ubicación actual
            folium.Marker(
                location=[lat, lon],
                popup="Ubicación actual",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
            
            # Añadir satélites visibles
            for norad_id in self.visible_satellites:
                # Obtener posición actual
                pos = self.tle_manager.get_satellite_position(norad_id)
                if not pos:
                    continue
                    
                sat_lat, sat_lon, sat_alt = pos
                
                # Obtener información del satélite
                satellite = self.db.get_satellite(norad_id)
                if not satellite:
                    satellite = self.tle_manager.export_satellite_info(norad_id)
                
                name = satellite.name if satellite else f"NORAD ID {norad_id}"
                
                # Añadir marcador para el satélite
                popup_html = f"""
                <b>{name}</b><br>
                NORAD ID: {norad_id}<br>
                Altitud: {sat_alt:.1f} km<br>
                """
                
                if satellite:
                    popup_html += f"""
                    Tipo: {satellite.type.name}<br>
                    País: {satellite.country}<br>
                    """
                
                folium.Marker(
                    location=[sat_lat, sat_lon],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color="red", icon="satellite")
                ).add_to(m)
                
                # Calcular y añadir trayectoria futura
                trajectory = self._calculate_satellite_trajectory(norad_id)
                if trajectory:
                    folium.PolyLine(
                        trajectory,
                        color="red",
                        weight=2,
                        opacity=0.7,
                        dash_array="5,5"
                    ).add_to(m)
            
            # Guardar mapa
            m.save(output_file)
            logger.info(f"Mapa de satélites generado: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error al generar mapa de satélites: {e}")
            return False
    
    def _calculate_satellite_trajectory(self, norad_id, points=20, interval_minutes=5):
        """Calcula la trayectoria futura de un satélite"""
        trajectory = []
        
        try:
            current_time = datetime.datetime.now()
            
            for i in range(points):
                # Calcular tiempo futuro
                future_time = current_time + datetime.timedelta(minutes=i*interval_minutes)
                
                # Obtener posición en ese tiempo
                pos = self.tle_manager.get_satellite_position(norad_id, future_time)
                if pos:
                    sat_lat, sat_lon, _ = pos
                    trajectory.append([sat_lat, sat_lon])
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Error al calcular trayectoria del satélite {norad_id}: {e}")
            return []

class SatGuardianApp(QtWidgets.QMainWindow):
    """Interfaz gráfica principal de la aplicación SatGuardian"""
    
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        super().__init__()
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Inicializar componentes
        self.db = SatelliteDatabase()
        self.tle_manager = TLEManager()
        self.signal_processor = SignalProcessor()
        self.satellite_tracker = SatelliteTracker(self.tle_manager, self.db)
        
        # Actualizar datos TLE
        self.tle_manager.update_tle_data()
        
        # Configurar interfaz gráfica
        self._setup_ui()
        
        # Cola de mensajes para comunicación entre hilos
        self.message_queue = queue.Queue()
        
        # Iniciar temporizador para actualizar UI
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self._process_queue)
        self.update_timer.start(100)  # 100 ms
        
        # Estado de monitoreo
        self.monitoring = False
        self.monitoring_thread = None
        
        # Actualizar ubicación
        self._update_location()
    
    def _load_config(self):
        """Carga la configuración desde archivo INI"""
        config = configparser.ConfigParser()
        
        # Valores predeterminados
        config['General'] = {
            'DataDirectory': DATA_DIR,
            'LogLevel': 'INFO'
        }
        
        config['SDR'] = {
            'Device': 'rtlsdr',
            'SampleRate': '2048000',
            'Gain': '30'
        }
        
        config['Frequencies'] = {
            'DefaultStart': '137000000',
            'DefaultEnd': '138000000',
            'Presets': 'NOAA:137100000,Meteor:137900000,
           }
        
        config['Location'] = {
            'UseGPS': 'True',
            'DefaultLatitude': '0.0',
            'DefaultLongitude': '0.0',
            'DefaultAltitude': '0.0'
        }
        
        config['Security'] = {
            'EnableAnomalyDetection': 'True',
            'AlertOnEncryptedSignals': 'True',
            'AlertThreshold': 'MEDIUM'
        }
        
        # Cargar configuración existente si existe
        if os.path.exists(config_path):
            config.read(config_path)
        else:
            # Crear directorios si no existen
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Guardar configuración predeterminada
            with open(config_path, 'w') as f:
                config.write(f)
        
        return config
    
    def _setup_ui(self):
        """Configura la interfaz gráfica de usuario"""
        self.setWindowTitle(f"SatGuardian - Sistema de Monitoreo y Análisis de Seguridad Satelital v{VERSION}")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Barra de herramientas
        toolbar = QtWidgets.QToolBar("Herramientas")
        self.addToolBar(toolbar)
        
        # Acciones
        start_action = QtWidgets.QAction("Iniciar Monitoreo", self)
        start_action.triggered.connect(self.start_monitoring)
        toolbar.addAction(start_action)
        
        stop_action = QtWidgets.QAction("Detener Monitoreo", self)
        stop_action.triggered.connect(self.stop_monitoring)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        update_tle_action = QtWidgets.QAction("Actualizar TLE", self)
        update_tle_action.triggered.connect(self.update_tle_data)
        toolbar.addAction(update_tle_action)
        
        update_location_action = QtWidgets.QAction("Actualizar Ubicación", self)
        update_location_action.triggered.connect(self._update_location)
        toolbar.addAction(update_location_action)
        
        generate_map_action = QtWidgets.QAction("Generar Mapa", self)
        generate_map_action.triggered.connect(self.generate_satellite_map)
        toolbar.addAction(generate_map_action)
        
        toolbar.addSeparator()
        
        export_action = QtWidgets.QAction("Exportar Datos", self)
        export_action.triggered.connect(self.export_data)
        toolbar.addAction(export_action)
        
        # Layout de pestañas
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Pestaña de monitoreo en tiempo real
        monitoring_tab = QtWidgets.QWidget()
        monitoring_layout = QtWidgets.QVBoxLayout(monitoring_tab)
        
        # Panel superior para espectro
        spectrum_groupbox = QtWidgets.QGroupBox("Espectro en Tiempo Real")
        spectrum_layout = QtWidgets.QVBoxLayout(spectrum_groupbox)
        
        self.spectrum_figure = plt.figure(figsize=(8, 4))
        self.spectrum_canvas = FigureCanvasQTAgg(self.spectrum_figure)
        spectrum_layout.addWidget(self.spectrum_canvas)
        
        monitoring_layout.addWidget(spectrum_groupbox)
        
        # Panel inferior para espectrograma
        spectrogram_groupbox = QtWidgets.QGroupBox("Espectrograma")
        spectrogram_layout = QtWidgets.QVBoxLayout(spectrogram_groupbox)
        
        self.spectrogram_figure = plt.figure(figsize=(8, 4))
        self.spectrogram_canvas = FigureCanvasQTAgg(self.spectrogram_figure)
        spectrogram_layout.addWidget(self.spectrogram_canvas)
        
        monitoring_layout.addWidget(spectrogram_groupbox)
        
        # Añadir pestaña de monitoreo
        self.tab_widget.addTab(monitoring_tab, "Monitoreo en Tiempo Real")
        
        # Pestaña de señales detectadas
        signals_tab = QtWidgets.QWidget()
        signals_layout = QtWidgets.QVBoxLayout(signals_tab)
        
        # Tabla de señales
        self.signals_table = QtWidgets.QTableWidget()
        self.signals_table.setColumnCount(8)
        self.signals_table.setHorizontalHeaderLabels([
            "ID", "Frecuencia (MHz)", "Ancho de banda (kHz)", "Potencia (dB)", 
            "Modulación", "Tipo", "Satélite", "Amenaza"
        ])
        self.signals_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        signals_layout.addWidget(self.signals_table)
        
        # Añadir pestaña de señales
        self.tab_widget.addTab(signals_tab, "Señales Detectadas")
        
        # Pestaña de satélites
        satellites_tab = QtWidgets.QWidget()
        satellites_layout = QtWidgets.QVBoxLayout(satellites_tab)
        
        # Tabla de satélites
        self.satellites_table = QtWidgets.QTableWidget()
        self.satellites_table.setColumnCount(6)
        self.satellites_table.setHorizontalHeaderLabels([
            "NORAD ID", "Nombre", "Tipo", "País", "Visible", "Última Señal"
        ])
        self.satellites_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        satellites_layout.addWidget(self.satellites_table)
        
        # Añadir pestaña de satélites
        self.tab_widget.addTab(satellites_tab, "Satélites")
        
        # Pestaña de alertas
        alerts_tab = QtWidgets.QWidget()
        alerts_layout = QtWidgets.QVBoxLayout(alerts_tab)
        
        # Lista de alertas
        self.alerts_list = QtWidgets.QListWidget()
        alerts_layout.addWidget(self.alerts_list)
        
        # Añadir pestaña de alertas
        self.tab_widget.addTab(alerts_tab, "Alertas de Seguridad")
        
        # Barra de estado
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Etiquetas de información en la barra de estado
        self.location_label = QtWidgets.QLabel("Ubicación: Desconocida")
        self.statusBar.addPermanentWidget(self.location_label)
        
        self.satellites_label = QtWidgets.QLabel("Satélites visibles: 0")
        self.statusBar.addPermanentWidget(self.satellites_label)
        
        self.status_label = QtWidgets.QLabel("Estado: Inactivo")
        self.statusBar.addWidget(self.status_label)
        
        # Inicializar tablas
        self._update_signals_table()
        self._update_satellites_table()
    
    def _update_location(self):
        """Actualiza la ubicación actual"""
        use_gps = self.config.getboolean('Location', 'UseGPS', fallback=True)
        
        if use_gps:
            success = self.satellite_tracker.update_location()
            if not success:
                # Usar ubicación predeterminada si GPS falla
                lat = self.config.getfloat('Location', 'DefaultLatitude', fallback=0.0)
                lon = self.config.getfloat('Location', 'DefaultLongitude', fallback=0.0)
                alt = self.config.getfloat('Location', 'DefaultAltitude', fallback=0.0)
                self.satellite_tracker.update_location(lat, lon, alt)
        else:
            # Usar ubicación predeterminada
            lat = self.config.getfloat('Location', 'DefaultLatitude', fallback=0.0)
            lon = self.config.getfloat('Location', 'DefaultLongitude', fallback=0.0)
            alt = self.config.getfloat('Location', 'DefaultAltitude', fallback=0.0)
            self.satellite_tracker.update_location(lat, lon, alt)
        
        # Actualizar satélites visibles
        self.satellite_tracker.update_visible_satellites()
        
        # Actualizar UI
        if self.satellite_tracker.current_location:
            lat, lon, alt = self.satellite_tracker.current_location
            self.location_label.setText(f"Ubicación: {lat:.4f}, {lon:.4f}, {alt:.1f}m")
        
        self.satellites_label.setText(f"Satélites visibles: {len(self.satellite_tracker.visible_satellites)}")
        
        # Actualizar tabla de satélites
        self._update_satellites_table()
    
    def update_tle_data(self):
        """Actualiza los datos TLE desde la fuente configurada"""
        self.statusBar.showMessage("Actualizando datos TLE...")
        
        def update_thread():
            success = self.tle_manager.update_tle_data(force=True)
            
            if success:
                # Actualizar satélites visibles
                self.satellite_tracker.update_visible_satellites()
                
                # Actualizar UI desde el hilo principal
                self.message_queue.put(("status", "Datos TLE actualizados correctamente"))
                self.message_queue.put(("update_satellites", None))
            else:
                self.message_queue.put(("status", "Error al actualizar datos TLE"))
        
        # Iniciar en un hilo separado para no bloquear la UI
        threading.Thread(target=update_thread, daemon=True).start()
    
    def generate_satellite_map(self):
        """Genera un mapa con los satélites visibles"""
        self.statusBar.showMessage("Generando mapa de satélites...")
        
        def map_thread():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(REPORTS_DIR, f"satellite_map_{timestamp}.html")
            
            # Asegurar que el directorio existe
            os.makedirs(REPORTS_DIR, exist_ok=True)
            
            success = self.satellite_tracker.generate_satellite_map(output_file)
            
            if success:
                self.message_queue.put(("status", f"Mapa generado: {output_file}"))
                
                # Abrir el mapa en el navegador
                webbrowser.open(f"file://{os.path.abspath(output_file)}")
            else:
                self.message_queue.put(("status", "Error al generar mapa de satélites"))
        
        # Iniciar en un hilo separado
        threading.Thread(target=map_thread, daemon=True).start()
    
    def start_monitoring(self):
        """Inicia el monitoreo de señales satelitales"""
        if self.monitoring:
            self.statusBar.showMessage("El monitoreo ya está en curso")
            return
        
        # Obtener parámetros de configuración
        device = self.config.get('SDR', 'Device', fallback='rtlsdr')
        sample_rate = self.config.getint('SDR', 'SampleRate', fallback=2048000)
        gain = self.config.getint('SDR', 'Gain', fallback=30)
        
        # Frecuencia inicial
        start_freq = self.config.getint('Frequencies', 'DefaultStart', fallback=137000000)
        
        self.statusBar.showMessage(f"Iniciando monitoreo en {start_freq/1e6} MHz...")
        self.status_label.setText("Estado: Monitoreando")
        
        # Limpiar gráficos
        self._clear_plots()
        
        # Iniciar monitoreo en un hilo separado
        self.monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_thread,
            args=(device, sample_rate, gain, start_freq),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _monitoring_thread(self, device, sample_rate, gain, center_freq):
        """Función principal del hilo de monitoreo"""
        try:
            # Inicializar SDR (simulado para este ejemplo)
            # En una implementación real, se conectaría con el dispositivo SDR
            logger.info(f"Iniciando monitoreo con {device} en {center_freq/1e6} MHz")
            
            # Historial para espectrograma
            spectrogram_history = []
            
            # Bucle principal de monitoreo
            while self.monitoring:
                # Simular adquisición de muestras
                # En una implementación real, se obtendrían muestras del SDR
                samples = self._simulate_signal(center_freq, sample_rate)
                
                # Procesar señal
                signal = self.signal_processor.process_signal(samples, center_freq, sample_rate)
                
                # Correlacionar con satélite
                if signal.power > -50:  # Umbral de detección
                    satellite_id = self.satellite_tracker.correlate_signal(signal)
                    
                    # Guardar señal en la base de datos
                    self.db.add_signal(signal)
                    
                    # Generar alerta si es necesario
                    if signal.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                        self._generate_security_alert(signal)
                    
                    # Actualizar UI
                    self.message_queue.put(("new_signal", signal))
                
                # Calcular espectro para visualización
                spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples, n=4096)))**2
                freq_range = np.linspace(center_freq - sample_rate/2, 
                                        center_freq + sample_rate/2, 
                                        len(spectrum)) / 1e6
                
                # Actualizar espectrograma
                spectrogram_history.append(10 * np.log10(spectrum + 1e-10))
                if len(spectrogram_history) > 100:
                    spectrogram_history.pop(0)
                
                # Actualizar gráficos
                self.message_queue.put(("update_spectrum", (freq_range, spectrum)))
                self.message_queue.put(("update_spectrogram", (freq_range, spectrogram_history)))
                
                # Pausa breve
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error en hilo de monitoreo: {e}")
            self.message_queue.put(("status", f"Error en monitoreo: {e}"))
        finally:
            self.message_queue.put(("monitoring_stopped", None))
    
    def _simulate_signal(self, center_freq, sample_rate):
        """Simula una señal para pruebas (en una implementación real, se obtendrían del SDR)"""
        # Generar ruido base
        samples = np.random.normal(0, 0.1, 8192) + 1j * np.random.normal(0, 0.1, 8192)
        
        # Añadir señales simuladas ocasionalmente
        if np.random.random() < 0.3:  # 30% de probabilidad
            # Simular una señal
            signal_freq_offset = np.random.uniform(-sample_rate/4, sample_rate/4)
            signal_type = np.random.choice(["satellite", "terrestrial", "noise"])
            
            t = np.arange(len(samples)) / sample_rate
            
            if signal_type == "satellite":
                # Simular señal satelital (más débil, banda estrecha)
                signal_power = np.random.uniform(0.2, 0.5)
                signal_bw = np.random.uniform(10000, 50000)
                
                # Añadir modulación
                mod_type = np.random.choice(["bpsk", "qpsk", "gmsk"])
                
                if mod_type == "bpsk":
                    # BPSK
                    symbol_rate = signal_bw / 2
                    symbols = np.random.choice([-1, 1], size=int(len(samples) * symbol_rate / sample_rate))
                    symbol_samples = np.repeat(symbols, int(sample_rate / symbol_rate))
                    signal = symbol_samples[:len(samples)] * np.exp(2j * np.pi * signal_freq_offset * t)
                elif mod_type == "qpsk":
                    # QPSK
                    symbol_rate = signal_bw / 2
                    symbols = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], size=int(len(samples) * symbol_rate / sample_rate))
                    symbol_samples = np.repeat(symbols, int(sample_rate / symbol_rate))
                    signal = symbol_samples[:len(samples)] * np.exp(2j * np.pi * signal_freq_offset * t)
                else:
                    # GMSK (simulado)
                    signal = np.exp(2j * np.pi * (signal_freq_offset * t + 0.1 * np.cumsum(np.random.choice([-1, 1], size=len(samples)))))
                
                # Añadir a las muestras con potencia ajustada
                samples += signal_power * signal
                
                # Ocasionalmente añadir encriptación simulada (alta entropía)
                if np.random.random() < 0.2:
                    # Simular señal encriptada
                    random_phase = 2 * np.pi * np.random.random(len(samples))
                    encrypted_signal = 0.3 * np.exp(1j * random_phase) * np.exp(2j * np.pi * signal_freq_offset * t)
                    samples += encrypted_signal
                
            elif signal_type == "terrestrial":
                # Simular señal terrestre (más fuerte)
                signal_power = np.random.uniform(0.5, 1.5)
                signal_bw = np.random.uniform(5000, 200000)
                
                # FM o AM
                if np.random.random() < 0.5:
                    # FM
                    mod_signal = 0.1 * np.cumsum(np.random.normal(0, 1, len(samples)))
                    signal = np.exp(2j * np.pi * (signal_freq_offset * t + 0.1 * mod_signal))
                else:
                    # AM
                    mod_signal = 0.5 + 0.5 * np.sin(2 * np.pi * 1000 * t)
                    signal = mod_signal * np.exp(2j * np.pi * signal_freq_offset * t)
                
                # Añadir a las muestras con potencia ajustada
                samples += signal_power * signal
            
            # Ocasionalmente añadir interferencia
            if np.random.random() < 0.1:
                interference = 0.7 * np.exp(2j * np.pi * np.random.uniform(-sample_rate/4, sample_rate/4) * t)
                samples += interference
        
        return samples
    
    def stop_monitoring(self):
        """Detiene el monitoreo de señales"""
        if not self.monitoring:
            self.statusBar.showMessage("El monitoreo no está en curso")
            return
        
        self.statusBar.showMessage("Deteniendo monitoreo...")
        self.monitoring = False
        
        # Esperar a que termine el hilo
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        self.status_label.setText("Estado: Inactivo")
        self.statusBar.showMessage("Monitoreo detenido", 3000)
    
    def _generate_security_alert(self, signal):
        """Genera una alerta de seguridad basada en una señal sospechosa"""
        # Crear mensaje de alerta
        timestamp = signal.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        if signal.satellite_id:
            satellite = self.db.get_satellite(signal.satellite_id)
            satellite_name = satellite.name if satellite else f"NORAD ID {signal.satellite_id}"
            description = f"Señal sospechosa detectada en {signal.center_frequency/1e6:.3f} MHz asociada a satélite {satellite_name}"
        else:
            description = f"Señal sospechosa detectada en {signal.center_frequency/1e6:.3f} MHz"
        
        # Añadir detalles
        if signal.is_encrypted:
            description += " - Encriptada"
        if signal.is_anomalous:
            description += " - Patrón anómalo"
        
        # Registrar en la base de datos
        self.db.log_security_event(
            timestamp=signal.timestamp,
            event_type="SUSPICIOUS_SIGNAL",
            description=description,
            satellite_id=signal.satellite_id,
            signal_id=signal.id,
            severity=signal.threat_level.name
        )
        
        # Añadir a la UI
        self.message_queue.put(("new_alert", f"[{timestamp}] [{signal.threat_level.name}] {description}"))
    
    def export_data(self):
        """Exporta datos recopilados a varios formatos"""
        # Crear menú de opciones de exportación
        export_menu = QtWidgets.QMenu(self)
        
        # Opciones
        export_signals_json = export_menu.addAction("Exportar señales a JSON")
        export_signals_csv = export_menu.addAction("Exportar señales a CSV")
        export_menu.addSeparator()
        export_satellites_json = export_menu.addAction("Exportar satélites a JSON")
        export_satellites_csv = export_menu.addAction("Exportar satélites a CSV")
        export_menu.addSeparator()
        export_alerts = export_menu.addAction("Exportar alertas a CSV")
        export_menu.addSeparator()
        export_report = export_menu.addAction("Generar informe completo")
        
        # Mostrar menú
        action = export_menu.exec_(QtGui.QCursor.pos())
        
        # Procesar acción seleccionada
        if action == export_signals_json:
            self._export_signals_json()
        elif action == export_signals_csv:
            self._export_signals_csv()
        elif action == export_satellites_json:
            self._export_satellites_json()
        elif action == export_satellites_csv:
            self._export_satellites_csv()
        elif action == export_alerts:
            self._export_alerts_csv()
        elif action == export_report:
            self._generate_report()
    
    def _export_signals_json(self):
        """Exporta señales detectadas a formato JSON"""
        # Implementación básica para ejemplo
        try:
            signals = self.db.get_recent_signals(limit=1000)
            
            if not signals:
                self.statusBar.showMessage("No hay señales para exportar", 3000)
                return
            
            # Crear directorio si no existe
            os.makedirs(self.config.get('General', 'DataDirectory', fallback=DATA_DIR), exist_ok=True)
            
            # Nombre de archivo con timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.config.get('General', 'DataDirectory', fallback=DATA_DIR),
                f"signals_export_{timestamp}.json"
            )
            
            # Convertir a formato serializable
            signals_data = []
            for signal in signals:
                signals_data.append({
                    "id": signal.id,
                    "center_frequency": signal.center_frequency,
                    "bandwidth": signal.bandwidth,
                    "power": signal.power,
                    "timestamp": signal.timestamp.isoformat(),
                    "satellite_id": signal.satellite_id,
                    "modulation": signal.modulation.name,
                    "signal_type": signal.signal_type.name,
                    "snr": signal.snr,
                    "duration": signal.duration,
                    "is_encrypted": signal.is_encrypted,
                    "is_anomalous": signal.is_anomalous,
                    "threat_level": signal.threat_level.name,
                    "detection_confidence": signal.detection_confidence,
                    "notes": signal.notes
                })
            
            # Guardar a archivo
            with open(filename, 'w') as f:
                json.dump(signals_data, f, indent=2)
            
            self.statusBar.showMessage(f"Señales exportadas a {filename}", 3000)
            
        except Exception as e:
            logger.error(f"Error al exportar señales a JSON: {e}")
            self.statusBar.showMessage(f"Error al exportar señales: {e}", 3000)
    
    def _export_signals_csv(self):
        """Exporta señales detectadas a formato CSV"""
        # Implementación similar a _export_signals_json pero en formato CSV
        pass
    
    def _export_satellites_json(self):
        """Exporta información de satélites a formato JSON"""
        # Implementación similar a _export_signals_json pero para satélites
        pass
    
    def _export_satellites_csv(self):
        """Exporta información de satélites a formato CSV"""
        # Implementación similar a _export_signals_json pero en formato CSV para satélites
        pass
    
    def _export_alerts_csv(self):
        """Exporta alertas de seguridad a formato CSV"""
        # Implementación similar para alertas
        pass
    
    def _generate_report(self):
        """Genera un informe completo en formato HTML"""
        # Implementación de un informe HTML completo
        pass
    
    def _update_signals_table(self):
        """Actualiza la tabla de señales detectadas"""
        # Obtener señales recientes
        signals = self.db.get_recent_signals(limit=100)
        
        # Actualizar tabla
        self.signals_table.setRowCount(len(signals))
        
        for row, signal in enumerate(signals):
            # ID
            self.signals_table.setItem(row, 0, QtWidgets.QTableWidgetItem(signal.id[:8] + "..."))
            
            # Frecuencia
            freq_item = QtWidgets.QTableWidgetItem(f"{signal.center_frequency/1e6:.6f}")
            self.signals_table.setItem(row, 1, freq_item)
            
            # Ancho de banda
            bw_item = QtWidgets.QTableWidgetItem(f"{signal.bandwidth/1e3:.1f}")
            self.signals_table.setItem(row, 2, bw_item)
            
            # Potencia
            power_item = QtWidgets.QTableWidgetItem(f"{signal.power:.1f}")
            self.signals_table.setItem(row, 3, power_item)
            
            # Modulación
            mod_item = QtWidgets.QTableWidgetItem(signal.modulation.name)
            self.signals_table.setItem(row, 4, mod_item)
            
            # Tipo
            type_item = QtWidgets.QTableWidgetItem(signal.signal_type.name)
            self.signals_table.setItem(row, 5, type_item)
            
            # Satélite
            if signal.satellite_id:
                satellite = self.db.get_satellite(signal.satellite_id)
                sat_name = satellite.name if satellite else f"NORAD ID {signal.satellite_id}"
                sat_item = QtWidgets.QTableWidgetItem(sat_name)
            else:
                sat_item = QtWidgets.QTableWidgetItem("Desconocido")
            self.signals_table.setItem(row, 6, sat_item)
            
            # Amenaza
            threat_item = QtWidgets.QTableWidgetItem(signal.threat_level.name)
            
            # Colorear según nivel de amenaza
            if signal.threat_level == ThreatLevel.HIGH:
                threat_item.setBackground(QtGui.QColor(255, 150, 150))
            elif signal.threat_level == ThreatLevel.CRITICAL:
                threat_item.setBackground(QtGui.QColor(255, 100, 100))
            elif signal.threat_level == ThreatLevel.MEDIUM:
                threat_item.setBackground(QtGui.QColor(255, 200, 150))
            
            self.signals_table.setItem(row, 7, threat_item)
    
    def _update_satellites_table(self):
        """Actualiza la tabla de satélites"""
        # Obtener satélites visibles
        visible_satellites = self.satellite_tracker.visible_satellites
        
        # Lista para almacenar filas
        rows = []
        
        # Añadir satélites visibles
        for norad_id in visible_satellites:
            satellite = self.db.get_satellite(norad_id)
            
            if not satellite:
                # Intentar obtener información básica del TLE
                satellite = self.tle_manager.export_satellite_info(norad_id)
                if satellite:
                    self.db.add_satellite(satellite)
            
            if satellite:
                rows.append({
                    "norad_id": norad_id,
                    "name": satellite.name,
                    "type": satellite.type.name,
                    "country": satellite.country,
                    "visible": True,
                    "last_signal": "N/A"  # Se actualizaría con datos reales
                })
        
        # Añadir otros satélites de interés (implementación simplificada)
        # En una implementación completa, se obtendrían de la base de datos
        
        # Actualizar tabla
        self.satellites_table.setRowCount(len(rows))
        
        for row, sat_data in enumerate(rows):
            # NORAD ID
            self.satellites_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(sat_data["norad_id"])))
            
            # Nombre
             self.satellites_table.setItem(row, 1, QtWidgets.QTableWidgetItem(sat_data["name"]))
            
            # Tipo
            self.satellites_table.setItem(row, 2, QtWidgets.QTableWidgetItem(sat_data["type"]))
            
            # País
            self.satellites_table.setItem(row, 3, QtWidgets.QTableWidgetItem(sat_data["country"]))
            
            # Visible
            visible_item = QtWidgets.QTableWidgetItem("Sí" if sat_data["visible"] else "No")
            if sat_data["visible"]:
                visible_item.setBackground(QtGui.QColor(200, 255, 200))
            self.satellites_table.setItem(row, 4, visible_item)
            
            # Última señal
            self.satellites_table.setItem(row, 5, QtWidgets.QTableWidgetItem(sat_data["last_signal"]))
    
    def _clear_plots(self):
        """Limpia los gráficos de espectro y espectrograma"""
        # Limpiar gráfico de espectro
        self.spectrum_figure.clear()
        self.spectrum_canvas.draw()
        
        # Limpiar gráfico de espectrograma
        self.spectrogram_figure.clear()
        self.spectrogram_canvas.draw()
    
    def _update_spectrum_plot(self, data):
        """Actualiza el gráfico de espectro"""
        freq_range, spectrum = data
        
        # Convertir a dB
        spectrum_db = 10 * np.log10(spectrum + 1e-10)
        
        # Crear gráfico
        self.spectrum_figure.clear()
        ax = self.spectrum_figure.add_subplot(111)
        ax.plot(freq_range, spectrum_db)
        ax.set_xlabel('Frecuencia (MHz)')
        ax.set_ylabel('Potencia (dB)')
        ax.set_title('Espectro de RF en Tiempo Real')
        ax.grid(True)
        
        # Ajustar límites
        ax.set_ylim([-80, 10])
        
        # Actualizar canvas
        self.spectrum_canvas.draw()
    
    def _update_spectrogram_plot(self, data):
        """Actualiza el gráfico de espectrograma"""
        freq_range, spectrogram_data = data
        
        if not spectrogram_data:
            return
            
        # Crear gráfico
        self.spectrogram_figure.clear()
        ax = self.spectrogram_figure.add_subplot(111)
        
        # Crear imagen de espectrograma
        extent = [freq_range[0], freq_range[-1], 0, len(spectrogram_data)]
        im = ax.imshow(spectrogram_data, aspect='auto', origin='lower', 
                      extent=extent, cmap='viridis', vmin=-80, vmax=0)
        
        ax.set_xlabel('Frecuencia (MHz)')
        ax.set_ylabel('Tiempo (muestras)')
        ax.set_title('Espectrograma')
        
        # Añadir barra de color
        self.spectrogram_figure.colorbar(im, ax=ax, label='Potencia (dB)')
        
        # Actualizar canvas
        self.spectrogram_canvas.draw()
    
    def _add_signal_to_table(self, signal):
        """Añade una nueva señal detectada a la tabla"""
        # Insertar al principio
        self.signals_table.insertRow(0)
        
        # ID
        self.signals_table.setItem(0, 0, QtWidgets.QTableWidgetItem(signal.id[:8] + "..."))
        
        # Frecuencia
        freq_item = QtWidgets.QTableWidgetItem(f"{signal.center_frequency/1e6:.6f}")
        self.signals_table.setItem(0, 1, freq_item)
        
        # Ancho de banda
        bw_item = QtWidgets.QTableWidgetItem(f"{signal.bandwidth/1e3:.1f}")
        self.signals_table.setItem(0, 2, bw_item)
        
        # Potencia
        power_item = QtWidgets.QTableWidgetItem(f"{signal.power:.1f}")
        self.signals_table.setItem(0, 3, power_item)
        
        # Modulación
        mod_item = QtWidgets.QTableWidgetItem(signal.modulation.name)
        self.signals_table.setItem(0, 4, mod_item)
        
        # Tipo
        type_item = QtWidgets.QTableWidgetItem(signal.signal_type.name)
        self.signals_table.setItem(0, 5, type_item)
        
        # Satélite
        if signal.satellite_id:
            satellite = self.db.get_satellite(signal.satellite_id)
            sat_name = satellite.name if satellite else f"NORAD ID {signal.satellite_id}"
            sat_item = QtWidgets.QTableWidgetItem(sat_name)
        else:
            sat_item = QtWidgets.QTableWidgetItem("Desconocido")
        self.signals_table.setItem(0, 6, sat_item)
        
        # Amenaza
        threat_item = QtWidgets.QTableWidgetItem(signal.threat_level.name)
        
        # Colorear según nivel de amenaza
        if signal.threat_level == ThreatLevel.HIGH:
            threat_item.setBackground(QtGui.QColor(255, 150, 150))
        elif signal.threat_level == ThreatLevel.CRITICAL:
            threat_item.setBackground(QtGui.QColor(255, 100, 100))
        elif signal.threat_level == ThreatLevel.MEDIUM:
            threat_item.setBackground(QtGui.QColor(255, 200, 150))
        
        self.signals_table.setItem(0, 7, threat_item)
        
        # Limitar a 100 filas
        if self.signals_table.rowCount() > 100:
            self.signals_table.removeRow(100)
    
    def _add_alert_to_list(self, alert_text):
        """Añade una alerta a la lista de alertas"""
        item = QtWidgets.QListWidgetItem(alert_text)
        
        # Colorear según nivel de amenaza
        if "CRITICAL" in alert_text:
            item.setBackground(QtGui.QColor(255, 100, 100))
        elif "HIGH" in alert_text:
            item.setBackground(QtGui.QColor(255, 150, 150))
        elif "MEDIUM" in alert_text:
            item.setBackground(QtGui.QColor(255, 200, 150))
        
        # Insertar al principio
        self.alerts_list.insertItem(0, item)
    
    def _process_queue(self):
        """Procesa mensajes de la cola"""
        try:
            while not self.message_queue.empty():
                message_type, data = self.message_queue.get_nowait()
                
                if message_type == "status":
                    self.statusBar.showMessage(data, 3000)
                elif message_type == "update_spectrum":
                    self._update_spectrum_plot(data)
                elif message_type == "update_spectrogram":
                    self._update_spectrogram_plot(data)
                elif message_type == "new_signal":
                    self._add_signal_to_table(data)
                elif message_type == "new_alert":
                    self._add_alert_to_list(data)
                elif message_type == "update_satellites":
                    self._update_satellites_table()
                    self.satellites_label.setText(f"Satélites visibles: {len(self.satellite_tracker.visible_satellites)}")
                elif message_type == "monitoring_stopped":
                    self.monitoring = False
                    self.status_label.setText("Estado: Inactivo")
                    self.statusBar.showMessage("Monitoreo detenido", 3000)
        except Exception as e:
            logger.error(f"Error al procesar cola de mensajes: {e}")
    
    def closeEvent(self, event):
        """Maneja el cierre de la aplicación"""
        # Detener monitoreo si está activo
        if self.monitoring:
            self.stop_monitoring()
        
        # Cerrar base de datos
        self.db.close()
        
        # Aceptar evento de cierre
        event.accept()

def main():
    """Función principal"""
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="SatGuardian - Sistema de Monitoreo y Análisis de Seguridad Satelital")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Ruta al archivo de configuración")
    parser.add_argument("--debug", action="store_true", help="Activar modo de depuración")
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear directorios necesarios
    for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, TLE_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Iniciar aplicación Qt
    app = QtWidgets.QApplication(sys.argv)
    window = SatGuardianApp(config_path=args.config)
    window.show()
    
    # Ejecutar bucle de eventos
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
​




        
